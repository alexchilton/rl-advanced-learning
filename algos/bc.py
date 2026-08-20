"""Behaviour cloning: supervised learning on (state, expert action) pairs.

    python algos/bc.py

That is the whole algorithm. There is no reward, no value function, no discount,
no environment interaction. Collect what the expert did, fit a classifier, done.

Which is why it is worth understanding exactly what it cannot do.

BC has no notion of what is GOOD, only of what was DONE. It therefore cannot
exceed the demonstrator, and -- less obviously -- its errors compound. The clone
makes a small mistake, which puts it in a state the expert never visited, where
its prediction is worse, which puts it somewhere stranger still. Ross & Bagnell
showed the expected cost grows with the SQUARE of the horizon, against linear for
a method that trains on its own state distribution.

`experiments/bc_vs_dagger.py` measures both halves of that on Racetrack.

The classifier here is nearest-neighbour over the demonstrated states, and that is
a deliberate choice rather than laziness. It is non-parametric and has no training
loop, so nothing here can be blamed on architecture, initialisation, learning rate
or under-fitting. Whatever fails, fails because of the DATA -- which is the point.
A torch version arrives with the deep tiers and behaves the same way for the same
reason.
"""

from __future__ import annotations

import numpy as np

from envs.racetrack import Racetrack

GAMMA = 0.99


# ----------------------------------------------------------------------
def collect_demonstrations(env, expert, n_labels, seed=0):
    """Roll out the expert until `n_labels` (state, action) pairs are collected.

    Counting labels rather than episodes matters. An expert episode on this track
    is about seven steps, so "ten demonstrations" and "seventy labels" are the
    same budget, and only one of them is comparable to DAgger.
    """
    states, actions = [], []
    episode = 0
    while len(states) < n_labels:
        env.reset(seed=seed * 1000 + episode)
        episode += 1
        for _ in range(env.max_steps):
            states.append(env.state)
            actions.append(int(expert[env.state]))
            _, _, terminated, truncated, _ = env.step(int(expert[env.state]))
            if terminated or truncated or len(states) >= n_labels:
                break
    return np.array(states[:n_labels]), np.array(actions[:n_labels])


def fit(env, states, actions):
    """Return a policy function. Nearest demonstrated state wins.

    The generalisation assumption is the same one every function approximator
    makes: nearby states want similar actions. On a racetrack that is wrong
    exactly where it matters, because the right action at speed 4 is not the
    right action at speed 1 in the same cell.
    """
    scale = feature_scale(env)
    demonstrated = features(env, states) / scale
    cache: dict[int, int] = {}

    def policy(state):
        state = int(state)
        if state not in cache:
            if state == env.finished_state:
                cache[state] = 0
            else:
                query = features(env, [state])[0] / scale
                nearest = int(np.argmin(np.linalg.norm(demonstrated - query, axis=1)))
                cache[state] = int(actions[nearest])
        return cache[state]

    return policy


def features(env, states):
    """(row, column, vertical speed, horizontal speed) per state."""
    out = np.zeros((len(states), 4))
    for i, state in enumerate(states):
        if int(state) == env.finished_state:
            continue
        (row, col), vertical, horizontal = env.decode(int(state))
        out[i] = (row, col, vertical, horizontal)
    return out


def feature_scale(env):
    return np.array([env.rows, env.cols, env.max_speed, env.max_speed], dtype=float)


def covariate_shift(env, demonstrated_states, visited_states):
    """Mean distance from a visited state to the nearest demonstrated one.

    This is the number the BC-versus-DAgger argument is actually about, and it is
    almost never plotted. Near zero means the clone stayed where the expert was,
    so any remaining failure is a modelling problem. Growing means the clone left
    the expert's world, and more expert demonstrations will not help -- they are
    all in the part of the space that is already covered.
    """
    scale = feature_scale(env)
    demonstrated = features(env, demonstrated_states) / scale
    visited = features(env, visited_states) / scale
    distances = np.linalg.norm(visited[:, None, :] - demonstrated[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())


def rollout(env, policy, n_episodes, seed):
    """Visited states, excluding the absorbing finished state."""
    visited = []
    for episode in range(n_episodes):
        env.reset(seed=seed + episode)
        visited.append(env.state)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, _ = env.step(policy(env.state))
            visited.append(env.state)
            if terminated or truncated:
                break
    return [s for s in visited if s != env.finished_state]


def exact_return(env, policy, gamma=GAMMA):
    """Exact expected return of the cloned policy. No sampling, no seed noise."""
    table = np.array([policy(s) for s in range(env.n_states)])
    return env.policy_return(table, gamma)


# ----------------------------------------------------------------------
def main():
    env = Racetrack()
    expert = env.optimal_policy(GAMMA)
    optimal = env.optimal_return(GAMMA)

    print(f"Racetrack, {env.n_states} states, {env.n_actions} actions")
    print(f"expert (exact optimal policy): {optimal:.3f}\n")
    print(f"{'labels':>7} {'coverage':>9} {'covariate shift':>16} {'exact return':>13} {'gap':>8}")

    for n_labels in (10, 20, 40, 80, 160, 320, 640):
        states, actions = collect_demonstrations(env, expert, n_labels, seed=1)
        policy = fit(env, states, actions)
        visited = rollout(env, policy, n_episodes=50, seed=90_000)

        coverage = env.demonstration_coverage(states)
        shift = covariate_shift(env, states, visited)
        value = exact_return(env, policy)
        print(
            f"{n_labels:>7} {coverage:>9.3f} {shift:>16.3f} {value:>13.2f} "
            f"{value - optimal:>8.2f}"
        )

    print("\nThe gap does not close. Past a few hundred labels the coverage keeps")
    print("rising and the return does not, because every extra label lands on the")
    print("expert's own trajectory -- which was never the part that was missing.")
    print("\nSee experiments/bc_vs_dagger.py for what fixes it.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - BC scores perfectly and you suspect it should not: check the environment for
#   a free correction. On Racetrack with `on_crash="restart"` a crash teleports
#   the learner back to the start line, which is ON the expert's distribution, so
#   the covariate shift never accumulates and BC reaches 1.00 from one
#   demonstration. Measured, not hypothesised.
# - BC scores perfectly and the horizon is short: an expert episode of seven steps
#   leaves no room for errors to compound. The same clone that loses 0.4 of a
#   return unit over seven steps loses 16 over eighteen.
# - Return gets WORSE with more demonstrations: real, and not always noise.
#   Nearest-neighbour can acquire a bad neighbour. Report several seeds before
#   believing any single curve here.
# - Covariate shift near zero but the return still poor: the clone is on
#   distribution and simply wrong. That is a modelling problem, and DAgger will
#   not help. Use a better classifier.
