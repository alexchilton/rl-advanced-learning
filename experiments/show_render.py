"""What the terminal diagnostics look like, using a bug you can see.

    python experiments/show_render.py

There is no agent here yet. To have something wrong to look at, this script builds
two value functions by hand:

  - the exact one, from value iteration
  - the same computation with the terminal mask removed, which is bug #1 in the
    catalogue and the most common bug in deep RL

Then it puts them side by side. The point of the exercise is the third panel. A
missing terminal mask does not produce noise; it produces a systematic bias, and a
systematic bias is one flat colour across the whole grid. Once you have seen that
shape you recognise it immediately, and you stop trying to fix it with the
learning rate.
"""

from __future__ import annotations

import numpy as np

from diagnostics.render import (
    compare_grids,
    legend,
    policy_overlay,
    sparkline,
    trajectory,
    value_heatmap,
    visit_heatmap,
)
from envs import solvers
from envs.cliff import CliffWalking
from envs.gridworld import GridWorld

GAMMA = 0.9


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def unmasked_backup(model, n_states, n_actions, gamma, sweeps=2000):
    """Value iteration with the `* continues` factor deleted. Bug #1, on purpose."""
    V = np.zeros(n_states)
    for _ in range(sweeps):
        backup = model.reward + gamma * V[model.next_state]  # <- no `* model.continues`
        Q = np.bincount(
            model.sa, weights=model.prob * backup, minlength=n_states * n_actions
        ).reshape(n_states, n_actions)
        V = Q.max(axis=1)
    return V


def auto_reset_model(env):
    """`P` as an auto-resetting vectorised environment hands it to you.

    This is the realistic shape of bug #1. A vectorised environment resets on your
    behalf, so the observation you receive after a terminal step already belongs
    to the NEXT episode. Store that as `next_obs`, forget to mask, and the backup
    bootstraps the value of the start state into the value of finishing.

    The episodic problem quietly becomes a continuing one, and the values stop
    meaning "return for this episode".
    """
    start = int(np.argmax(env.start_distribution))
    P = {
        s: {
            a: [
                (prob, start if terminated else s2, reward, terminated)
                for prob, s2, reward, terminated in env.P[s][a]
            ]
            for a in range(env.n_actions)
        }
        for s in range(env.n_states)
    }
    return solvers.flatten(P, env.n_states, env.n_actions)


def main() -> None:
    rule("colour key")
    print(legend())

    env = GridWorld(size=7, reward_mode="sparse", walls=[(2, 2), (2, 3), (3, 2), (4, 5)])

    rule("exact V*, and the greedy policy that comes with it")
    print(value_heatmap(env, env.true_v(GAMMA), "V*"))
    print()
    print(policy_overlay(env, env.true_q(GAMMA), "Q* with greedy arrows"))

    rule("bug #1a: masking removed, absorbing terminal state -- and nothing happens")
    truth = env.true_v(GAMMA)
    naive = unmasked_backup(solvers.flatten(env.P, env.n_states, env.n_actions),
                            env.n_states, env.n_actions, GAMMA)
    print(f"max |error| with the mask deleted: {np.abs(naive - truth).max():.6f}")
    print("\nZero. Deleting the mask changed nothing, and that is worth understanding")
    print("before trusting any tabular test of it.")
    print("\nIn this table the goal state is absorbing with zero reward, so V(goal) is")
    print("0 whether you mask or not, and gamma * 0 is 0 either way. The mask only")
    print("matters when the value AFTER termination is non-zero -- which is exactly")
    print("what a neural network gives you, since it will happily predict some")
    print("arbitrary value for a state the episode never continues from.")

    rule("bug #1b: the same bug in its realistic form -- auto-reset")
    broken = unmasked_backup(auto_reset_model(env), env.n_states, env.n_actions, GAMMA)
    print(compare_grids(env, broken, truth, gamma_label=f"(gamma={GAMMA})"))
    print("\nA vectorised environment resets for you, so the observation after a")
    print("terminal step belongs to the next episode. Unmasked, reaching the goal now")
    print("bootstraps the value of starting again, and the agent is being scored on an")
    print("infinite sequence of episodes rather than this one.")
    print("\nThe error panel is one colour, all the same sign. That is a systematic")
    print("bias, and no learning rate will fix it. Contrast it with the next panel.")

    rule("for comparison: unbiased noise on top of the exact values")
    rng = np.random.default_rng(0)
    noisy = env.true_v(GAMMA) + rng.normal(0, 0.05, env.n_states)
    print(compare_grids(env, noisy, env.true_v(GAMMA), gamma_label=f"(gamma={GAMMA})"))
    print("\nSpeckled, both signs, no structure. That is an agent that needs more data.")
    print("The panel above is an agent that needs a different line of code.")

    rule("where a random policy actually goes")
    counts = random_walk_visits(env, episodes=400, seed=0)
    print(visit_heatmap(env, counts, "visits under a uniform random policy"))
    print("\nHalf of 'it is not learning' is 'it never went there'. On a 7x7 grid")
    print(f"the goal was reached {int(counts[env.index(env.goal)])} times in 400 episodes.")

    rule("one trajectory, numbered by step")
    states = greedy_rollout(env, env.true_q(GAMMA))
    print(trajectory(env, states, f"optimal path, {len(states) - 1} steps"))

    rule("the cliff, which is wider than it is tall")
    cliff = CliffWalking()
    print(policy_overlay(cliff, cliff.true_q(1.0), "Q* on CliffWalking (gamma=1)", width=7))

    rule("sparklines, for learning curves in a log")
    steps = np.arange(500)
    print("healthy:   " + sparkline(1 - np.exp(-steps / 120) + rng.normal(0, 0.02, 500)))
    print("collapsed: " + sparkline(np.where(steps < 300, steps / 300, 0.1)))
    print("flat:      " + sparkline(np.zeros(500)))
    print()


def random_walk_visits(env, episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    counts = np.zeros(env.n_states)
    for episode in range(episodes):
        env.reset(seed=seed * 10_000 + episode)
        counts[env.state] += 1
        while True:
            _, _, terminated, truncated, _ = env.step(int(rng.integers(0, env.n_actions)))
            counts[env.state] += 1
            if terminated or truncated:
                break
    return counts


def greedy_rollout(env, Q, max_steps: int = 100):
    policy = np.asarray(Q).argmax(axis=1)
    env.reset(seed=0)
    states = [env.state]
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(int(policy[env.state]))
        states.append(env.state)
        if terminated or truncated:
            break
    return states


if __name__ == "__main__":
    main()
