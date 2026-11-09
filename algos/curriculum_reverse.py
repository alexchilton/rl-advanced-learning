"""Reverse curriculum (Florensa, Held, Wulfmeier, Zhang & Abbeel, 2017).

    python algos/curriculum_reverse.py

On a sparse task the agent learns nothing until it reaches the goal once. So
start it next to the goal, where it reaches it by accident, then move the start
outward as it gets good:

    stage d: start in a random state within d steps of the goal
             train until greedy rollouts from those starts succeed >= 80% of the time
             then d += 1, until the real start is included

Every early episode sees reward, so there is always something to learn from, and
each new ring of start states sits next to states whose value is already known.

The catch, and it is a real one: this needs the ability to reset the environment
to an arbitrary state. A simulator can; most real systems cannot.

The learner is `algos/count_based_exploration.train` with no bonus, i.e. plain
epsilon-greedy Q-learning. Only the start distribution changes.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from algos import count_based_exploration as learner
from envs.chain import Chain

GAMMA = 0.99
CHUNK = 500
SUCCESS = 0.8


def distance_to_goal(env):
    """Fewest steps from each state to a terminating transition, by BFS on env.P.
    Unreachable states get a large number, and so do absorbing states (every
    action loops back, terminated): GridWorld's goal cell is one, and starting
    an episode there "succeeds" in one step with no reward."""
    n = env.n_states
    predecessors = {s: set() for s in range(n)}
    goal_adjacent = set()
    for s in range(n):
        if all(s2 == s and terminated for a in range(env.n_actions) for _, s2, _, terminated in env.P[s][a]):
            continue
        for a in range(env.n_actions):
            for prob, s2, _, terminated in env.P[s][a]:
                if prob > 0 and terminated:
                    goal_adjacent.add(s)
                elif prob > 0:
                    predecessors[s2].add(s)
    distance = np.full(n, 10**6)
    queue = deque()
    for s in goal_adjacent:
        distance[s] = 1
        queue.append(s)
    while queue:
        s = queue.popleft()
        for p in predecessors[s]:
            if distance[p] > distance[s] + 1:
                distance[p] = distance[s] + 1
                queue.append(p)
    return distance


def greedy_success(env, Q, starts, rng, episodes=20):
    """Fraction of greedy rollouts from random `starts` that reach the goal in time."""
    wins = 0
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(2**31 - 1)))
        env.state = int(rng.choice(starts))
        while True:
            _, _, terminated, truncated, _ = env.step(int(Q[env.state].argmax()))
            if terminated or truncated:
                wins += terminated
                break
    return wins / episodes


def train(env, n_steps=200_000, chunk=CHUNK, success=SUCCESS, gamma=GAMMA, eval_every=1_000, seed=0, **kwargs):
    """Returns (Q, curve, stages). `curve` is (steps, exact greedy return from the real
    start); `stages` is (steps when the start radius grew, new radius)."""
    rng = np.random.default_rng(seed + 1)
    distance = distance_to_goal(env)
    real_start = int(np.argmax(env.start_distribution))
    final_radius = int(distance[real_start])
    radius, Q, steps, stages = 1, None, 0, [(0, 1)]
    curve = [(0, env.policy_return(np.zeros(env.n_states, dtype=int), gamma))]

    while steps < n_steps:
        starts = np.flatnonzero(distance <= radius)
        Q, _, _ = learner.train(env, n_steps=chunk, gamma=gamma, reset=lambda e, r: r.choice(starts),
                                eval_every=chunk, seed=int(rng.integers(2**31 - 1)), Q=Q, **kwargs)
        steps += chunk
        if steps % eval_every < chunk:
            curve.append((steps, env.policy_return(Q.argmax(axis=1), gamma)))
        if radius < final_radius and greedy_success(env, Q, starts, rng) >= success:
            radius += 1
            stages.append((steps, radius))
    return Q, curve, stages


def main():
    for n in (20, 30):
        env = Chain(n=n, max_steps=4 * n)
        optimal = env.optimal_return(GAMMA)
        print(f"Chain(n={n}), exact optimum {optimal:.4f}")
        for seed in range(3):
            _, curve, stages = train(env, n_steps=60_000, seed=seed)
            solved = next((s for s, v in curve if v >= 0.95 * optimal), None)
            print(f"  seed {seed}: radius reached {stages[-1][1]} by step {stages[-1][0]:,}; "
                  f"greedy policy from the real start at 95% of optimum by step {solved}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The radius stops growing: the success threshold is unreachable with this
#   epsilon or step budget per chunk. Print the success rate per stage.
# - It grows and the old stages are forgotten: with a table that cannot happen,
#   but with a network it does. Re-evaluate the earlier rings after each stage.
# - It needs a simulator you can set the state of. If you cannot reset to an
#   arbitrary state, this method is not available, and HER or a bonus is.
