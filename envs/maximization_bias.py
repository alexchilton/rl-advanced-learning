"""The maximisation-bias MDP (Sutton & Barto, Example 6.7).

    A --right--> end, reward 0
    A --left---> B,   reward 0
    B --any of k actions--> end, reward -0.1 + 1 or -0.1 - 1, each with probability 1/2

Going left is worth -0.1 in expectation whichever action is taken at B, so right
is optimal. But Q-learning estimates max_b Q(B, b), and the max of k noisy
estimates of -0.1 is positive for a long time. So Q-learning goes left far more
often than epsilon-greedy exploration alone would. Double Q-learning does not.

The textbook draws B's rewards from N(-0.1, 1). Here they are ±1 around -0.1 with
equal probability -- same mean, same variance -- so the MDP fits the finite
transition table and the exact solvers apply.
"""

from __future__ import annotations

import numpy as np

from envs.tabular import TabularEnv

A, B, END = 0, 1, 2
LEFT, RIGHT = 0, 1


class MaximizationBias(TabularEnv):
    def __init__(self, n_b_actions: int = 10, mean: float = -0.1, spread: float = 1.0):
        n_actions = max(2, n_b_actions)
        P = {s: {a: [] for a in range(n_actions)} for s in range(3)}
        for a in range(n_actions):
            # Actions beyond left/right at A behave like right; only B uses all k.
            P[A][a] = [(1.0, B, 0.0, False)] if a == LEFT else [(1.0, END, 0.0, True)]
            P[B][a] = [(0.5, END, mean + spread, True), (0.5, END, mean - spread, True)]
            P[END][a] = [(1.0, END, 0.0, True)]
        super().__init__(n_states=3, n_actions=n_actions, P=P,
                         start_distribution=np.array([1.0, 0.0, 0.0]), max_steps=10)
        self.n_b_actions = n_b_actions
