"""Bit flipping (Andrychowicz et al. 2017, section 3.1): the HER paper's motivating task.

The state is n bits; the goal is another n bits, drawn at random each episode.
Each action flips one bit. Reward is -1 every step until the state equals the
goal, then 0 and the episode ends. Episodes last n steps.

A random policy matches a random n-bit goal with probability about 2^-n per step,
so from n around 15 a standard agent essentially never sees a 0 reward, and
every target it computes says "everything is worth the same". The task is
trivial for any agent that knows the goal is the point: flip the bits that differ.

The observation is the state and the goal concatenated, so a goal-conditioned
network can read both. `compute_reward(achieved, goal)` is exposed because HER
must recompute rewards for goals the episode never had.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class BitFlip:
    def __init__(self, n_bits: int = 10):
        self.n = int(n_bits)
        self.action_space = SimpleNamespace(n=self.n)
        self.rng = np.random.default_rng()

    @staticmethod
    def compute_reward(achieved, goal):
        return 0.0 if np.array_equal(achieved, goal) else -1.0

    def observation(self):
        return np.concatenate([self.state, self.goal]).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.integers(0, 2, self.n).astype(np.float32)
        self.goal = self.rng.integers(0, 2, self.n).astype(np.float32)
        while np.array_equal(self.state, self.goal):
            self.goal = self.rng.integers(0, 2, self.n).astype(np.float32)
        self.t = 0
        return self.observation(), {}

    def step(self, action):
        self.state[int(action)] = 1.0 - self.state[int(action)]
        self.t += 1
        reward = self.compute_reward(self.state, self.goal)
        terminated = reward == 0.0
        truncated = not terminated and self.t >= self.n
        return self.observation(), reward, terminated, truncated, {}
