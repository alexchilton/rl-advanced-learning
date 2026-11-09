"""Observation normalisation on PointMass: same agent, raw vs normalised inputs.

    python experiments/obs_normalisation.py [--resume]

`envs/pointmass.py` is badly scaled on purpose: position stays in [-1, 1] while
velocity reaches tens. A network fed that raw has to learn first-layer weights
of very different sizes before it can learn anything else.

The fix in deep RL is a running mean/std normaliser on the observation, the
input-side relative of batch normalisation (Ioffe & Szegedy 2015, which
normalises hidden activations per batch). Here: TD3, 30,000 steps, 3 seeds,
raw against normalised, scored against the hand-written PD controller that
`PointMass.expert_action` provides.

One JSON per run in experiments/results/normalisation/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "normalisation"
STEPS = 30_000
SEEDS = range(3)


class RunningNorm:
    """Running mean and variance (Welford), shared between training and evaluation."""

    def __init__(self, dim):
        self.n, self.mean, self.m2, self.frozen = 0, np.zeros(dim), np.zeros(dim), False

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.frozen:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)
        std = np.sqrt(self.m2 / max(self.n - 1, 1)) + 1e-8
        return ((x - self.mean) / std).clip(-10, 10).astype(np.float32)


class Adapter:
    """PointMass with Gymnasium-style spaces, optional observation normaliser."""

    def __init__(self, env, norm=None):
        self.env, self.norm = env, norm
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(shape=(2,), high=np.ones(2, dtype=np.float32))

    def _obs(self, obs):
        return self.norm(obs) if self.norm is not None else np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(obs), reward, terminated, truncated, info


def expert_return(episodes=5, seed=10_000):
    from envs.pointmass import PointMass

    total = 0.0
    env = PointMass()
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.expert_action())
            total += reward
            done = terminated or truncated
    return total / episodes


def run(normalise, seed):
    import torch

    from algos import td3
    from algos.ddpg import evaluate_gym
    from envs.pointmass import PointMass

    torch.set_num_threads(1)
    norm = RunningNorm(4) if normalise else None

    def make_eval():
        if norm is not None:
            norm.frozen = True
        return Adapter(PointMass(), norm)

    def evaluate(actor):
        score = evaluate_gym(make_eval)(actor)
        if norm is not None:
            norm.frozen = False
        return score

    _, log = td3.train(Adapter(PointMass(), norm), total_steps=STEPS, evaluate=evaluate, eval_every=3_000, seed=seed)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"PD controller (expert_action) return: {expert_return():.2f}")
    for normalise in (False, True):
        for seed in SEEDS:
            path = RESULTS / f"obs__{'normalised' if normalise else 'raw'}__seed{seed}.json"
            if args.resume and path.exists():
                print(f"skip {path.name}"); continue
            start = time.time()
            log = run(normalise, seed)
            path.write_text(json.dumps(dict(normalise=normalise, seed=seed, log=log, expert=expert_return())))
            print(f"done {path.name}: final {log[-1]['score']:.2f} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
