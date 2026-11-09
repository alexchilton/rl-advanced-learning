"""Observation normalisation on PointMass: same agent, raw vs normalised inputs.

    python experiments/obs_normalisation.py [--resume]

`envs/pointmass.py` is badly scaled on purpose: position stays in [-1, 1] while
velocity reaches tens. A network fed that raw has to learn first-layer weights
of very different sizes before it can learn anything else.

The fix in deep RL is a running mean/std normaliser on the observation, the
input-side relative of batch normalisation (Ioffe & Szegedy 2015, which
normalises hidden activations per batch). Here: TD3, 30,000 steps, 3 seeds,
scored against the hand-written PD controller that `PointMass.expert_action`
provides. Three modes:

    raw     the observation as the environment returns it
    store   normalised by the environment wrapper, so replay holds each entry on
            the scale of the statistics at the moment it was collected
    sample  replay holds raw observations; every batch is normalised with the
            CURRENT statistics when it is sampled (`td3.train(obs_transform=...)`)

`store` learned slower than `raw` on every seed. `sample` tests whether the stale
scales in replay are the reason: it differs from `store` only in when the
statistics are applied.

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

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.frozen:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)

    def std(self):
        return np.sqrt(self.m2 / max(self.n - 1, 1)) + 1e-8

    def __call__(self, x):
        self.update(x)
        return ((np.asarray(x, np.float64) - self.mean) / self.std()).clip(-10, 10).astype(np.float32)

    def tensor(self, x):
        """Normalise a batch of stored raw observations with the current statistics, no update."""
        import torch

        mean, std = (torch.as_tensor(v, dtype=torch.float32) for v in (self.mean, self.std()))
        return ((x - mean) / std).clamp(-10, 10)


class Adapter:
    """PointMass with Gymnasium-style spaces and the normaliser for `mode`."""

    def __init__(self, env, norm=None, mode="raw"):
        self.env, self.norm, self.mode = env, norm, mode
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(shape=(2,), high=np.ones(2, dtype=np.float32))

    def _obs(self, obs):
        if self.mode == "store":
            return self.norm(obs)
        if self.mode == "sample":
            self.norm.update(obs)
        return np.asarray(obs, dtype=np.float32)

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


MODES = ("raw", "store", "sample")


def run(mode, seed):
    import torch

    from algos import td3
    from algos.ddpg import evaluate_gym
    from envs.pointmass import PointMass

    torch.set_num_threads(1)
    norm = RunningNorm(4)
    transform = norm.tensor if mode == "sample" else None

    def make_eval():
        norm.frozen = True
        return Adapter(PointMass(), norm, mode)

    def evaluate(actor):
        policy = actor if transform is None else (lambda x: actor(transform(x)))
        score = evaluate_gym(make_eval)(policy)
        norm.frozen = False
        return score

    _, log = td3.train(Adapter(PointMass(), norm, mode), total_steps=STEPS, evaluate=evaluate, eval_every=3_000,
                       seed=seed, obs_transform=transform)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"PD controller (expert_action) return: {expert_return():.2f}")
    for mode in MODES:
        for seed in SEEDS:
            path = RESULTS / f"obs__{mode}__seed{seed}.json"
            if args.resume and path.exists():
                print(f"skip {path.name}"); continue
            start = time.time()
            log = run(mode, seed)
            path.write_text(json.dumps(dict(mode=mode, seed=seed, log=log, expert=expert_return())))
            print(f"done {path.name}: final {log[-1]['score']:.2f} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
