"""Reward scale on CartPole: multiply every reward by 1,000 and compare three fixes.

    python experiments/reward_normalisation.py [--resume]

The optimal policy does not change when every reward is multiplied by a
positive constant. The learning dynamics do. Five configurations, DQN, 100,000
steps, 3 seeds, greedy return always reported in the ORIGINAL units (steps
balanced, at most 500):

    scale 1              the reference
    scale 1000           nothing done about it
    scale 1000, clipped  rewards clipped to [-1, 1], as DQN did on Atari
    scale 1000, /1000    rewards divided by the known scale (the oracle fix)
    scale 1000, PopArt   adaptive target normalisation, `algos/popart_dqn.py`

One JSON per run in experiments/results/normalisation/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "normalisation"
STEPS = 100_000
SEEDS = range(3)
CONFIGS = {
    "scale 1": (1.0, "none", "dqn"),
    "scale 1000": (1000.0, "none", "dqn"),
    "scale 1000, clipped": (1000.0, "clip", "dqn"),
    "scale 1000, /1000": (1000.0, "divide", "dqn"),
    "scale 1000, PopArt": (1000.0, "none", "popart_dqn"),
}


class ScaledReward:
    def __init__(self, env, scale, fix):
        self.env, self.scale, self.fix = env, scale, fix
        self.action_space, self.observation_space = env.action_space, env.observation_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.scale
        if self.fix == "clip":
            reward = max(-1.0, min(1.0, reward))
        elif self.fix == "divide":
            reward /= self.scale
        return obs, reward, terminated, truncated, info


def main():
    import importlib

    import gymnasium as gym
    import torch

    from algos.dqn import evaluate_gym

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    make = lambda: gym.make("CartPole-v1")
    for name, (scale, fix, module) in CONFIGS.items():
        for seed in SEEDS:
            path = RESULTS / f"reward__{name.replace(' ', '_').replace(',', '').replace('/', 'div')}__seed{seed}.json"
            if args.resume and path.exists():
                print(f"skip {path.name}"); continue
            start = time.time()
            train = importlib.import_module(f"algos.{module}").train
            _, log = train(ScaledReward(make(), scale, fix), n_steps=STEPS, evaluate=evaluate_gym(make),
                           eval_every=10_000, seed=seed)
            path.write_text(json.dumps(dict(name=name, seed=seed, log=log)))
            print(f"done {path.name}: final {log[-1]['score']:.1f}, peak |mean Q| "
                  f"{max(abs(e['q']) for e in log):.1f} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
