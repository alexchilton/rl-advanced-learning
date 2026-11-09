"""Epsilon-greedy against noisy networks on long chains.

    python experiments/noisy_chain.py [--resume]

`algos/dqn.py` (epsilon 1 -> 0.05 over the first 10% of steps) against
`algos/noisy_dqn.py` (epsilon 0, learned weight noise) with the noise resampled
every step and once per episode, everything else equal.
Chain(20) and Chain(30), one-hot observations, 50,000 steps, 5 seeds; the same
chains and budget as `experiments/rnd_chain.py`.

What the arm changes: only how actions are randomised. For noise to win, a
weight sample has to push the agent right for n steps in a row. The noise here
resampled every step (the paper's DQN agent) cannot hold a direction, so that
has to come from the learned mean; held for a whole episode it can.

Every chain is wrapped in `envs.chain.ScrambledActions` (a fresh action
permutation per state and per seed). Without it, RIGHT is action 1 in every state,
and an untrained network that happens to prefer index 1 solves the chain at
step ~20: the first unscrambled run of this file measured exactly that.

Per run: the step the goal is first reached (a wrapper counts it), and the exact
return of the greedy policy every 10,000 steps. One JSON per run in
experiments/results/exploration/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "exploration"
LENGTHS = (20, 30)
SEEDS = range(5)
STEPS = 50_000
AGENTS = {"DQN (epsilon-greedy)": ("dqn", {}), "Noisy DQN": ("noisy_dqn", {}),
          "Noisy DQN, per episode": ("noisy_dqn", dict(resample="episode"))}


class FirstGoal:
    """Passes everything through; records the environment step of the first reward."""

    def __init__(self, env):
        self.env, self.steps, self.first = env, 0, None
        self.n_actions = env.n_actions

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        self.steps += 1
        out = self.env.step(action)
        if out[1] > 0 and self.first is None:
            self.first = self.steps
        return out


def slug(name):
    return name.split(" (")[0].replace(", ", "_").replace(" ", "_")


def main():
    import importlib

    import torch

    from algos.dqn import q_table
    from envs.chain import Chain, ScrambledActions

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for n in LENGTHS:
        for name, (module, kwargs) in AGENTS.items():
            for seed in SEEDS:
                env = ScrambledActions(Chain(n=n, max_steps=4 * n, obs_mode="onehot"), seed=seed)
                evaluate = lambda net: env.policy_return(q_table(net, env).argmax(1), 0.99)
                path = RESULTS / f"noisy_chain__n{n}__{slug(name)}__seed{seed}.json"
                if args.resume and path.exists():
                    print(f"skip {path.name}", flush=True)
                    continue
                start = time.time()
                wrapped = FirstGoal(env)
                _, log = importlib.import_module(f"algos.{module}").train(
                    wrapped, n_steps=STEPS, evaluate=evaluate, eval_every=10_000, seed=seed, **kwargs)
                path.write_text(json.dumps(dict(n=n, name=name, seed=seed, first_goal=wrapped.first, log=log,
                                                optimum=env.optimal_return(0.99), seconds=time.time() - start)))
                print(f"done {path.name}: first goal {wrapped.first}, final greedy {log[-1]['score']:.3f}, "
                      f"{time.time() - start:.0f}s", flush=True)
    for n in LENGTHS:
        for name in AGENTS:
            rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"noisy_chain__n{n}__{slug(name)}__seed*.json"))]
            if rows:
                solved = sum(r["log"][-1]["score"] >= 0.95 * r["optimum"] for r in rows)
                firsts = [r["first_goal"] for r in rows if r["first_goal"] is not None]
                print(f"Chain({n}) {name:>23}: goal found on {len(firsts)}/{len(rows)} seeds "
                      f"(median step {int(np.median(firsts)) if firsts else None}); "
                      f"greedy policy optimal at the end on {solved}/{len(rows)}")


if __name__ == "__main__":
    main()
