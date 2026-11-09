"""DQN against DQN + RND (two value heads) on long chains.

    python experiments/rnd_chain.py [--resume]

Chain(20) and Chain(30), one-hot observations, 50,000 steps, 5 seeds each. Per
run: the step the goal is first reached, and the exact return of the greedy
policy read from the extrinsic value head every 10,000 steps. One JSON per run
in experiments/results/exploration/, so --resume skips finished runs.
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


def main():
    import torch

    from algos import rnd
    from algos.dqn import q_table
    from envs.chain import Chain

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for n in LENGTHS:
        env = Chain(n=n, max_steps=4 * n, obs_mode="onehot")
        evaluate = lambda net: env.policy_return(q_table(net, env).argmax(1), 0.99)
        for name, beta in (("DQN", 0.0), ("DQN + RND", 1.0)):
            for seed in SEEDS:
                path = RESULTS / f"rnd_chain__n{n}__{name.replace(' + ', '_')}__seed{seed}.json"
                if args.resume and path.exists():
                    print(f"skip {path.name}", flush=True)
                    continue
                start = time.time()
                _, log, first = rnd.train(env, n_steps=STEPS, beta=beta, evaluate=evaluate, eval_every=10_000, seed=seed)
                path.write_text(json.dumps(dict(n=n, name=name, seed=seed, first_goal=first, log=log,
                                                optimum=env.optimal_return(0.99), seconds=time.time() - start)))
                print(f"done {path.name}: first goal {first}, final greedy {log[-1]['score']:.3f}, "
                      f"{time.time() - start:.0f}s", flush=True)
    for n in LENGTHS:
        for name in ("DQN", "DQN + RND"):
            rows = [json.loads(p.read_text()) for p in RESULTS.glob(f"rnd_chain__n{n}__{name.replace(' + ', '_')}__*.json")]
            if rows:
                solved = sum(r["log"][-1]["score"] >= 0.95 * r["optimum"] for r in rows)
                firsts = [r["first_goal"] for r in rows if r["first_goal"] is not None]
                print(f"Chain({n}) {name:>10}: goal found on {len(firsts)}/{len(rows)} seeds "
                      f"(median step {int(np.median(firsts)) if firsts else None}); "
                      f"greedy policy optimal at the end on {solved}/{len(rows)}")


if __name__ == "__main__":
    main()
