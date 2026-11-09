"""DQN family on CartPole: the replay/target ablation and the five-variant comparison.

    python experiments/deep_value_sweep.py ablation  [--resume]
    python experiments/deep_value_sweep.py compare   [--resume]
    python experiments/deep_value_sweep.py report

ablation   DQN with replay and the target network removed one at a time
           (`buffer_size=1` is no replay, `target_update=1` is no target network).
compare    DQN, Double, Dueling, PER, 3-step, C51 and Noisy, identical settings otherwise.

Every run writes one JSON file to experiments/results/deep_value/ as soon as it
finishes, so a crash costs one run and `--resume` skips everything already done.
Each file holds the greedy-return curve, the health log, and the overestimation
measurement (mean predicted max Q against the discounted return actually received).

The numbers are CartPole's, which is a weak test of the variants: rewards are
deterministic and capped by a 500-step time limit, so there is little noise for
Double DQN or PER to fix. The comparison is reported anyway, because whether a
fix helps where the problem it fixes is absent is itself worth knowing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "deep_value"
N_STEPS = 100_000
EVAL_EVERY = 5_000
SEEDS = range(5)

ABLATION = {
    "replay + target": dict(),
    "no target": dict(target_update=1),
    "no replay": dict(buffer_size=1),
    "neither": dict(buffer_size=1, target_update=1),
}
VARIANTS = ("dqn", "double_dqn", "dueling_dqn", "per_dqn", "n_step_dqn", "c51_dqn", "noisy_dqn")


def run_one(module_name, kwargs, seed):
    import importlib

    import gymnasium as gym
    import torch

    from algos.dqn import evaluate_gym, value_vs_return

    torch.set_num_threads(1)
    module = importlib.import_module(f"algos.{module_name}")
    make = lambda: gym.make("CartPole-v1")
    start = time.time()
    net, log = module.train(make(), n_steps=N_STEPS, evaluate=evaluate_gym(make),
                            eval_every=EVAL_EVERY, seed=seed, **kwargs)
    predicted, actual = value_vs_return(net, make)
    return dict(module=module_name, kwargs=kwargs, seed=seed, log=log, predicted=predicted,
                actual=actual, seconds=time.time() - start)


def sweep(kind, resume):
    jobs = ([(name, "dqn", kw) for name, kw in ABLATION.items()] if kind == "ablation"
            else [(v, v, {}) for v in VARIANTS])
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, module, kwargs in jobs:
        for seed in SEEDS:
            path = RESULTS / f"{kind}__{name.replace(' ', '_').replace('+', 'and')}__seed{seed}.json"
            if resume and path.exists():
                print(f"skip  {path.name} (done)", flush=True)
                continue
            result = run_one(module, kwargs, seed)
            result["name"] = name
            path.write_text(json.dumps(result))
            print(f"done  {path.name}  final {result['log'][-1]['score']:.1f}  "
                  f"Q {result['predicted']:.2f} vs actual {result['actual']:.2f}  {result['seconds']:.0f}s",
                  flush=True)


def load(kind):
    rows = {}
    for path in sorted(RESULTS.glob(f"{kind}__*.json")):
        result = json.loads(path.read_text())
        rows.setdefault(result["name"], []).append(result)
    return rows


def report():
    for kind in ("ablation", "compare"):
        rows = load(kind)
        if not rows:
            continue
        print(f"\n{kind}: CartPole-v1, {N_STEPS:,} steps, median [min, max] over seeds\n")
        print(f"{'':>16} {'seeds':>5} {'final greedy':>22} {'mean over curve':>16} "
              f"{'predicted Q':>12} {'actual':>8} {'max |mean Q|':>13}")
        for name, results in rows.items():
            final = np.array([r["log"][-1]["score"] for r in results])
            area = np.array([np.mean([e["score"] for e in r["log"]]) for r in results])
            predicted = np.median([r["predicted"] for r in results])
            actual = np.median([r["actual"] for r in results])
            peak_q = np.median([max(abs(e["q"]) for e in r["log"]) for r in results])
            print(f"{name:>16} {len(results):>5} "
                  f"{f'{np.median(final):6.1f} [{final.min():5.1f}, {final.max():5.1f}]':>22} "
                  f"{np.median(area):>16.1f} {predicted:>12.2f} {actual:>8.2f} {peak_q:>13.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("ablation", "compare", "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.what == "report":
        report()
    else:
        sweep(args.what, args.resume)
        report()
