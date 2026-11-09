"""Catastrophic forgetting across curriculum stages, `algos/curriculum_failure.py`.

    python experiments/curriculum_failure.py [--resume]

Four goals trained one after another, 30,000 steps each, 5 seeds, three arms:

    plain        rehearsal 0,    replay 2,000     the schedule as written
    rehearsal    rehearsal 0.25, replay 2,000     a quarter of episodes revisit earlier goals
    big replay   rehearsal 0,    replay 120,000   every transition of every stage kept

What the arms change: only which goals' transitions are in the batches the network
is trained on. Forgetting shows in `plain` if training goal k moves the shared
weights that goal 1 relied on; the other two arms win only if putting earlier
goals back into the batches is enough to hold them.

Every goal is evaluated every 1,000 steps. One JSON per run in
experiments/results/exploration/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "exploration"
SEEDS = range(5)
ARMS = {"plain": dict(rehearsal=0.0, buffer_size=2_000), "rehearsal": dict(rehearsal=0.25, buffer_size=2_000),
        "big replay": dict(rehearsal=0.0, buffer_size=120_000)}


def main():
    import torch

    from algos import curriculum_failure

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, kwargs in ARMS.items():
        for seed in SEEDS:
            path = RESULTS / f"curriculum_failure__{name.replace(' ', '_')}__seed{seed}.json"
            if args.resume and path.exists():
                print(f"skip {path.name}", flush=True)
                continue
            start = time.time()
            _, log = curriculum_failure.train(eval_every=1_000, seed=seed, **kwargs)
            path.write_text(json.dumps(dict(name=name, seed=seed, kwargs=kwargs, log=log,
                                            stage_steps=curriculum_failure.STAGE_STEPS,
                                            seconds=time.time() - start)))
            ends = [[r for r in log if r["stage"] == k][-1]["success"] for k in range(4)]
            print(f"done {path.name}: success per goal at the end of each stage {ends}, "
                  f"{time.time() - start:.0f}s", flush=True)
    print("\nsuccess on each goal after the last stage, summed over seeds (goal 4 is the last trained)")
    for name in ARMS:
        rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"curriculum_failure__{name.replace(' ', '_')}__seed*.json"))]
        if rows:
            final = np.sum([r["log"][-1]["success"] for r in rows], axis=0)
            print(f"{name:>11}: {final.tolist()} of {len(rows)}")


if __name__ == "__main__":
    main()
