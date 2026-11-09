"""Curriculum against exploration, on the same sparse tasks, with one metric.

    python experiments/curriculum_vs_exploration.py [--resume]

The plan's framing: a curriculum is an exploration method wearing different
clothes. Both work by making sure the agent sees reward early. So run them side
by side and count the same thing for each: environment steps until the greedy
policy, evaluated exactly from the real start, reaches 95% of the optimum.

    epsilon-greedy      the baseline
    optimistic Q        every Q starts at 1, the largest possible return
    count bonus         r + 0.1 / sqrt(N(s, a))
    reverse curriculum  start next to the goal, move outward (needs arbitrary resets)
    size curriculum     solve small instances first, transfer Q aligned at the goal
                        (needs a family of tasks and a mapping between them)
    random task         reverse-curriculum starts, bucket chosen uniformly per episode
    learning progress   the same buckets chosen by a teacher (`algos/curriculum_automatic.py`)

Two tasks: Chain(30), where a random walk reaches the goal with probability
2^-29 per episode, and a 14x14 sparse GridWorld, where it does not need help.
Tabular Q-learning throughout, alpha 0.5, epsilon 0.1, gamma 0.99, 5 seeds,
budget 200,000 steps. Every step counts, including the curricula's easy stages.
`--resume` keeps every (task, method) row already in the JSON and runs the rest.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from algos import count_based_exploration as learner
from algos import curriculum_automatic, curriculum_reverse, curriculum_size
from envs.chain import Chain
from envs.gridworld import GridWorld

GAMMA = 0.99
BUDGET = 200_000
EVAL_EVERY = 100
SEEDS = range(5)
RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "exploration" / "curriculum_vs_exploration.json"

TASKS = {
    "Chain(30)": dict(make=lambda n=30: Chain(n=n, max_steps=4 * n), sizes=(5, 10, 15, 20, 25, 30)),
    "GridWorld 14x14": dict(make=lambda n=14: GridWorld(size=n, reward_mode="sparse", max_steps=300),
                            sizes=(4, 6, 8, 10, 12, 14)),
}


def steps_to_solve(curve, optimal):
    return next((step for step, value in curve if value >= 0.95 * optimal), None)


def run(task, method, seed):
    make, sizes = TASKS[task]["make"], TASKS[task]["sizes"]
    env = make()
    optimal = env.optimal_return(GAMMA)
    if method == "size curriculum":
        Q, total, stages = curriculum_size.train(make, sizes, stage_budget=BUDGET, eval_every=50, seed=seed)
        solved = all(s["solved"] for s in stages) and env.policy_return(Q.argmax(1), GAMMA) >= 0.95 * optimal
        return total if solved else None
    if method in ("random task", "learning progress"):
        selector = "random" if method == "random task" else "progress"
        _, curve, _ = curriculum_automatic.train(env, selector, n_steps=BUDGET, eval_every=EVAL_EVERY, seed=seed)
        return steps_to_solve(curve, optimal)
    if method == "reverse curriculum":
        _, curve, _ = curriculum_reverse.train(env, n_steps=BUDGET, eval_every=EVAL_EVERY, seed=seed)
        return steps_to_solve(curve, optimal)
    kwargs = {"epsilon-greedy": {}, "optimistic Q": dict(q_init=1.0), "count bonus": dict(bonus=0.1)}[method]
    _, curve, _ = learner.train(env, n_steps=BUDGET, eval_every=EVAL_EVERY, seed=seed, **kwargs)
    return steps_to_solve(curve, optimal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    methods = ("epsilon-greedy", "optimistic Q", "count bonus", "reverse curriculum", "size curriculum",
               "random task", "learning progress")
    print("Steps until the greedy policy reaches 95% of the exact optimum, from the real start.")
    print(f"Median over {len(SEEDS)} seeds [min, max]; 'never k/5' = solved on k seeds within {BUDGET:,} steps.\n")
    table = json.loads(RESULTS.read_text())["table"] if args.resume and RESULTS.exists() else {}
    for task in TASKS:
        print(task)
        for method in methods:
            if method in table.get(task, {}):
                hits = table[task][method]
            else:
                hits = [run(task, method, seed) for seed in SEEDS]
            table.setdefault(task, {})[method] = hits
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            RESULTS.write_text(json.dumps(dict(budget=BUDGET, table=table), indent=1))
            found = [h for h in hits if h is not None]
            if len(found) >= 3:
                cell = f"{int(np.median(found)):>8,} [{min(found):,}, {max(found):,}]"
                if len(found) < len(hits):
                    cell += f"  ({len(found)}/5 solved)"
            else:
                cell = f"never {len(found)}/5"
            print(f"  {method:>19}  {cell}", flush=True)
        print()


if __name__ == "__main__":
    main()
