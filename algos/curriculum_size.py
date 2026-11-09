"""Size curriculum (Bengio, Louradour, Collobert & Weston, 2009, applied to RL).

    python algos/curriculum_size.py

Bengio et al. trained supervised models on easy examples first and harder ones
later, and found faster training and better final solutions. The RL version:
solve a small instance, then a larger one, carrying the learned values across.

Here the task grows (Chain 5 -> 10 -> ... -> n, or GridWorld 4x4 -> ... -> 14x14)
and the Q table is transferred by aligning states relative to the goal: the cell
k steps from the goal in the small task maps to the cell k steps from the goal in
the large one. New states start at zero. Each stage trains until the greedy
policy reaches 95% of that stage's exact optimum, or its step budget runs out.

Unlike the reverse curriculum, this needs no arbitrary resets. It needs a family
of related tasks and a way to map knowledge between them, which is the same
problem in a different place.
"""

from __future__ import annotations

import numpy as np

from algos import count_based_exploration as learner
from envs.chain import Chain
from envs.gridworld import GridWorld

GAMMA = 0.99


def transfer(small, big, Q_small):
    """Copy Q from a smaller Chain or GridWorld into a larger one, aligned at the goal."""
    Q = np.zeros((big.n_states, big.n_actions))
    if isinstance(big, Chain):
        shift = big.n - small.n
        Q[shift:shift + small.n] = Q_small
        return Q
    shift = big.size - small.size
    for s in range(small.n_states):
        r, c = small.cell(s)
        Q[big.index((r + shift, c + shift))] = Q_small[s]
    return Q


def train(make, sizes, stage_budget=50_000, gamma=GAMMA, eval_every=50, seed=0, **kwargs):
    """`make(size)` builds the task. Returns (Q, total steps, per-stage log)."""
    Q, prev, total, stages = None, None, 0, []
    for size in sizes:
        env = make(size)
        if Q is not None:
            Q = transfer(prev, env, Q)
        optimal = env.optimal_return(gamma)
        Q, curve, _ = learner.train(env, n_steps=stage_budget, gamma=gamma, eval_every=eval_every,
                                    seed=seed + size, Q=Q, **kwargs)
        solved = next((s for s, v in curve if v >= 0.95 * optimal), None)
        used = solved if solved is not None else stage_budget
        # Count only the steps needed: retrain to exactly that many, so later stages
        # start from the table the agent had at the moment it solved this one.
        if solved is not None and solved < stage_budget:
            start = None if prev is None else transfer(prev, env, stages[-1]["Q"])
            Q, _, _ = learner.train(env, n_steps=solved, gamma=gamma, eval_every=solved, seed=seed + size,
                                    Q=start, **kwargs)
        total += used
        stages.append(dict(size=size, steps=used, solved=solved is not None, Q=Q))
        prev = env
    return Q, total, stages


def main():
    make = lambda n: Chain(n=n, max_steps=4 * n)
    print("Chain, epsilon-greedy Q-learning, growing n by 5 each stage")
    for seed in range(3):
        _, total, stages = train(make, (5, 10, 15, 20, 25, 30), seed=seed)
        print(f"  seed {seed}: total {total:,} steps; per stage "
              + ", ".join(f"n={s['size']}: {s['steps']:,}{'' if s['solved'] else ' (not solved)'}" for s in stages))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - A later stage takes as long as training from scratch: the transfer is
#   misaligned, so the copied values point the wrong way. Check that the cell
#   next to the goal maps to the cell next to the goal.
# - Values from the small task are too large for the big one: the discounted
#   return from k steps away is the same number in both, so this is fine for a
#   goal-aligned mapping and wrong for any other.
# - The final stage is never solved: the stages grow too fast. The point of the
#   curriculum is that each stage is easy given the last.
