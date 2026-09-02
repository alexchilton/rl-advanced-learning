"""Sparse, dense, shaped and misspecified reward, on the same task.

    python experiments/sparse_vs_dense.py

Four reward functions on one grid. The agent trains on each of them and is scored
on the SAME thing every time: the sparse objective, "did you reach the goal, and
how fast". That detail is the whole experiment. Scoring a shaped agent by its own
shaped reward measures how big its bonus was, not whether it solved anything.

Three of the four have the identical optimal policy, which `tests/test_gridworld.py`
asserts numerically. So any difference below is a difference in how hard the reward
makes the problem to LEARN, not in what it asks for.

What comes out:

  * dense is flat in grid size -- about 5,000 steps whether the grid is 6x6 or 14x14
  * sparse degrades badly -- 5,000 to 35,000 over the same range
  * potential-based shaping, the provably-safe one, is SLOWER than raw sparse
    reward at every size from 10x10 up
  * the misspecified reward never reaches the goal at any size

The third result is the one worth the trip, and the mechanism is in
`shaping_margin` below.
"""

from __future__ import annotations

import numpy as np

from algos.q_learning import steps_to_threshold, train
from envs.gridworld import POTENTIAL_SCALE, REWARD_MODES, GridWorld

GAMMA = 0.95
SIZES = (6, 8, 10, 12, 14)
SEEDS = (0, 1, 2)
N_STEPS = 120_000
EVAL_EVERY = 5_000
THRESHOLD = 0.95  # fraction of optimal that counts as solved


def sweep():
    print("=" * 78)
    print("Steps to reach 95% of optimal, scored on the sparse objective")
    print("=" * 78)
    print(f"\nQ-learning, gamma={GAMMA}, {len(SEEDS)} seeds, {N_STEPS:,} step budget.")
    print("Median across seeds; 'never' means fewer than two seeds got there.\n")
    print(f"{'grid':>7} " + "".join(f"{m:>15}" for m in REWARD_MODES))

    results = {}
    for size in SIZES:
        scorer = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA, max_steps=300)
        optimal = scorer.optimal_return(GAMMA)

        cells = []
        for mode in REWARD_MODES:
            hits = []
            for seed in SEEDS:
                env = GridWorld(size=size, reward_mode=mode, gamma=GAMMA, max_steps=300)
                _, curve = train(
                    env, gamma=GAMMA, n_steps=N_STEPS, eval_every=EVAL_EVERY,
                    eval_env=scorer, seed=seed,
                )
                hits.append(steps_to_threshold(curve, optimal * THRESHOLD))
            reached = [h for h in hits if h is not None]
            results[(size, mode)] = reached
            cells.append(f"{int(np.median(reached)):,}" if len(reached) >= 2 else "never")
        print(f"{f'{size}x{size}':>7} " + "".join(f"{c:>15}" for c in cells))

    return results


def shaping_margin():
    """Why potential-based shaping makes learning harder, in one identity.

    Shaping is guaranteed not to change the optimal policy, and it does not. What
    it changes is the SCALE of the value function relative to the decision the
    learner has to make.

        V_shaped(s) = V_sparse(s) - Phi(s)

    With Phi(s) = -0.1 * distance(s), that adds up to +2.2 to every value on a
    14x14 grid. Meanwhile the agent's alternative -- stand still and collect the
    shaping term forever -- is worth exactly -Phi(s), because a stationary
    transition pays (gamma - 1) * Phi each step and that sums to -Phi.

    So the margin between "solve the task" and "loiter forever" is

        V_shaped(s) - (-Phi(s)) = V_sparse(s)

    exactly the sparse return, and no larger. Shaping inflated the values by 2.2
    and left the thing being decided unchanged. Every TD update now has to resolve
    a 0.28 difference sitting on top of a 2.2 baseline, with the same learning rate
    and the same noise.

    That is a signal-to-noise problem, and it is invisible in the theorem.
    """
    print("\n" + "=" * 78)
    print("Why shaping is slower: the values grow and the margin does not")
    print("=" * 78)
    print(f"\n{'grid':>7} {'V_sparse(start)':>16} {'-Phi(start)':>13} "
          f"{'V_shaped(start)':>16} {'margin':>9} {'ratio':>8}")

    for size in SIZES:
        sparse = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
        shaped = GridWorld(size=size, reward_mode="shaped", gamma=GAMMA)
        potential = POTENTIAL_SCALE * sparse.distance(sparse.start)
        v_sparse = sparse.optimal_return(GAMMA)
        v_shaped = shaped.optimal_return(GAMMA)
        margin = v_shaped - potential
        print(f"{f'{size}x{size}':>7} {v_sparse:>16.4f} {potential:>13.4f} "
              f"{v_shaped:>16.4f} {margin:>9.4f} {margin / v_shaped:>8.1%}")

    print("\n'margin' is what the agent gains by solving the task rather than")
    print("standing still and collecting the shaping term forever. It equals the")
    print("sparse return exactly, at every size.")
    print("\n'ratio' is that margin as a fraction of the values being estimated. It")
    print("falls as the grid grows, and that -- not the optimal policy, which is")
    print("provably unchanged -- is what makes shaped reward slow to learn here.")


def reward_hacking():
    """The misspecified mode, and what the agent does with it."""
    print("\n" + "=" * 78)
    print("The misspecified reward: what the agent learns instead")
    print("=" * 78)

    size = 8
    hacked = GridWorld(size=size, reward_mode="misspecified", gamma=GAMMA)
    honest = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)

    print(f"\nOptimal policy under the misspecified reward ({size}x{size}):\n")
    print(hacked.render_policy(hacked.optimal_policy(GAMMA)))
    print("\nLook at the two cells next to the goal. The arrows point AWAY from it.")

    hacked_value = hacked.policy_return(hacked.optimal_policy(GAMMA), GAMMA)
    honest_value = hacked.policy_return(honest.optimal_policy(GAMMA), GAMMA)
    print(f"\n  value of oscillating forever    {hacked_value:+.4f}")
    print(f"  value of actually finishing     {honest_value:+.4f}")
    print("\nThe difference between this reward and the correct potential-based one")
    print("is a single max(0, ...). It rewards progress and does not penalise")
    print("regress, so stepping towards the goal and back again pays forever.")
    print("\nThe agent is not broken. It is doing exactly what it was asked.")


def main():
    results = sweep()
    commentary(results)
    shaping_margin()
    reward_hacking()


def commentary(results):
    def median(size, mode):
        reached = results[(size, mode)]
        return int(np.median(reached)) if len(reached) >= 2 else None

    small, large = SIZES[0], SIZES[-1]
    print("\n" + "=" * 78)
    print("Reading the table")
    print("=" * 78)

    print(f"\n1. Dense reward is flat in grid size: about {median(small, 'dense'):,} steps at "
          f"{small}x{small} and\n   {median(large, 'dense'):,} at {large}x{large}. A per-step "
          "progress signal removes the exploration\n   problem entirely, because there is "
          "always a gradient to follow.")

    sparse_small, sparse_large = median(small, "sparse"), median(large, "sparse")
    print(f"\n2. Sparse reward degrades: {sparse_small:,} to {sparse_large:,} steps over the "
          f"same range,\n   a factor of {sparse_large / sparse_small:.0f}. Nothing is learnable "
          "until the goal is reached by\n   chance at least once, and that gets rarer as the "
          "grid grows.")

    print("\n3. Potential-based shaping is SLOWER than sparse at every size from")
    print("   10x10 up. That is the result worth stopping on, because shaping is the")
    print("   textbook safe option -- Ng et al. 1999 prove it cannot change the")
    print("   optimal policy, and this repo's tests confirm it numerically.")
    print("\n   Both things are true. The policy is unchanged and the problem got")
    print("   harder to learn. See the margin analysis below for the mechanism.")

    print("\n4. The misspecified reward never reaches the goal at any size, and its")
    print("   agent is behaving optimally throughout.")

    print("\nOne caveat: three seeds. The shaped row is non-monotonic in size, which")
    print("means its spread is wide enough to reorder adjacent entries. The gap")
    print("between shaped and dense is far larger than that noise; the gap between")
    print("shaped at 10x10 and shaped at 12x12 is not.")


if __name__ == "__main__":
    main()
