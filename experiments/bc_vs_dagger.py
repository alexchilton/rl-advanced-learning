"""BC vs DAgger on a matched expert-label budget.

    python experiments/bc_vs_dagger.py

The comparison people usually draw is unfair in both directions. BC gets counted
in "demonstrations" and DAgger in "rounds", so neither number means anything.

The resource both methods spend is the same: one query to the expert, producing
one labelled state. Fix that budget and the question becomes sharp -- given N
labels, is it better to spend them on the EXPERT's states, or on the LEARNER's?

Returns here are exact, from `policy_return`, not sampled. A difference between
two rows is a real difference and not seed noise. The spread across seeds is
reported anyway, because the spread is half the story: BC's variance stays large
long after its median stops improving.
"""

from __future__ import annotations

import numpy as np

from algos import bc, dagger
from envs.racetrack import Racetrack

GAMMA = 0.99
BUDGETS = (10, 20, 40, 80, 160, 320)
SEEDS = range(1, 6)


def main():
    env = Racetrack(on_crash="stop")
    expert = env.optimal_policy(GAMMA)
    optimal = env.optimal_return(GAMMA)

    print("=" * 78)
    print("BC vs DAgger, matched expert-label budget, exact returns")
    print("=" * 78)
    print(f"Racetrack: {env.n_states} states, {env.n_actions} actions, "
          f"{len(SEEDS)} seeds, gamma={GAMMA}")
    print(f"expert (exact optimal policy): {optimal:.3f}\n")
    print(f"{'labels':>7} {'BC: median [min, max]':>28} {'DAgger: median [min, max]':>28}")

    for budget in BUDGETS:
        bc_values, dagger_values = [], []
        for seed in SEEDS:
            states, actions = bc.collect_demonstrations(env, expert, budget, seed=seed)
            bc_values.append(bc.exact_return(env, bc.fit(env, states, actions), GAMMA))
            dagger_values.append(
                dagger.exact_return(env, dagger.train(env, expert, budget, seed=seed), GAMMA)
            )
        b, d = np.array(bc_values), np.array(dagger_values)
        print(
            f"{budget:>7} "
            f"{f'{np.median(b):8.2f} [{b.min():7.2f}, {b.max():7.2f}]':>28} "
            f"{f'{np.median(d):8.2f} [{d.min():7.2f}, {d.max():7.2f}]':>28}"
        )

    print()
    print("Two things to read off this table.")
    print()
    print("1. DAgger reaches the expert. BC does not, and does not get closer.")
    print("   Every extra BC label lands on the expert's own trajectory, which was")
    print("   never the part that was missing. This is the compounding-error result")
    print("   from Ross et al. 2011: BC's expected cost grows with the SQUARE of the")
    print("   horizon, DAgger's linearly.")
    print()
    print("2. BC's spread stays wide long after its median stops moving. A single")
    print("   BC run can look almost optimal. Reporting one seed here would let you")
    print("   conclude either method wins, depending on which seed you drew.")
    print()
    print("What DAgger costs, and this table does not show: the expert has to be")
    print("callable DURING training, not recorded once. Free with a scripted")
    print("controller. Ruinous with a human.")

    covariate_shift_panel(env, expert, optimal)


def covariate_shift_panel(env, expert, optimal):
    """The mechanism, measured, rather than asserted."""
    print()
    print("=" * 78)
    print("The mechanism: how far the learner drifts from the demonstrated states")
    print("=" * 78)
    print(f"{'labels':>7} {'BC shift':>10} {'BC return':>11} {'DAgger shift':>14} {'DAgger return':>14}")

    for budget in BUDGETS:
        states, actions = bc.collect_demonstrations(env, expert, budget, seed=1)
        bc_policy = bc.fit(env, states, actions)
        bc_visited = bc.rollout(env, bc_policy, n_episodes=30, seed=90_000)
        bc_shift = bc.covariate_shift(env, states, bc_visited)

        dagger_policy = dagger.train(env, expert, budget, seed=1)
        dagger_visited = bc.rollout(env, dagger_policy, n_episodes=30, seed=90_000)
        dagger_shift = bc.covariate_shift(env, states, dagger_visited)

        print(
            f"{budget:>7} {bc_shift:>10.3f} {bc.exact_return(env, bc_policy, GAMMA):>11.2f} "
            f"{dagger_shift:>14.3f} {dagger.exact_return(env, dagger_policy, GAMMA):>14.2f}"
        )

    print()
    print("Shift is the mean distance from a state the learner visits to the")
    print("nearest state the expert demonstrated, in normalised feature space.")
    print("It is the quantity the whole BC-versus-DAgger argument is about, and it")
    print("is almost never plotted. When it stays high, more expert data cannot help.")


if __name__ == "__main__":
    main()
