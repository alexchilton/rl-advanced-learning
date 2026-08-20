"""What the dataset is worth more than what the algorithm is.

    python experiments/offline_dataset_quality.py

Same environment, same budget of transitions, four behaviour policies. Three
methods that all see exactly the same log and never touch the environment:

    behaviour cloning   copy what the data did
    offline Q-learning  bootstrap through max_a', with no protection
    CQL                 the same, with out-of-distribution actions pushed down

Every return is exact. The comparison is the fourth row of the access table in
docs/rl-curriculum-plan.md -- "can exceed the expert" -- made concrete.

The result to look for is not that CQL wins. It is WHERE it wins. On expert data
behaviour cloning is as good as anything and far simpler, so every conservative
mechanism is pure overhead. On mixed data BC collapses, because it averages good
and bad behaviour into something that is neither, while a method that can read
the reward keeps the good half.

That single row is the reason offline RL exists as a field.
"""

from __future__ import annotations

import numpy as np

from algos import cql_tabular, offline_q
from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
N_TRANSITIONS = 20_000
ALPHA = 1.0
SEEDS = (0, 1, 2)


def behaviour_cloning(dataset):
    """The most-taken action per state. Uniform-random where the log is silent.

    No reward, no bootstrapping, no discount. Just the empirical mode.
    """
    counts = dataset.action_counts()
    policy = counts.argmax(axis=1)
    unseen = counts.sum(axis=1) == 0
    rng = np.random.default_rng(0)
    policy[unseen] = rng.integers(0, dataset.n_actions, int(unseen.sum()))
    return policy.astype(np.int64)


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)

    print("=" * 84)
    print("Offline RL: the dataset matters more than the algorithm")
    print("=" * 84)
    print(f"Racetrack, {env.n_states} states. {N_TRANSITIONS:,} transitions per dataset, "
          f"{len(SEEDS)} seeds.")
    print(f"Exact optimal return: {optimal:.2f}   (nothing here interacts with the environment)\n")
    print(f"{'dataset':>8} {'coverage':>9} {'behaviour':>10} {'BC':>16} "
          f"{'offline Q':>16} {'CQL':>16}")

    rows = {}
    for quality in datasets.QUALITIES:
        behaviour_value = env.policy_return(
            datasets.behaviour_policy_for(env, quality, GAMMA), GAMMA
        )
        bc_values, q_values, cql_values, coverages = [], [], [], []

        for seed in SEEDS:
            dataset = datasets.build(env, quality, N_TRANSITIONS, GAMMA, seed=seed)
            coverages.append(dataset.coverage())
            bc_values.append(env.policy_return(behaviour_cloning(dataset), GAMMA))
            q_values.append(
                env.policy_return(offline_q.train(dataset, gamma=GAMMA).argmax(axis=1), GAMMA)
            )
            cql_values.append(
                env.policy_return(
                    cql_tabular.train(dataset, alpha=ALPHA, gamma=GAMMA).argmax(axis=1), GAMMA
                )
            )

        rows[quality] = (
            behaviour_value,
            float(np.median(bc_values)),
            float(np.median(cql_values)),
        )
        print(
            f"{quality:>8} {np.mean(coverages):>9.3f} {behaviour_value:>10.2f} "
            f"{summarise(bc_values):>16} {summarise(q_values):>16} {summarise(cql_values):>16}"
        )

    print("\nmedian [min, max] across seeds. 'behaviour' is the exact return of the")
    print("policy that generated the log -- the number every method must beat to have")
    print("been worth running.")

    commentary(rows, optimal)


def summarise(values):
    values = np.asarray(values)
    return f"{np.median(values):6.2f} [{values.min():6.2f}]"


def commentary(rows, optimal):
    print()
    print("=" * 84)
    print("Reading the table")
    print("=" * 84)

    expert_behaviour, expert_bc, expert_cql = rows["expert"]
    medium_behaviour, medium_bc, _ = rows["medium"]
    random_behaviour, random_bc, random_cql = rows["random"]

    print(f"\n1. On EXPERT data, BC scores {expert_bc:.2f} against CQL's {expert_cql:.2f}.")
    print("   When the log is already good, copying it is the right algorithm and every")
    print("   conservative mechanism is overhead. Reach for offline RL because the data")
    print("   is mediocre, not because the problem is important.")

    print(f"\n2. On RANDOM data, BC scores {random_bc:.2f} -- the value of never finishing --")
    print(f"   and CQL scores {random_cql:.2f} from the same log.")
    print("   This is the row that justifies the field. Purely random behaviour contains")
    print("   no good trajectory to copy, so BC has nothing to imitate and produces")
    print("   nonsense. But it contains the reward on every transition, and a method that")
    print("   reads the reward can assemble a good policy out of pieces of bad ones. Same")
    print("   data, same access, opposite outcome.")

    print(f"\n3. BC on MEDIUM data scores {medium_bc:.2f} against a behaviour policy worth")
    print(f"   {medium_behaviour:.2f}, which looks like BC exceeding its demonstrator and is not.")
    print("   The behaviour policy is a good policy plus exploration noise. Taking the")
    print("   most-common action per state strips the noise and recovers the policy")
    print("   underneath. Denoising, not improvement -- BC still cannot do anything the")
    print("   demonstrator was not already trying to do.")

    print("\n4. Offline Q-learning without the penalty is not a weak baseline, it is a")
    print("   broken one. It scores about -100 on every dataset, which is the value of")
    print("   never finishing, while predicting about -2 for itself. See algos/offline_q.py.")

    print(f"\n5. Nothing exceeded the optimal policy, and nothing could. CQL turning a")
    print(f"   {random_behaviour:.2f} random log into {random_cql:.2f} is stitching, not magic: it")
    print("   recombines transitions the log already contains. The ceiling is the data's")
    print("   coverage, not the demonstrator's skill.")

    print("\nOne caveat this demo does not hide: alpha was chosen by looking at the exact")
    print("return, which is cheating. In a real offline problem there is no validation")
    print("environment, because using one would be online interaction. Selecting")
    print("hyperparameters offline is an open problem and it is the main reason these")
    print("methods are harder to deploy than the papers suggest.")


if __name__ == "__main__":
    main()
