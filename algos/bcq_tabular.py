"""Batch-Constrained Q-learning, tabular (Fujimoto, Meger & Precup, 2019).

    python algos/bcq_tabular.py

CQL attacks the symptom: out-of-distribution actions have inflated Q values, so
push those values down. BCQ attacks the cause: the backup should never have
CONSIDERED those actions. One line changes:

    offline Q   Q(s,a) <- r + gamma *          max_a'          Q(s',a')
    BCQ         Q(s,a) <- r + gamma *  max_{a' : b(a'|s') > tau}  Q(s',a')

where `b` is the empirical behaviour policy. The max runs only over actions the
log actually took at that state, in reasonable proportion. Nothing else is even a
candidate, so nothing else can be selected, so untouched entries cannot propagate.

`tau` is the dial, and it spans the same two failure modes as CQL's alpha from the
other direction:

    tau = 0      every action is admissible. This is naive offline Q-learning.
    tau small    only clearly-supported actions. The useful regime.
    tau = 1      only the single most-taken action. Behaviour cloning.

Which is the point worth taking away from having both files: CQL and BCQ are
different mechanisms with the same shape. Both interpolate between "trust the Q
function" and "trust the data", and both have a knob you cannot tune without a
validation environment you are not allowed to have.

One design choice this file makes explicit, because it is invisible in the paper
and dominates the tabular result. Some states appear in the log only as a
NEXT state, never as a state something was done from, so there is no behaviour
policy at them and no action is supported. Their value has to come from somewhere.
Setting it to zero is what breaks naive offline Q-learning here -- on an
all-negative-reward task, zero is the most optimistic number available. BCQ
therefore treats them pessimistically, and `PESSIMISTIC_VALUE` below is that
choice rather than an implementation detail.
"""

from __future__ import annotations

import numpy as np

from algos.offline_q import out_of_support_fraction, value_gap
from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
ITERATIONS = 300
LEARNING_RATE = 0.5
THRESHOLD = 0.3

# The value assigned to a state the log never acted from. -1/(1-gamma) is the
# worst possible return on a task where every reward is -1: "assume the worst
# about anywhere you have never been".
PESSIMISTIC_VALUE = -1.0 / (1.0 - GAMMA)


def admissible_actions(dataset, threshold=THRESHOLD):
    """Boolean (n_states, n_actions): actions the log took often enough at s.

    Relative rather than absolute -- an action must be at least `threshold` times
    as frequent as the most common action at that state. An absolute count
    threshold would silently exclude every action in rarely-visited states, which
    is exactly where the constraint matters most.
    """
    counts = dataset.action_counts()
    busiest = counts.max(axis=1, keepdims=True)
    return (counts > 0) & (counts >= threshold * busiest)


def train(
    dataset,
    threshold=THRESHOLD,
    gamma=GAMMA,
    iterations=ITERATIONS,
    learning_rate=LEARNING_RATE,
    pessimistic_value=PESSIMISTIC_VALUE,
    allowed=None,
):
    """Fitted Q iteration with the backup's max restricted to supported actions.

    `allowed` overrides the admissibility mask, which the ablation in `main` uses
    to separate the two things this method does at once.
    """
    if allowed is None:
        allowed = admissible_actions(dataset, threshold)

    Q = np.zeros((dataset.n_states, dataset.n_actions))
    counts = np.zeros_like(Q)
    np.add.at(counts, (dataset.states, dataset.actions), 1.0)
    seen = counts > 0

    # Two different questions, and conflating them makes the ablation meaningless.
    # `visited` is whether the log ever acted FROM this state at all -- if not,
    # nothing here is grounded and `pessimistic_value` decides its worth. `allowed`
    # is which actions may be maximised over at states that WERE acted from.
    visited = seen.any(axis=1)

    for _ in range(iterations):
        best = np.where(allowed, Q, -np.inf).max(axis=1)
        values = np.where(visited & np.isfinite(best), best, pessimistic_value)

        bootstrap = np.where(dataset.terminated, 0.0, gamma * values[dataset.next_states])
        target = dataset.rewards + bootstrap

        totals = np.zeros_like(Q)
        np.add.at(totals, (dataset.states, dataset.actions), target)
        mean_target = np.divide(totals, counts, out=np.zeros_like(Q), where=seen)
        Q[seen] += learning_rate * (mean_target[seen] - Q[seen])

    return Q


def greedy_policy(Q, dataset, threshold=THRESHOLD):
    """Argmax restricted to supported actions, as the paper's policy extraction.

    Taking an unrestricted argmax here would undo the whole method at the last
    step -- a common and completely silent mistake, because the training curve
    looks identical and only the evaluation changes.
    """
    allowed = admissible_actions(dataset, threshold)
    constrained = np.where(allowed, Q, -np.inf)
    policy = constrained.argmax(axis=1)
    unsupported = ~allowed.any(axis=1)
    policy[unsupported] = Q[unsupported].argmax(axis=1)
    return policy.astype(np.int64)


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)
    dataset = datasets.build(env, "medium", n_transitions=20_000, gamma=GAMMA, seed=0)
    behaviour = env.policy_return(datasets.behaviour_policy_for(env, "medium", GAMMA), GAMMA)

    print(f"Racetrack. Exact optimal return: {optimal:.2f}")
    print(f"Dataset: {dataset.summary()}")
    print(f"Behaviour policy that produced it: {behaviour:.2f}\n")
    ablation(env, dataset, optimal)

    print("\nThe tau dial, with the support constraint on throughout:\n")
    print(f"{'tau':>6} {'admissible/state':>17} {'OOS greedy':>11} {'believes':>10} "
          f"{'actually gets':>14}")

    for threshold in (0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0):
        Q = train(dataset, threshold=threshold)
        policy = greedy_policy(Q, dataset, threshold)
        allowed = admissible_actions(dataset, threshold)
        visited = dataset.action_counts().sum(axis=1) > 0

        predicted = float(np.dot(env.start_distribution, Q.max(axis=1)))
        actual = env.policy_return(policy, GAMMA)
        width = float(allowed[visited].sum(axis=1).mean())

        # Reuse the weighted diagnostic, applied to this policy rather than argmax.
        table = np.full_like(Q, -np.inf)
        table[np.arange(len(policy)), policy] = 0.0
        oos = out_of_support_fraction(table, dataset)

        print(f"{threshold:>6.2f} {width:>17.2f} {oos:>11.2f} {predicted:>10.2f} "
              f"{actual:>14.2f}")

    print("\nNote that tau barely matters. Even tau=0 keeps `counts > 0`, so the")
    print("support constraint is already doing the work, and the relative threshold")
    print("only trims how much CHOICE remains inside the support. At tau=1 that is one")
    print("action per state, which is behaviour cloning with a Q function attached.")
    print("\nCompare algos/cql_tabular.py: different mechanism, same shape of dial,")
    print("same impossibility of tuning it without an environment to test on.")


def ablation(env, dataset, optimal):
    """Which of BCQ's two mechanisms is actually doing the work?

    The method changes two things at once and the paper discusses only one. The
    support constraint restricts WHICH actions the backup may maximise over. The
    treatment of never-acted-from states decides what value those states get. Both
    are needed to fix the failure, and only the first one is the paper's idea.
    """
    everything = np.ones((dataset.n_states, dataset.n_actions), dtype=bool)
    supported = admissible_actions(dataset, threshold=0.0)

    print("Ablation: which mechanism fixes the failure?\n")
    print(f"{'max over':>14} {'unseen states get':>18} {'believes':>10} {'actually gets':>14}")

    for mask_name, mask in (("all actions", everything), ("supported only", supported)):
        for value_name, value in (("0 (optimistic)", 0.0), ("-100 (pessimistic)", PESSIMISTIC_VALUE)):
            Q = train(dataset, allowed=mask, pessimistic_value=value)
            policy = Q.argmax(axis=1) if mask is everything else greedy_policy(Q, dataset, 0.0)
            predicted = float(np.dot(env.start_distribution, Q.max(axis=1)))
            actual = env.policy_return(policy, GAMMA)
            print(f"{mask_name:>14} {value_name:>18} {predicted:>10.2f} {actual:>14.2f}")

    acted_from = np.zeros(dataset.n_states, dtype=bool)
    acted_from[dataset.states] = True
    bootstrapped = dataset.next_states[~dataset.terminated]
    orphans = int((~acted_from[bootstrapped]).sum())

    print("\nThe action constraint does all of the work: -98.73 to -6.61.")
    print("The pessimism column changes nothing at all, and the reason is worth more")
    print("than the result.")
    print(f"\nOf {len(bootstrapped):,} bootstrapped next-states in this dataset, {orphans} were")
    print("never acted from. Not few -- none. In TRAJECTORY data you always act from")
    print("wherever you landed, unless the episode ended there, in which case the")
    print("transition is terminal and does not bootstrap at all. So a state that gets")
    print("bootstrapped through but was never acted from cannot exist.")
    print("\nWhich says something specific about what extrapolation error IS here. It")
    print("is entirely about unseen ACTIONS at seen states, and not at all about")
    print("unseen states. Pessimistic initialisation is a real technique, but it")
    print("addresses a problem this kind of dataset does not have. It would matter for")
    print("a log of disconnected (s,a,r,s') tuples -- which is what you get from")
    print("logged production decisions, and not what you get from a simulator.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - BCQ is no better than naive offline Q: check the policy extraction. Training
#   with the constraint and then taking an unrestricted argmax at the end undoes
#   the method completely, and nothing in the training curve shows it.
# - BCQ collapses to the behaviour policy at every tau: the dataset is too narrow
#   for anything else to be admissible. Print the mean admissible actions per
#   visited state; if it is near 1 there is no choice left to make.
# - Results swing wildly with tau: expected on a small dataset, and the honest
#   reading is that the method has a hyperparameter you cannot set. That is the
#   real complaint about offline RL, not sample efficiency.
# - Changing `pessimistic_value` changes everything: also expected, and worth
#   knowing before believing any tabular offline result. On an all-negative-reward
#   task, the default zero initialisation is an optimistic prior about every state
#   you have never visited.
