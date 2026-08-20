"""Offline Q-learning, and the way it fails.

    python algos/offline_q.py

Fitted Q iteration on a fixed dataset. The update is the one from `q_learning.py`
with the environment removed:

    Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

Nothing about that looks dangerous. It is, and the danger is in `max_a'`.

The max runs over ALL actions, including actions the dataset never took at s'.
Those entries were never updated by anything, so they hold whatever the
initialisation left there. The max then selects the largest of them, that value
propagates backwards through the whole table, and the final greedy policy
consists mostly of actions no one has ever tried.

This is extrapolation error, and it is why offline RL is a separate field rather
than online RL with the data collection deleted.

**On Racetrack you do not need any trick to see it.** Every reward is -1, so
every honest Q value is negative, and an untouched entry sitting at its zero
initialisation is the most optimistic number in the table. Q init = 0 IS the
optimism. On a task with positive rewards you would have to initialise
optimistically to reproduce this in tabular form -- which is worth knowing,
because it means a tabular test of this bug can pass while the deep version is
broken. A network's Q at an unseen action is arbitrary in either direction.

The fix is `algos/cql_tabular.py`.
"""

from __future__ import annotations

import numpy as np

from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
ITERATIONS = 300
LEARNING_RATE = 0.5


def train(dataset, gamma=GAMMA, iterations=ITERATIONS, learning_rate=LEARNING_RATE, seed=0):
    """Fitted Q iteration. Returns the Q table.

    Swept over the whole dataset each iteration rather than sampling minibatches,
    because the point here is the bootstrapping target and not the optimiser.
    """
    Q = np.zeros((dataset.n_states, dataset.n_actions))
    for _ in range(iterations):
        bootstrap = np.where(dataset.terminated, 0.0, gamma * Q[dataset.next_states].max(axis=1))
        target = dataset.rewards + bootstrap
        # Average the target over duplicate (s, a) pairs, then move Q towards it.
        totals = np.zeros_like(Q)
        counts = np.zeros_like(Q)
        np.add.at(totals, (dataset.states, dataset.actions), target)
        np.add.at(counts, (dataset.states, dataset.actions), 1.0)
        seen = counts > 0
        mean_target = np.divide(totals, counts, out=np.zeros_like(Q), where=seen)
        Q[seen] += learning_rate * (mean_target[seen] - Q[seen])
    return Q


def out_of_support_fraction(Q, dataset) -> float:
    """How often the greedy action is one the data never took, where it matters.

    Weighted by how often the dataset visits each state. An unweighted count over
    all states is useless here: with a coverage of 6% most states are unreachable
    and their greedy action is arbitrary forever, so the number sits at 0.87
    whether the policy is optimal or catastrophic. It has to be measured where the
    agent actually goes.
    """
    support = dataset.support()
    greedy = Q.argmax(axis=1)
    in_support = support[np.arange(len(greedy)), greedy].astype(float)

    weights = np.zeros(dataset.n_states)
    np.add.at(weights, dataset.states, 1.0)
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float(1.0 - (weights @ in_support) / total)


def value_gap(Q, env, dataset, gamma=GAMMA) -> tuple[float, float]:
    """(what the agent thinks it will get, what it actually gets).

    Both exact. The distance between them is extrapolation error, in return units.
    An agent that believes it will score -7 and scores -95 has not mis-tuned a
    hyperparameter.
    """
    greedy = Q.argmax(axis=1)
    predicted = float(np.dot(env.start_distribution, Q.max(axis=1)))
    actual = env.policy_return(greedy, gamma)
    return predicted, actual


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)
    print(f"Racetrack, {env.n_states} states. Exact optimal return: {optimal:.2f}\n")
    print(f"{'dataset':>8} {'transitions':>12} {'coverage':>9} {'OOS greedy':>11} "
          f"{'believes':>10} {'actually gets':>14}")

    for quality in datasets.QUALITIES:
        dataset = datasets.build(env, quality, n_transitions=20_000, gamma=GAMMA, seed=0)
        Q = train(dataset)
        predicted, actual = value_gap(Q, env, dataset)
        print(
            f"{quality:>8} {len(dataset):>12,} {dataset.coverage():>9.3f} "
            f"{out_of_support_fraction(Q, dataset):>11.2f} {predicted:>10.2f} {actual:>14.2f}"
        )

    print("\n'believes' is the value the learned Q claims for the start state.")
    print("'actually gets' is the exact return of the policy it implies.")
    print("The gap between them is extrapolation error, and no amount of extra")
    print("training closes it -- more iterations propagate the untouched entries")
    print("further, not less far.")
    print("\nSee algos/cql_tabular.py for the fix.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Offline Q-learning works fine and you expected it to fail: check the sign of
#   the rewards. With rewards >= 0 and Q initialised to 0, an untouched entry is
#   the most PESSIMISTIC number in the table, so the max never selects it and the
#   failure never appears. Tabular tests of extrapolation error are sensitive to
#   this in a way the deep version is not.
# - The predicted value looks reasonable: it always does. That is the point. The
#   agent's own estimate is the thing that has been corrupted, so it cannot be
#   used to detect the corruption. Compare against an exact evaluation, or against
#   the behaviour policy's return, but never against the agent's own number.
# - 'OOS greedy' near zero but the return is still poor: the dataset covers the
#   actions but not the states worth reaching. Coverage of (s,a) pairs and
#   coverage of good trajectories are different things.
