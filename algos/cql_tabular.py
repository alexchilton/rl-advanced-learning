"""Conservative Q-Learning, tabular (Kumar et al. 2020).

    python algos/cql_tabular.py

`algos/offline_q.py` fails because `max_a' Q(s',a')` selects actions the dataset
never took, whose Q values were never grounded in anything. CQL's answer is to
make those values small on purpose, by adding one term to the loss:

    alpha * ( logsumexp_a Q(s,a)  -  Q(s, a_data) )

Read it as: push DOWN the Q of every action at s, push UP the Q of the action the
data actually took there. The soft-max weighting means the push-down lands hardest
on whichever action currently looks best, which is exactly the one the extrapolation
error inflated.

`alpha` is the whole dial and it interpolates between two failure modes.

    alpha = 0     naive offline Q-learning. Believes -1, gets -100.
    alpha small   the conservatism is too weak to suppress the untouched entries.
    alpha right   the policy stays where the data is, and improves on it.
    alpha large   Q is dominated by "do what the data did". This is behaviour
                  cloning with extra steps, and it inherits BC's ceiling.

That last row is the honest part. A large enough conservatism penalty does not
make offline RL safe, it makes it stop being RL.
"""

from __future__ import annotations

import numpy as np

from algos.offline_q import out_of_support_fraction, value_gap
from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
ITERATIONS = 300
LEARNING_RATE = 0.5
ALPHA = 1.0


def train(dataset, alpha=ALPHA, gamma=GAMMA, iterations=ITERATIONS, learning_rate=LEARNING_RATE):
    """Fitted Q iteration plus the conservative penalty. Returns the Q table.

    With alpha=0 this is exactly `offline_q.train`, which is deliberate: the two
    should be one edit apart so the penalty is the only difference on screen.
    """
    Q = np.zeros((dataset.n_states, dataset.n_actions))
    seen_counts = np.zeros_like(Q)
    np.add.at(seen_counts, (dataset.states, dataset.actions), 1.0)
    seen = seen_counts > 0

    state_counts = np.zeros(dataset.n_states)
    np.add.at(state_counts, dataset.states, 1.0)
    visited = state_counts > 0

    for _ in range(iterations):
        # --- the ordinary TD update -----------------------------------
        bootstrap = np.where(dataset.terminated, 0.0, gamma * Q[dataset.next_states].max(axis=1))
        target = dataset.rewards + bootstrap
        totals = np.zeros_like(Q)
        np.add.at(totals, (dataset.states, dataset.actions), target)
        mean_target = np.divide(totals, seen_counts, out=np.zeros_like(Q), where=seen)
        Q[seen] += learning_rate * (mean_target[seen] - Q[seen])

        if alpha == 0.0:
            continue

        # --- the conservative penalty ---------------------------------
        # d/dQ of alpha * (logsumexp_a Q(s,a) - Q(s,a_data))
        #   = alpha * (softmax(Q(s,.)) - onehot(a_data))
        shifted = Q[visited] - Q[visited].max(axis=1, keepdims=True)
        soft = np.exp(shifted)
        soft /= soft.sum(axis=1, keepdims=True)

        gradient = np.zeros_like(Q)
        gradient[visited] = soft * state_counts[visited, None]
        np.subtract.at(gradient, (dataset.states, dataset.actions), 1.0)
        gradient[visited] /= state_counts[visited, None]

        Q -= learning_rate * alpha * gradient

    return Q


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)
    dataset = datasets.build(env, "medium", n_transitions=20_000, gamma=GAMMA, seed=0)

    behaviour_value = env.policy_return(
        datasets.behaviour_policy_for(env, "medium", GAMMA), GAMMA
    )

    print(f"Racetrack. Exact optimal return: {optimal:.2f}")
    print(f"Dataset: {dataset.summary()}")
    print(f"Behaviour policy that produced it: {behaviour_value:.2f}\n")
    print("The alpha dial, from naive offline Q to behaviour cloning with extra steps:\n")
    print(f"{'alpha':>8} {'OOS greedy':>11} {'believes':>10} {'actually gets':>14} {'verdict':>28}")

    for alpha in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        Q = train(dataset, alpha=alpha)
        oos = out_of_support_fraction(Q, dataset)
        predicted, actual = value_gap(Q, env, dataset, GAMMA)

        if actual <= -99:
            verdict = "never finishes"
        elif actual >= optimal - 1.0:
            verdict = "near optimal"
        elif actual > behaviour_value:
            verdict = "beats the data, short of optimal"
        else:
            verdict = "worse than the data"
        print(f"{alpha:>8.1f} {oos:>11.2f} {predicted:>10.2f} {actual:>14.2f} {verdict:>28}")

    print("\nAt alpha=0 the agent believes it will score about -1 and scores -100.")
    print("Raising alpha suppresses the untouched entries, the greedy policy moves")
    print("back inside the data's support, and the predicted value stops being a lie.")
    print("\nPush alpha far enough and Q just reproduces what the data did, which is")
    print("behaviour cloning wearing a Q function. Conservatism does not make offline")
    print("RL safe. It trades away the thing that made it worth doing.")
    print("\nWatch the 'believes' column on the way. It is a lie at alpha=0 (-1.84")
    print("against a real -98.73), becomes honest around alpha=0.5, and then goes")
    print("nonsense in the other direction: at alpha=100 it reports +536 on a task")
    print("where no return can exceed 0. Past a point the penalty overwhelms the TD")
    print("term and Q stops estimating return at all -- it becomes a score for 'did")
    print("the data do this'. The POLICY it implies is still meaningful. The numbers")
    print("are not. Never read a conservative Q as a value prediction.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - No alpha helps: the dataset does not contain a good trajectory to find. Print
#   the behaviour policy's exact return first. Offline RL can stitch better paths
#   out of mediocre pieces, but it cannot invent a state nobody ever reached.
# - The best alpha changes completely between datasets: expected, and the reason
#   offline RL is hard to deploy. There is no validation environment to tune it
#   on -- that would be online interaction. Tuning alpha against the true return,
#   as this file does, is cheating, and it is worth being explicit that the demo
#   cheats so the difficulty is visible.
# - 'believes' matches 'actually gets' but both are poor: the conservatism is too
#   strong. Q has been flattened towards the behaviour policy and the agent is now
#   honest about being mediocre.
# - Q goes to -inf: alpha is large and the penalty is being applied without the
#   softmax normalisation, so every action is pushed down every sweep with nothing
#   pushing back.
