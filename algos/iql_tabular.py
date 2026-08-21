"""Implicit Q-Learning, tabular (Kostrikov, Nair & Levine, 2021).

    python algos/iql_tabular.py

CQL pushes down the Q of out-of-distribution actions. BCQ refuses to maximise
over them. IQL does something cleverer: it never evaluates them.

The trick is to stop taking a max over actions at all. Instead, learn a separate
state-value function by **expectile regression** on the Q values of actions the
dataset actually took:

    V(s)   <- expectile_tau of  Q(s,a)   over a ~ dataset
    Q(s,a) <-  r + gamma * V(s')

Look at what is absent. The Q update reads `V(s')`, a single number, not
`max_a' Q(s',a')`. No out-of-sample action appears anywhere in the target, so
there is nothing to be conservative about and no penalty to tune.

The expectile is what recovers optimism. Ordinary least squares fits the MEAN of
the Q values under the behaviour policy, which would give you policy evaluation
of the behaviour policy -- correct, but useless. Expectile regression at
tau -> 1 weights over-predictions more and more heavily, so V approaches the
maximum over the actions PRESENT in the data. It is an in-sample max.

    tau = 0.5   plain least squares. V is the behaviour policy's value.
    tau = 0.7   mildly optimistic.
    tau = 0.9   close to an in-sample max. The paper's usual range.
    tau -> 1    the in-sample max, and increasingly high variance because fewer
                and fewer samples carry any weight.

That is the trade the whole method rests on, and it is a bias-variance dial like
every other one in this repo.
"""

from __future__ import annotations

import numpy as np

from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
ITERATIONS = 300
LEARNING_RATE = 0.5
EXPECTILE = 0.9
BETA = 3.0


def train(
    dataset,
    expectile=EXPECTILE,
    gamma=GAMMA,
    iterations=ITERATIONS,
    learning_rate=LEARNING_RATE,
):
    """Returns (Q, V). Neither update ever touches an action outside the dataset."""
    Q = np.zeros((dataset.n_states, dataset.n_actions))
    V = np.zeros(dataset.n_states)

    counts = np.zeros_like(Q)
    np.add.at(counts, (dataset.states, dataset.actions), 1.0)
    seen = counts > 0

    state_counts = np.zeros(dataset.n_states)
    np.add.at(state_counts, dataset.states, 1.0)
    visited = state_counts > 0

    for _ in range(iterations):
        # --- V by expectile regression on in-sample Q ------------------
        # Loss is |tau - 1{u < 0}| * u^2 with u = Q(s,a) - V(s), so the gradient
        # with respect to V is -2 * weight * u. Over-predictions (u > 0) get
        # weight tau, under-predictions get 1 - tau. At tau = 0.5 this is least
        # squares and V becomes the behaviour policy's value.
        residual = Q[dataset.states, dataset.actions] - V[dataset.states]
        weight = np.where(residual > 0, expectile, 1.0 - expectile)
        gradient = np.zeros(dataset.n_states)
        np.add.at(gradient, dataset.states, weight * residual)
        V[visited] += learning_rate * gradient[visited] / state_counts[visited]

        # --- Q towards r + gamma * V(s') -------------------------------
        # No max, no argmax, no action from outside the dataset.
        target = dataset.rewards + np.where(
            dataset.terminated, 0.0, gamma * V[dataset.next_states]
        )
        totals = np.zeros_like(Q)
        np.add.at(totals, (dataset.states, dataset.actions), target)
        mean_target = np.divide(totals, counts, out=np.zeros_like(Q), where=seen)
        Q[seen] += learning_rate * (mean_target[seen] - Q[seen])

    return Q, V


def extract_policy(dataset, Q, V, beta=BETA):
    """Advantage-weighted policy extraction, then take the mode.

    IQL's Q is only meaningful on actions the dataset took, so an unrestricted
    argmax would read entries the training never touched -- the exact mistake the
    method exists to avoid, reintroduced at the last step. Weight each observed
    action by exp(beta * A(s,a)) instead, and pick the heaviest.

    `beta` is temperature: 0 gives behaviour cloning, large gives the in-sample
    argmax.
    """
    advantage = Q - V[:, None]
    counts = dataset.action_counts()

    # Stabilised softmax: subtract the per-state maximum before exponentiating.
    # Clipping the advantage alone is not enough -- it is `beta * advantage` that
    # overflows, and at beta=100 that happens well inside a plausible range.
    scaled = np.where(counts > 0, beta * advantage, -np.inf)
    has_data = counts.sum(axis=1, keepdims=True) > 0
    # Rows with no data are all -inf, and -inf minus -inf is a nan. Guard the
    # shift so the empty rows stay -inf rather than becoming nan.
    row_max = np.where(has_data, np.max(scaled, axis=1, keepdims=True, initial=-np.inf), 0.0)
    weights = np.where(counts > 0, np.exp(scaled - row_max) * counts, 0.0)

    policy = weights.argmax(axis=1)
    unseen = weights.sum(axis=1) == 0
    policy[unseen] = Q[unseen].argmax(axis=1)
    return policy.astype(np.int64)


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)
    # The discriminating dataset. BC scores -99.96 on random data, so the dials
    # below have somewhere to move. On "medium" every setting scores about -6.5
    # and the demo shows nothing at all.
    QUALITY = "random"
    dataset = datasets.build(env, QUALITY, n_transitions=20_000, gamma=GAMMA, seed=0)
    behaviour = env.policy_return(datasets.behaviour_policy_for(env, QUALITY, GAMMA), GAMMA)

    print(f"Racetrack. Exact optimal return: {optimal:.2f}")
    print(f"Dataset ({QUALITY}): {dataset.summary()}")
    print(f"Behaviour policy that produced it: {behaviour:.2f}\n")

    print("The expectile dial: from policy evaluation to an in-sample max.\n")
    print(f"{'tau':>6} {'V(start)':>10} {'actually gets':>14} {'note':>34}")
    for expectile in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        Q, V = train(dataset, expectile=expectile)
        policy = extract_policy(dataset, Q, V)
        actual = env.policy_return(policy, GAMMA)
        start_value = float(np.dot(env.start_distribution, V))

        if expectile == 0.5:
            note = "least squares: the data's own value"
        elif actual >= optimal - 1.0:
            note = "near optimal"
        elif actual > behaviour:
            note = "beats the data"
        else:
            note = "worse than the data"
        print(f"{expectile:>6.2f} {start_value:>10.2f} {actual:>14.2f} {note:>34}")

    print("\nAt tau=0.5 the expectile is a mean, V is the behaviour policy's value, and")
    print("the method is doing policy evaluation rather than improvement. Raising tau")
    print("makes V approach the best action PRESENT IN THE DATA at each state, which")
    print("is the whole idea: an in-sample max, obtained without ever writing a max.")
    print()
    print("But watch the top of the range. Past about tau=0.9 the return collapses,")
    print("and V(start) climbs to -1.47 -- which is the same optimistic lie that naive")
    print("offline Q-learning tells in algos/offline_q.py. The mechanism is different")
    print("and the symptom is identical: at tau close to 1 the expectile is carried by")
    print("one or two transitions per state, so V overestimates, and the Q target")
    print("inherits the overestimate. IQL does not remove the failure mode. It moves it")
    print("from 'unseen actions' to 'thinly-seen actions', where a hyperparameter")
    print("controls it instead of an initialisation.")

    print("\nThe beta dial, at tau=0.9 -- how sharply to prefer high-advantage actions:\n")
    print(f"{'beta':>6} {'actually gets':>14} {'note':>34}")
    Q, V = train(dataset, expectile=0.9)
    for beta in (0.0, 0.5, 1.0, 3.0, 10.0, 100.0):
        actual = env.policy_return(extract_policy(dataset, Q, V, beta=beta), GAMMA)
        note = "behaviour cloning" if beta == 0.0 else ("in-sample argmax" if beta >= 100 else "")
        print(f"{beta:>6.1f} {actual:>14.2f} {note:>34}")

    print("\nCompare the three offline methods now in this repo. CQL adds a penalty and")
    print("has alpha. BCQ restricts the max and has tau. IQL removes the max and has an")
    print("expectile and a temperature. All three interpolate between trusting the Q")
    print("function and trusting the data, and none of them can be tuned without an")
    print("environment to test on -- which offline RL, by definition, does not have.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - IQL performs exactly like behaviour cloning: the expectile is near 0.5, so V
#   is fitting the mean and no improvement is possible. Print V(start) against the
#   behaviour policy's exact return -- if they match, that is what is happening.
# - IQL performs like naive offline Q-learning: the policy extraction is taking an
#   unrestricted argmax over Q. Q was never trained on unseen actions, so those
#   entries are still at their initialisation, and reading them at the last step
#   throws away everything the method did.
# - High tau makes it unstable: fewer and fewer samples carry weight in the
#   expectile, so V's estimate at rarely-visited states is driven by one or two
#   transitions. This is variance, not a bug, and it is the reason the paper does
#   not simply use tau = 0.999.
# - V exceeds every Q at a state: impossible for tau <= 1 at convergence. If it
#   happens, the expectile weights are the wrong way round -- over-predictions
#   must carry weight tau, not 1 - tau.
