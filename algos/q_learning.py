"""Tabular Q-learning (Watkins, 1989).

    python algos/q_learning.py

The update is one line:

    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

`max_a'` is what makes it **off-policy**: the target assumes the greedy action will
be taken next, regardless of what the agent's exploration actually does. SARSA
substitutes the action it really took, and that single difference produces two
visibly different policies on the cliff. See `algos/sarsa.py`.

Two things this file does that a textbook implementation usually does not, and both
are the reason the repo exists.

**It evaluates exactly.** Every point on the learning curve is
`policy_return(greedy(Q))` -- the true expected return of the current greedy policy,
from solving its Bellman equations. Not a moving average of sampled episodes. So a
curve that wobbles is a policy that changed, not a measurement that was noisy, and
two curves that differ, differ.

**It reports steps-to-threshold, not final return.** Final return hides the entire
point of most of these methods. Two algorithms that both reach optimal are not
equally good if one takes twenty times as long.
"""

from __future__ import annotations

import numpy as np

from envs.gridworld import GridWorld

GAMMA = 0.95
ALPHA = 0.5
EPSILON = 0.1
N_STEPS = 60_000
EVAL_EVERY = 2_000


def argmax_random_ties(values, rng):
    """argmax with ties broken uniformly at random.

    Not a nicety. `np.argmax` returns the FIRST maximal entry, and a Q table
    initialised to zeros is entirely ties -- so a deterministic argmax makes the
    agent choose action 0 in every unvisited state, forever. On a grid that means
    driving into the same wall 90% of the time and exploring only through epsilon.

    Measured before this was fixed: sparse-reward Q-learning on an 8x8 grid scored
    0.0000 against an optimum of 0.5133 and never improved. It still struggles
    afterwards -- that part is the real lesson -- but it was failing for two
    reasons and only one of them was interesting.
    """
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return argmax_random_ties(Q[state], rng)


def train(
    env,
    gamma=GAMMA,
    alpha=ALPHA,
    epsilon=EPSILON,
    n_steps=N_STEPS,
    eval_every=EVAL_EVERY,
    eval_env=None,
    seed=0,
):
    """Returns (Q, curve) where curve is [(steps, exact return), ...].

    `eval_env` scores the learned policy somewhere other than where it trained.
    That is essential for comparing reward functions: an agent trained on a shaped
    reward must be scored on the REAL task, or you are just measuring how large its
    bonus was. Defaults to `env`.
    """
    eval_env = eval_env or env
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    curve = [(0, eval_env.policy_return(Q.argmax(axis=1), gamma))]

    env.reset(seed=seed)
    for step in range(1, n_steps + 1):
        state = env.state
        action = epsilon_greedy(Q, state, epsilon, rng)
        _, reward, terminated, truncated, _ = env.step(action)

        # The `0.0 if terminated` is the terminal masking. One conditional, and
        # forgetting it silently corrupts every value in the table.
        target = reward + (0.0 if terminated else gamma * Q[env.state].max())
        Q[state, action] += alpha * (target - Q[state, action])

        if terminated or truncated:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        if step % eval_every == 0:
            curve.append((step, eval_env.policy_return(Q.argmax(axis=1), gamma)))

    return Q, curve


def steps_to_threshold(curve, threshold):
    """First evaluation point reaching `threshold`, or None if it never does.

    The number worth reporting. Final return says whether a method got there;
    this says what it cost.
    """
    for steps, value in curve:
        if value >= threshold:
            return steps
    return None


def main():
    env = GridWorld(size=8, reward_mode="dense", gamma=GAMMA)
    optimal = env.optimal_return(GAMMA)
    Q, curve = train(env, seed=0)

    print(f"GridWorld 8x8, dense reward, gamma={GAMMA}")
    print(f"exact optimal return: {optimal:.4f}\n")
    print(f"{'steps':>8} {'exact return':>14}")
    for steps, value in curve[:: max(len(curve) // 12, 1)]:
        print(f"{steps:>8,} {value:>14.4f}")

    reached = steps_to_threshold(curve, optimal - 1e-6)
    print(f"\nsteps to optimal: {reached:,}" if reached else "\nnever reached optimal")
    print("\nEvery number above is the exact return of the greedy policy at that")
    print("moment, not an average of sampled episodes. A wobble is a real change.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Return climbs then collapses: alpha is too high, so a single unlucky target
#   overwrites a good estimate. Tabular Q-learning tolerates alpha=0.5 on a
#   deterministic task and much less on a stochastic one.
# - Return is flat at zero forever on a sparse reward: the agent has never reached
#   the goal, so there is nothing to learn from and no learning rate helps. Print
#   the visit count of the goal state before touching any hyperparameter.
# - Learned Q looks plausible and the policy is wrong: compare against
#   `env.true_q(gamma)` cell by cell rather than eyeballing the curve.
# - Q values grow without bound: the terminal mask is missing, or gamma is 1 on a
#   task that need not terminate.
# - Training return looks worse than SARSA's: expected. Q-learning learns the
#   optimal policy and behaves worse while exploring. Evaluate the GREEDY policy
#   separately -- which is what the curve here already does.
