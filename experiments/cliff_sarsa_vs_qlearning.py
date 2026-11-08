"""SARSA vs Q-learning on the cliff (Sutton & Barto, Example 6.6).

    python experiments/cliff_sarsa_vs_qlearning.py

Same environment, same epsilon, same learning rate, same steps. The two methods
differ in one term of the target and learn different policies.

Two numbers per method, both exact:

  * greedy       the return of argmax Q, i.e. the policy each method claims is best
  * as followed  the return of the epsilon-greedy policy each method actually runs
                 while training, from `solvers.epsilon_greedy_policy`

Q-learning should win the first and lose the second. SARSA should do the
opposite. A method that wins both columns would mean the comparison is broken.

Then an epsilon sweep. In theory SARSA's policy approaches Q-learning's as
epsilon goes to 0, because the cost of walking the edge goes to 0. The sweep
measures whether that happens within this step budget.
"""

from __future__ import annotations

import numpy as np

from algos import q_learning, sarsa
from envs import solvers
from envs.cliff import CliffWalking

GAMMA = 0.99
ALPHA = 0.2  # see algos/sarsa.py: 0.5 leaves SARSA's greedy policy looping on 2 of 5 seeds
EPSILON = 0.1
N_STEPS = 30_000
SEEDS = range(5)


def run(method, env, epsilon, seed):
    Q, _ = method.train(env, gamma=GAMMA, alpha=ALPHA, epsilon=epsilon, n_steps=N_STEPS,
                        eval_every=N_STEPS, seed=seed)
    greedy = env.policy_return(Q.argmax(axis=1), GAMMA)
    followed = env.policy_return(solvers.epsilon_greedy_policy(Q, epsilon), GAMMA)
    steps = len(sarsa.greedy_path(env, Q)) - 1
    return Q, greedy, followed, steps


def main():
    env = CliffWalking()
    print("=" * 78)
    print("SARSA vs Q-learning on the cliff, exact returns")
    print("=" * 78)
    print(f"4x12 cliff, gamma={GAMMA}, alpha={ALPHA}, epsilon={EPSILON}, {N_STEPS:,} steps, "
          f"{len(SEEDS)} seeds")
    print(f"exact optimum (greedy, 13 steps): {env.optimal_return(GAMMA):.3f}\n")
    print(f"{'method':>11} {'seed':>5} {'greedy':>9} {'as followed':>12} {'path steps':>11}")

    results = {}
    for name, method in (("Q-learning", q_learning), ("SARSA", sarsa)):
        rows = [run(method, env, EPSILON, seed) for seed in SEEDS]
        results[name] = rows
        for seed, (_, greedy, followed, steps) in zip(SEEDS, rows):
            print(f"{name:>11} {seed:>5} {greedy:>9.3f} {followed:>12.3f} {steps:>11}")

    print("\nmedians")
    for name, rows in results.items():
        g = np.median([r[1] for r in rows])
        f = np.median([r[2] for r in rows])
        s = np.median([r[3] for r in rows])
        print(f"{name:>11}       {g:>9.3f} {f:>12.3f} {s:>11.0f}")

    print("\nThe policies, seed 0:\n")
    for name, rows in results.items():
        print(name)
        print(env.render_policy(rows[0][0]))
        print()

    print("=" * 78)
    print("Epsilon sweep: SARSA's path length, median of seeds")
    print("=" * 78)
    print(f"{'epsilon':>8} {'Q-learning steps':>17} {'SARSA steps':>12}")
    for epsilon in (0.2, 0.1, 0.05, 0.01):
        q_steps = np.median([run(q_learning, env, epsilon, s)[3] for s in SEEDS])
        s_steps = np.median([run(sarsa, env, epsilon, s)[3] for s in SEEDS])
        print(f"{epsilon:>8.2f} {q_steps:>17.0f} {s_steps:>12.0f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Both methods take 13 steps: SARSA is bootstrapping from max Q, or epsilon is 0.
# - Q-learning's greedy path is longer than 13: it has not converged, or ties are
#   broken deterministically (see `algos/q_learning.py`).
# - 'as followed' is about -100 for both: the evaluation is treating the cliff
#   fall as terminal, or gamma is 1 with a policy that can loop forever.
