"""Double Q-learning, tabular (van Hasselt, 2010).

    python algos/double_q.py

Q-learning's target uses max_a' Q(s', a'). If every Q(s', a') is an unbiased but
noisy estimate, their max is biased upwards: E[max] >= max E. The agent then
prefers whatever it has been lucky with.

Double Q-learning keeps two tables. On each step a coin picks one to update; that
table chooses the best next action, and the OTHER table says what it is worth:

    Q1(s,a) <- Q1(s,a) + alpha * (r + gamma * Q2(s', argmax_a' Q1(s', a')) - Q1(s,a))

The two tables' errors are independent, so the chosen action's lucky error is not
the one used to score it. Acting uses Q1 + Q2.

The demo is Sutton & Barto's Example 6.7 (`envs/maximization_bias.py`): the
fraction of episodes in which the agent goes LEFT from A, where left is worth
-0.1 and right 0. Epsilon-greedy alone goes left 5% of the time.
"""

from __future__ import annotations

import numpy as np

from envs.maximization_bias import A, LEFT, MaximizationBias

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 1.0
EPISODES = 300
RUNS = 2_000


def epsilon_greedy(values, n_valid, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_valid))
    values = values[:n_valid]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def run(env, double, episodes=EPISODES, alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, seed=0):
    """One run. Returns a boolean array: did episode i go left from A?"""
    rng = np.random.default_rng(seed)
    q1 = np.zeros((env.n_states, env.n_actions))
    q2 = np.zeros_like(q1)
    went_left = np.zeros(episodes, dtype=bool)
    valid = {0: 2, 1: env.n_b_actions, 2: 1}       # A offers left/right; B offers k

    for episode in range(episodes):
        env.reset(seed=int(rng.integers(2**31 - 1)))
        while True:
            s = env.state
            a = epsilon_greedy(q1[s] + q2[s] if double else q1[s], valid[s], epsilon, rng)
            if s == A:
                went_left[episode] = a == LEFT
            _, r, terminated, _, _ = env.step(a)
            s2 = env.state
            n2 = valid[s2]
            if not double:
                target = r + (0.0 if terminated else gamma * q1[s2, :n2].max())
                q1[s, a] += alpha * (target - q1[s, a])
            else:
                # ---- the Double Q change: one table selects, the other evaluates
                update, other = (q1, q2) if rng.random() < 0.5 else (q2, q1)
                best = int(np.argmax(update[s2, :n2]))
                target = r + (0.0 if terminated else gamma * other[s2, best])
                update[s, a] += alpha * (target - update[s, a])
            if terminated:
                break
    return went_left


def main():
    env = MaximizationBias()
    print(f"Maximisation-bias MDP, {env.n_b_actions} actions at B, alpha={ALPHA}, epsilon={EPSILON}")
    print(f"exact: Q(A, left) = {env.true_q(GAMMA)[A, LEFT]:+.2f}, Q(A, right) = {env.true_q(GAMMA)[A, 1]:+.2f}")
    print(f"{RUNS:,} runs of {EPISODES} episodes each. Epsilon-greedy alone goes left {EPSILON / 2:.0%}.\n")
    curves = {name: np.mean([run(env, double, seed=seed) for seed in range(RUNS)], axis=0)
              for name, double in (("Q-learning", False), ("Double Q", True))}
    print(f"{'episodes':>10} {'Q-learning % left':>18} {'Double Q % left':>16}")
    for lo, hi in ((0, 10), (10, 50), (50, 100), (100, 200), (200, 300)):
        cells = "".join(f"{curves[n][lo:hi].mean():>17.1%} " for n in curves)
        print(f"{f'{lo + 1}-{hi}':>10} {cells}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Double Q looks exactly like Q-learning: both tables are updated on every step
#   from the same sample, or the same table both selects and evaluates.
# - Double Q goes left LESS than 5%: it is acting greedily on one table only, and
#   that table has seen half the data. Act on Q1 + Q2.
# - No bias appears at all: B has too few actions or no reward noise. The bias
#   grows with the number of noisy estimates the max runs over.
