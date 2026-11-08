"""Tabular SARSA (Rummery & Niranjan, 1994).

    python algos/sarsa.py

The update is Q-learning's with one substitution:

    Q-learning  Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    SARSA       Q(s,a) <- Q(s,a) + alpha * (r + gamma *       Q(s',a') - Q(s,a))

where a' is the action the agent WILL ACTUALLY TAKE next, drawn from its own
epsilon-greedy policy before the update. That makes it **on-policy**: SARSA learns
the value of the policy it is following, exploration included. Q-learning learns
the value of the greedy policy it is not following.

On the cliff that one substitution produces two different policies. Q-learning
walks the edge, because the greedy policy never steps off. SARSA walks one row
further up, because its own epsilon-greedy policy sometimes does.
`experiments/cliff_sarsa_vs_qlearning.py` measures both.

The duplication with `algos/q_learning.py` is deliberate. The difference between
the two methods is two lines, and it should be visible by reading one file.
"""

from __future__ import annotations

import numpy as np

from envs.cliff import CliffWalking

GAMMA = 0.99
# 0.5 is Q-learning's default and is too high here: SARSA's target contains the
# random next action, so Q stays noisy. Measured on the cliff at 30,000 steps:
# alpha=0.5 leaves a looping greedy policy on 2 of 5 seeds; 0.2 on 0 of 5.
ALPHA = 0.2
EPSILON = 0.1
N_STEPS = 30_000
EVAL_EVERY = 1_000


def argmax_random_ties(values, rng):
    """argmax with ties broken uniformly at random. See `algos/q_learning.py`."""
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
    seed=0,
):
    """Returns (Q, curve) where curve is [(steps, exact return of greedy policy), ...]."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    curve = [(0, env.policy_return(Q.argmax(axis=1), gamma))]

    env.reset(seed=seed)
    action = epsilon_greedy(Q, env.state, epsilon, rng)
    for step in range(1, n_steps + 1):
        state = env.state
        _, reward, terminated, truncated, _ = env.step(action)

        # The whole difference from Q-learning: choose the NEXT action first, then
        # bootstrap from it. Using Q[s'].max() here instead turns this file into
        # Q-learning, silently.
        next_action = epsilon_greedy(Q, env.state, epsilon, rng)
        target = reward + (0.0 if terminated else gamma * Q[env.state, next_action])
        Q[state, action] += alpha * (target - Q[state, action])
        action = next_action

        if terminated or truncated:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            action = epsilon_greedy(Q, env.state, epsilon, rng)
        if step % eval_every == 0:
            curve.append((step, env.policy_return(Q.argmax(axis=1), gamma)))

    return Q, curve


def greedy_path(env, Q, max_steps=100):
    """Cells visited by the greedy policy from the start. Deterministic when slip=0."""
    env.reset(seed=0)
    path = [env.cell(env.state)]
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(int(Q[env.state].argmax()))
        path.append(env.cell(env.state))
        if terminated or truncated:
            break
    return path


def main():
    env = CliffWalking()
    Q, curve = train(env, seed=0)
    path = greedy_path(env, Q)

    print(f"CliffWalking 4x12, gamma={GAMMA}, epsilon={EPSILON}, {N_STEPS:,} steps\n")
    print(env.render_policy(Q))
    print(f"\ngreedy path: {len(path) - 1} steps, rows used {sorted({r for r, _ in path[1:-1]})}")
    print(f"exact return of greedy policy: {curve[-1][1]:.3f}")
    print(f"exact optimum:                 {env.optimal_return(GAMMA):.3f}")
    print("\nThe path is longer than the optimal 13 steps because it keeps away from")
    print("the edge. See experiments/cliff_sarsa_vs_qlearning.py for both methods.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - SARSA learns the same edge-hugging path as Q-learning: the target is using
#   Q[s'].max() instead of Q[s', a'] for the action actually taken next, or
#   epsilon is 0. Both turn SARSA into Q-learning with no error.
# - The next action is chosen AFTER the update: then a' was picked from a Q that
#   already includes this step's change. Small effect, but it is no longer the
#   algorithm in the paper.
# - Return collapses whenever epsilon decays: expected. SARSA's Q is the value of
#   the exploring policy; change epsilon and the policy it was valuing is gone.
#   It re-converges, more slowly than Q-learning would.
# - The greedy policy loops and never reaches the goal (exact return -100): alpha
#   is too high for a stochastic target. Measured on the cliff, 30,000 steps:
#   alpha=0.5 loops on 2 of 5 seeds (a 2-cycle between (1,3) and (2,3), or into
#   the top wall); alpha=0.2, or alpha=0.5 for 100,000 steps, loops on none.
