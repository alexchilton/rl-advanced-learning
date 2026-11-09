"""Dyna-Q (Sutton, 1990): learn a model from experience, then plan with it.

    python algos/dyna_q.py

Q-learning uses each real transition once. Dyna-Q also stores it in a model,
model[s, a] = (r, s', terminated), and after every real step replays
`planning_steps` remembered transitions, chosen at random, through the same
Q-learning update:

    real step:      Q(s,a) += alpha (r + gamma max Q(s') - Q(s,a));  model[s,a] = (r, s')
    n times:        pick a remembered (s,a); (r, s') = model[s,a]; same update

Real experience is expensive and planning is cheap, so the planning steps trade
computation for data. With n = 0 this is Q-learning.

The model here is a table of the last outcome seen for each (s, a), which is
exact on a deterministic task and wrong on a stochastic one: it remembers one
sample, not the distribution.
"""

from __future__ import annotations

import numpy as np

from envs.gridworld import GridWorld

GAMMA = 0.95
ALPHA = 0.5
EPSILON = 0.1


def train(env, n_steps=20_000, planning_steps=10, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, eval_every=100, seed=0):
    """Returns (Q, curve) with curve = [(real steps, exact return of the greedy policy)]."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    model = {}

    def update(s, a, r, s2, terminated):
        target = r + (0.0 if terminated else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])

    curve = [(0, env.policy_return(Q.argmax(1), gamma))]
    env.reset(seed=seed)
    for step in range(1, n_steps + 1):
        s = env.state
        if rng.random() < epsilon:
            a = int(rng.integers(env.n_actions))
        else:
            a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        _, r, terminated, truncated, _ = env.step(a)
        update(s, a, r, env.state, terminated)
        # ---- the Dyna change: remember, then replay remembered transitions
        model[(s, a)] = (r, env.state, terminated)
        keys = list(model)
        for _ in range(planning_steps):
            ps, pa = keys[int(rng.integers(len(keys)))]
            pr, ps2, pterm = model[(ps, pa)]
            update(ps, pa, pr, ps2, pterm)
        if terminated or truncated:
            env.reset(seed=int(rng.integers(2**31 - 1)))
        if step % eval_every == 0:
            curve.append((step, env.policy_return(Q.argmax(1), gamma)))
    return Q, curve


def main():
    env = GridWorld(size=10, reward_mode="sparse", walls=[(r, 4) for r in range(0, 8)], max_steps=300)
    optimal = env.optimal_return(GAMMA)
    print(f"GridWorld 10x10 with a wall, sparse reward. Exact optimum {optimal:.4f}")
    print("Real environment steps until the greedy policy is optimal, 5 seeds\n")
    print(f"{'planning steps':>15} {'median':>8} {'per seed':>40}")
    for n in (0, 5, 20, 50):
        hits = []
        for seed in range(5):
            _, curve = train(env, n_steps=20_000, planning_steps=n, seed=seed)
            hits.append(next((s for s, v in curve if v >= optimal - 1e-6), None))
        found = [h for h in hits if h is not None]
        print(f"{n:>15} {int(np.median(found)) if found else 'never':>8} {str(hits):>40}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Planning makes it WORSE on a stochastic task: the table keeps the last outcome
#   only, so planning replays one sample as if it were certain. Store counts, or
#   sample from all outcomes seen.
# - The environment changes and the agent does not notice (Sutton & Barto's
#   "blocking maze"): the model keeps replaying the old world. Dyna-Q+ adds a
#   bonus for transitions not tried recently.
# - No speed-up at all: planning picks from (s, a) pairs that carry no new
#   information. Prioritised sweeping plans backwards from where values changed.
