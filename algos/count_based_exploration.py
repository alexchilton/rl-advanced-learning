"""Count-based exploration, tabular (Strehl & Littman 2008; Bellemare et al. 2016).

    python algos/count_based_exploration.py

On `Chain(n)` the only reward is at the far end, and one "left" sends you back to
the start. A uniformly random policy reaches the goal with probability 2^-(n-1)
per episode. Epsilon-greedy is that random policy until something is learned, so
it waits for a lottery win.

A count bonus makes the agent seek out what it has tried least:

    r_total = r + beta / sqrt(N(s, a))

Untried (s, a) pairs are worth more than tried ones, so the greedy policy itself
heads for the frontier. Exploration becomes directed instead of random.

Bellemare et al. 2016 is about the case where states never repeat (Atari pixels)
and N(s, a) is always 1. They derive a *pseudo-count* from a density model, which
generalises between similar states. With a finite state space the true count is
available, so this file is the exact special case of their bonus. RND
(`algos/rnd.py`) is the neural version for large spaces.

`q_init` gives optimistic initialisation, the cheapest directed exploration there
is: start every Q above any achievable return, and every untried action looks
best until tried. `reset` lets a curriculum choose start states
(`algos/curriculum_reverse.py`).
"""

from __future__ import annotations

import numpy as np

from envs.chain import Chain

GAMMA = 0.99
ALPHA = 0.5
EPSILON = 0.1


def train(env, n_steps=50_000, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, bonus=0.0, q_init=0.0,
          reset=None, eval_env=None, eval_every=1_000, seed=0, Q=None):
    """Tabular Q-learning with an optional count bonus and optimistic start.

    Returns (Q, curve, first_goal_step). `curve` holds (step, exact return of the
    greedy policy on `eval_env`). `first_goal_step` is when a terminating
    transition was first seen, or None. `reset(env, rng)` chooses each episode's
    start; default is the environment's own start distribution. `Q` warm-starts
    from an existing table (the size curriculum uses it).
    """
    eval_env = eval_env or env
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, env.n_actions), float(q_init)) if Q is None else Q.copy()
    counts = np.zeros((env.n_states, env.n_actions))

    def start():
        env.reset(seed=int(rng.integers(2**31 - 1)))
        if reset is not None:
            env.state = int(reset(env, rng))

    def greedy_return():
        return eval_env.policy_return(Q.argmax(axis=1), gamma)

    curve, first_goal = [(0, greedy_return())], None
    start()
    for step in range(1, n_steps + 1):
        s = env.state
        if rng.random() < epsilon:
            a = int(rng.integers(env.n_actions))
        else:
            best = np.flatnonzero(Q[s] == Q[s].max())
            a = int(rng.choice(best))
        _, r, terminated, truncated, _ = env.step(a)
        counts[s, a] += 1
        # ---- the count bonus: shrinks as the pair is tried
        r_total = r + bonus / np.sqrt(counts[s, a])
        target = r_total + (0.0 if terminated else gamma * Q[env.state].max())
        Q[s, a] += alpha * (target - Q[s, a])
        if terminated and first_goal is None:
            first_goal = step
        if terminated or truncated:
            start()
        if step % eval_every == 0:
            curve.append((step, greedy_return()))
    return Q, curve, first_goal


def main():
    print("Chain(n): reward 1 only at the far end; 'left' returns to the start.")
    print("Steps until the goal is first reached, median of 5 seeds (None = never, budget 200,000)\n")
    print(f"{'n':>4} {'random walk P(goal)':>20} {'epsilon-greedy':>15} {'optimistic Q':>13} {'count bonus':>12}")
    for n in (5, 10, 15, 20, 30):
        env = Chain(n=n, max_steps=4 * n)
        row = []
        for kwargs in (dict(), dict(q_init=1.0), dict(bonus=0.1)):
            hits = [train(env, n_steps=200_000, eval_every=200_000, seed=seed, **kwargs)[2] for seed in range(5)]
            found = [h for h in hits if h is not None]
            row.append(f"{int(np.median(found)):,}" if len(found) >= 3 else f"never ({len(found)}/5)")
        print(f"{n:>4} {env.random_walk_success_probability():>20.2e} {row[0]:>15} {row[1]:>13} {row[2]:>12}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The bonus never fades and the agent keeps wandering after finding the goal:
#   beta is too large relative to the reward. The bonus must shrink below the
#   value differences that matter. Print the bonus at the final counts.
# - Counted per state instead of per (state, action): the agent stops trying a
#   new action in a state it has visited often. Count the pair.
# - Optimistic init with gamma near 1 and no bonus decay: works, but Q values
#   start at q_init everywhere and take many visits to come down. Set q_init just
#   above the best achievable return, not at 1e6.
# - Works on the chain, nothing on pixels: every image is new, so every count is
#   1 and the bonus is constant. That is what pseudo-counts and RND are for.
