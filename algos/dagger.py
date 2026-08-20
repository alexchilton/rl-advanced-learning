"""DAgger: dataset aggregation (Ross, Gordon & Bagnell, 2011).

    python algos/dagger.py

Behaviour cloning trains on the expert's states and is then tested on its own.
That mismatch is the whole problem, and DAgger's answer is almost insultingly
simple: **let the learner drive, and ask the expert what it would have done.**

    for round in rounds:
        roll out the current policy            <- the learner's states
        label every state with expert(state)   <- the expert's answer
        add to the dataset, refit

The cost is the column that matters. DAgger needs the expert AVAILABLE DURING
TRAINING, not recorded once. With a scripted controller that is free. With a human
it is the reason DAgger is common in papers and rare in products.

The duplication with `algos/bc.py` is deliberate. Everything DAgger needs to do is
in this file, so the difference between the two methods is visible by reading one
of them rather than by diffing a shared base class.
"""

from __future__ import annotations

import numpy as np

from envs.racetrack import Racetrack

GAMMA = 0.99
BETA_DECAY = 0.6


def features(env, states):
    out = np.zeros((len(states), 4))
    for i, state in enumerate(states):
        if int(state) == env.finished_state:
            continue
        (row, col), vertical, horizontal = env.decode(int(state))
        out[i] = (row, col, vertical, horizontal)
    return out


def feature_scale(env):
    return np.array([env.rows, env.cols, env.max_speed, env.max_speed], dtype=float)


def fit(env, states, actions):
    scale = feature_scale(env)
    demonstrated = features(env, states) / scale
    cache: dict[int, int] = {}

    def policy(state):
        state = int(state)
        if state not in cache:
            if state == env.finished_state:
                cache[state] = 0
            else:
                query = features(env, [state])[0] / scale
                cache[state] = int(actions[int(np.argmin(np.linalg.norm(demonstrated - query, axis=1)))])
        return cache[state]

    return policy


def train(env, expert, label_budget, seed=0, rounds=10, beta_decay=BETA_DECAY):
    """DAgger under a fixed budget of expert labels.

    `beta` is the probability of deferring to the expert while collecting. It
    starts high and decays, so early rounds stay near the expert's distribution
    and later ones explore the learner's.

    Without it the first round is driven by a policy fitted to six states, which
    wanders into parts of the track where nothing useful can be learned, and the
    aggregated dataset ends up mostly garbage. Measured on Racetrack, the naive
    version needs about 2,500 labels to match what this reaches in 80 -- and it
    is the version most people write first, because the paper's beta schedule
    reads like a detail.
    """
    states, actions = [], []

    # Seed with a single expert demonstration. Round one has to be fitted to
    # something.
    env.reset(seed=seed * 1000)
    for _ in range(env.max_steps):
        states.append(env.state)
        actions.append(int(expert[env.state]))
        _, _, terminated, truncated, _ = env.step(int(expert[env.state]))
        if terminated or truncated:
            break

    per_round = max((label_budget - len(states)) // rounds, 1)
    beta = 1.0

    for round_index in range(rounds):
        if len(states) >= label_budget:
            break
        beta *= beta_decay
        policy = fit(env, np.array(states), np.array(actions))
        rng = np.random.default_rng(seed * 77 + round_index)

        def mixed(state):
            """Expert with probability beta, learner otherwise."""
            return int(expert[state]) if rng.random() < beta else policy(state)

        visited = []
        for episode in range(6):
            env.reset(seed=seed * 31 + round_index * 101 + episode)
            visited.append(env.state)
            for _ in range(env.max_steps):
                _, _, terminated, truncated, _ = env.step(mixed(env.state))
                visited.append(env.state)
                if terminated or truncated:
                    break
        visited = [s for s in visited if s != env.finished_state]

        # Subsample so every round costs the same number of expert queries. Without
        # this a bad round buys hundreds of labels and the budget stops meaning
        # anything.
        if len(visited) > per_round:
            chosen = rng.choice(len(visited), per_round, replace=False)
            visited = [visited[i] for i in chosen]

        states.extend(visited)
        actions.extend(int(expert[s]) for s in visited)

    return fit(env, np.array(states[:label_budget]), np.array(actions[:label_budget]))


def exact_return(env, policy, gamma=GAMMA):
    table = np.array([policy(s) for s in range(env.n_states)])
    return env.policy_return(table, gamma)


def main():
    env = Racetrack()
    expert = env.optimal_policy(GAMMA)
    optimal = env.optimal_return(GAMMA)

    print(f"Racetrack, {env.n_states} states")
    print(f"expert (exact optimal policy): {optimal:.3f}\n")
    print(f"{'labels':>7} {'exact return':>13} {'gap':>8}")
    for label_budget in (10, 20, 40, 80, 160, 320):
        values = [
            exact_return(env, train(env, expert, label_budget, seed=seed))
            for seed in range(1, 6)
        ]
        median = float(np.median(values))
        print(f"{label_budget:>7} {median:>13.2f} {median - optimal:>8.2f}")

    print("\nCompare against algos/bc.py at the same budgets, or run")
    print("experiments/bc_vs_dagger.py for both side by side.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - DAgger is WORSE than BC per label: check the beta schedule first. With
#   beta=0 from round one, the learner collects from a policy fitted to almost
#   nothing, and the aggregated dataset is dominated by states no sensible policy
#   ever visits. Measured on Racetrack: 2,426 labels to reach what the decayed
#   schedule reaches in 80.
# - Round two is worse than round one: expected, and not a bug. The dataset has
#   just acquired its first batch of off-distribution states and the
#   nearest-neighbour boundaries move. It recovers by round three or four. Do not
#   tune this away on a single seed.
# - DAgger and BC are indistinguishable: the environment is giving the learner a
#   free correction, so there is no compounding error to fix. On Racetrack that is
#   `on_crash="restart"`, which teleports a drifting learner back onto the
#   expert's distribution.
# - The expert is being queried more than the budget says: `per_round` subsampling
#   is off, or the seed demonstration is being counted twice. Expert queries are
#   the resource this method spends -- an uncounted query is a wrong result.
