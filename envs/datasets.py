"""Offline datasets: a fixed log of transitions, and no environment to poke.

Offline RL's defining constraint is that you cannot try anything. Whatever the
behaviour policy did is all you will ever see, and the quality of that log turns
out to matter more than the choice of algorithm.

`quality` is the axis worth sweeping, and it is the one comparison that explains
why offline RL exists as a separate field:

    expert   the optimal policy. Behaviour cloning wins here, and every
             conservative offline method is just overhead.
    medium   the optimal policy with substantial exploration noise. Mediocre but
             coherent. This is what a deployed heuristic looks like.
    random   uniform actions. Almost no reward, but broad coverage.
    mixed    a little expert and a lot of random. BC collapses on this, because
             it averages good and bad behaviour into something that is neither.
             Offline RL does not, because it can tell them apart using the
             reward. This row is the argument.

Lives in `envs/` rather than `algos/` because a dataset is data. Every offline
algorithm reads the same one, which is the only way their comparison means
anything.
"""

from __future__ import annotations

import numpy as np

QUALITIES = ("expert", "medium", "random", "mixed")


class Dataset:
    """A fixed log of transitions. Arrays, not an environment.

    Deliberately exposes no `step`. If an algorithm needs to interact, it is not
    an offline algorithm, and the type system should say so.
    """

    def __init__(self, states, actions, rewards, next_states, terminated, n_states, n_actions):
        self.states = np.asarray(states, dtype=np.int64)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.rewards = np.asarray(rewards, dtype=np.float64)
        self.next_states = np.asarray(next_states, dtype=np.int64)
        self.terminated = np.asarray(terminated, dtype=bool)
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if not (
            len(self.states)
            == len(self.actions)
            == len(self.rewards)
            == len(self.next_states)
            == len(self.terminated)
        ):
            raise ValueError("dataset arrays have mismatched lengths")

    def __len__(self):
        return len(self.states)

    def support(self):
        """Boolean (n_states, n_actions): which pairs the log actually contains.

        The single most useful thing to print before debugging an offline result.
        Everything outside this mask is a guess, and the whole field is about not
        trusting guesses.
        """
        mask = np.zeros((self.n_states, self.n_actions), dtype=bool)
        mask[self.states, self.actions] = True
        return mask

    def coverage(self) -> float:
        """Fraction of (state, action) pairs present. Report it with every result."""
        return float(self.support().mean())

    def action_counts(self):
        counts = np.zeros((self.n_states, self.n_actions))
        np.add.at(counts, (self.states, self.actions), 1)
        return counts

    def behaviour_policy(self):
        """The empirical policy that produced the log, as action probabilities.

        Rows with no data are uniform. This is what behaviour cloning fits, and
        what the conservative methods try not to stray from.
        """
        counts = self.action_counts()
        totals = counts.sum(axis=1, keepdims=True)
        uniform = np.full_like(counts, 1.0 / self.n_actions)
        return np.where(totals > 0, counts / np.maximum(totals, 1), uniform)

    def summary(self) -> str:
        return (
            f"{len(self):,} transitions   coverage {self.coverage():.3f}   "
            f"mean reward {self.rewards.mean():+.3f}   "
            f"terminated {self.terminated.sum():,}"
        )


def behaviour_policy_for(env, quality: str, gamma: float, seed: int = 0):
    """The stochastic policy that generates a dataset of the given quality.

    Returns an (n_states, n_actions) array of action probabilities.
    """
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}, got {quality!r}")

    n_states, n_actions = env.n_states, env.n_actions
    uniform = np.full((n_states, n_actions), 1.0 / n_actions)
    if quality == "random":
        return uniform

    optimal = np.zeros((n_states, n_actions))
    optimal[np.arange(n_states), env.optimal_policy(gamma)] = 1.0
    if quality == "expert":
        return 0.95 * optimal + 0.05 * uniform  # a little noise, or coverage is nil

    if quality == "medium":
        return 0.4 * optimal + 0.6 * uniform

    # mixed: mostly random, with a thin seam of expert running through it. The
    # hard case, and the realistic one -- logs are rarely uniform in quality.
    return 0.15 * optimal + 0.85 * uniform


def collect(env, policy, n_transitions: int, seed: int = 0) -> Dataset:
    """Roll out a stochastic policy and log every transition."""
    policy = np.asarray(policy, dtype=np.float64)
    rng = np.random.default_rng(seed)
    states, actions, rewards, next_states, terminated = [], [], [], [], []

    episode = 0
    while len(states) < n_transitions:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        episode += 1
        for _ in range(env.max_steps):
            state = env.state
            action = int(rng.choice(env.n_actions, p=policy[state]))
            _, reward, term, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(env.state)
            terminated.append(term)

            if term or trunc or len(states) >= n_transitions:
                break

    return Dataset(
        states, actions, rewards, next_states, terminated, env.n_states, env.n_actions
    )


def build(env, quality: str, n_transitions: int, gamma: float, seed: int = 0) -> Dataset:
    """Convenience: behaviour policy of the given quality, then collect from it."""
    policy = behaviour_policy_for(env, quality, gamma, seed)
    return collect(env, policy, n_transitions, seed)


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Every method scores about the same: check `coverage()`. Below a percent or so
#   there is nothing to distinguish, and above about a third the problem is close
#   to online RL. The interesting regime is narrow and it is worth reporting the
#   number rather than the dataset name.
# - The expert dataset has near-zero coverage: correct, and the reason
#   `behaviour_policy_for` mixes in 5% uniform. A deterministic expert visits one
#   trajectory and nothing else is in the log at all.
# - An offline result improves when you rerun it: something is touching the
#   environment. `Dataset` has no `step` on purpose -- if an algorithm needs one,
#   it is not offline and the comparison is meaningless.
