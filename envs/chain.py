"""Chain MDPs: exploration, discounting, and the deadly triad.

Two environments, both deliberately degenerate, both making a point that a grid
cannot make as cleanly.

`Chain` is a line of n states with the only reward at the far end, and a "left"
action that sends you all the way back to the start. To reach the reward once, a
uniformly random policy must pick "right" n-1 times in a row: probability 2^-(n-1).
At n=20 that is one episode in half a million. Epsilon-greedy does not fix this,
because epsilon-greedy IS the uniform random policy until something is learned.
This is the cleanest argument that exploration is a design problem rather than a
hyperparameter, and it is the environment where count-based exploration and RND
earn their keep.

`Baird` is Baird's counterexample (1995). Off-policy updates, plus bootstrapping,
plus linear function approximation: the deadly triad. All rewards are zero, so the
true value function is zero and is exactly representable by the features. Off-policy
TD still diverges to infinity. Remove any one of the three and it does not.
`experiments/deadly_triad.py` runs all four combinations.
"""

from __future__ import annotations

import numpy as np

from envs.tabular import TabularEnv

LEFT, RIGHT = 0, 1
CHAIN_ACTION_NAMES = ("left", "right")


class Chain(TabularEnv):
    """n states in a line. Reward only at the far end.

    Args:
        n: number of states. Difficulty is exponential in n. n=5 is easy, n=12 is
            hard for epsilon-greedy, n=20 is hopeless without directed exploration.
        small_reward: reward for taking "left" from the start state. The classic
            NChain has a small consolation prize that a greedy agent latches onto
            and never leaves. Set to 0 for pure sparse.
        goal_reward: reward for entering the final state.
        slip: probability the intended action is reversed. The original NChain
            uses 0.2, which stops a purely deterministic plan from working.
        max_steps: truncation limit.
        obs_mode: "index" or "onehot".
    """

    def __init__(
        self,
        n: int = 10,
        small_reward: float = 0.0,
        goal_reward: float = 1.0,
        slip: float = 0.0,
        max_steps: int = 100,
        obs_mode: str = "index",
    ):
        if n < 3:
            raise ValueError(f"n must be at least 3, got {n}")
        if not 0.0 <= slip <= 1.0:
            raise ValueError(f"slip must be in [0, 1], got {slip}")

        self.n = int(n)
        self.small_reward = float(small_reward)
        self.goal_reward = float(goal_reward)
        self.slip = float(slip)

        P = self._build_transitions()
        start_distribution = np.zeros(self.n)
        start_distribution[0] = 1.0

        super().__init__(
            n_states=self.n,
            n_actions=2,
            P=P,
            start_distribution=start_distribution,
            max_steps=max_steps,
            obs_mode=obs_mode,
        )

    def _resolve(self, state: int, action: int):
        """(next_state, reward, terminated) for an action that is not slipping."""
        if action == RIGHT:
            next_state = state + 1
            if next_state == self.n - 1:
                return next_state, self.goal_reward, True
            return next_state, 0.0, False
        # LEFT always returns to the start. That is what makes the chain hard:
        # one wrong step undoes every right one.
        return 0, self.small_reward, False

    def _build_transitions(self):
        P = {s: {a: [] for a in range(2)} for s in range(self.n)}
        for s in range(self.n):
            for a in range(2):
                if s == self.n - 1:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue

                merged: dict[tuple[int, float, bool], float] = {}
                outcomes = [(a, 1.0 - self.slip), (1 - a, self.slip)] if self.slip > 0 else [(a, 1.0)]
                for effective_action, prob in outcomes:
                    if prob == 0.0:
                        continue
                    key = self._resolve(s, effective_action)
                    merged[key] = merged.get(key, 0.0) + prob

                P[s][a] = [
                    (prob, next_state, reward, terminated)
                    for (next_state, reward, terminated), prob in merged.items()
                ]
        return P

    def random_walk_success_probability(self) -> float:
        """Chance a uniformly random policy reaches the goal in one episode.

        Print this before blaming your update rule. If it is 1e-6, no amount of
        learning-rate tuning will help.
        """
        # Under a uniform random policy the slip is symmetric -- it turns as many
        # lefts into rights as the reverse -- so the chance of moving right is
        # 0.5 regardless of slip. Reaching the goal needs n-1 of them in a row.
        return float(0.5 ** (self.n - 1))

    def render(self) -> str:
        cells = []
        for s in range(self.n):
            if self.state is not None and s == self.state:
                cells.append("A")
            elif s == self.n - 1:
                cells.append("G")
            elif s == 0:
                cells.append("S")
            else:
                cells.append(".")
        return " ".join(cells)


# ----------------------------------------------------------------------
# Baird's counterexample
# ----------------------------------------------------------------------
BAIRD_N_STATES = 7
BAIRD_N_FEATURES = 8
BAIRD_DASHED, BAIRD_SOLID = 0, 1


def _baird_features():
    """States 0..5: 2*e_s + e_7. State 6: e_6 + 2*e_7. (Sutton & Barto, fig 11.1)"""
    features = np.zeros((BAIRD_N_STATES, BAIRD_N_FEATURES))
    for s in range(6):
        features[s, s] = 2.0
        features[s, 7] = 1.0
    features[6, 6] = 1.0
    features[6, 7] = 2.0
    return features


class Baird(TabularEnv):
    """Baird's counterexample: the deadly triad, minimal form.

    Seven states, two actions, every reward zero.

    - "dashed" goes to one of states 0..5 uniformly.
    - "solid" goes to state 6.

    The behaviour policy takes "dashed" with probability 6/7. The target policy
    always takes "solid". The true value function is zero everywhere and IS
    exactly representable by the linear features below -- so approximation error
    is not the problem. Off-policy semi-gradient TD diverges anyway, because the
    updates are not following the gradient of any objective, and the off-policy
    state distribution reweights them into instability.

    Use `features` as the linear feature matrix, shape (7, 8).
    """

    features = _baird_features()

    def __init__(self, max_steps: int = 1000, obs_mode: str = "index"):
        P = {
            s: {
                BAIRD_DASHED: [(1.0 / 6.0, s2, 0.0, False) for s2 in range(6)],
                BAIRD_SOLID: [(1.0, 6, 0.0, False)],
            }
            for s in range(BAIRD_N_STATES)
        }
        super().__init__(
            n_states=BAIRD_N_STATES,
            n_actions=2,
            P=P,
            start_distribution=np.full(BAIRD_N_STATES, 1.0 / BAIRD_N_STATES),
            max_steps=max_steps,
            obs_mode=obs_mode,
        )

    @staticmethod
    def behaviour_policy():
        """Takes 'solid' one seventh of the time. Shape (7, 2)."""
        pi = np.zeros((BAIRD_N_STATES, 2))
        pi[:, BAIRD_DASHED] = 6.0 / 7.0
        pi[:, BAIRD_SOLID] = 1.0 / 7.0
        return pi

    @staticmethod
    def target_policy():
        """Always 'solid'. Shape (7, 2)."""
        pi = np.zeros((BAIRD_N_STATES, 2))
        pi[:, BAIRD_SOLID] = 1.0
        return pi

    @staticmethod
    def initial_weights():
        """The textbook starting point: w = (1,1,1,1,1,1,10,1).

        Deliberately not zero. Starting at zero is already the solution, and the
        divergence would be invisible.
        """
        w = np.ones(BAIRD_N_FEATURES)
        w[6] = 10.0
        return w


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Chain: the agent learns to take "left" forever. With small_reward > 0 that is
#   correct greedy behaviour and the point of the environment. It is a value
#   problem, not a bug.
# - Chain: return stuck at exactly 0 with n large. Print
#   `random_walk_success_probability()`. If the agent has never seen the reward,
#   there is nothing to learn from and the update rule is irrelevant.
# - Baird: weights stay near zero. The behaviour and target policies are the same,
#   or the update is using on-policy transitions. The divergence needs the
#   importance ratio to be doing real work.
# - Baird: weights go to NaN rather than diverging smoothly. That is divergence,
#   just faster. Lower the step size to watch it happen.
