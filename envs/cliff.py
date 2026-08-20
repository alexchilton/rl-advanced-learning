"""Cliff Walking (Sutton & Barto, Example 6.6).

The environment that separates on-policy from off-policy learning in one picture.

A 4x12 grid. Start bottom-left, goal bottom-right, and the ten cells between them
are a cliff: step in and you take -100 and get dragged back to the start. Every
other step costs -1.

Run SARSA and Q-learning here with the same epsilon and they learn different
policies. Q-learning learns the optimal path, hugging the cliff edge, because its
target assumes the greedy action will be taken next. SARSA learns the safe path
one row up, because its target uses the action its own epsilon-greedy policy will
actually take -- and that policy sometimes steps off the cliff.

Neither is wrong. They are evaluating different policies. Q-learning finds the
better policy and gets a worse return during training; SARSA finds a worse policy
and gets a better return during training. As epsilon goes to zero they converge.

`experiments/cliff_sarsa_vs_qlearning.py` reproduces the figure.
"""

from __future__ import annotations

import numpy as np

from envs.tabular import TabularEnv

# (row, col) deltas. Row 0 is the top; the cliff is on the bottom row.
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_ARROWS = ("^", ">", "v", "<")

STEP_REWARD = -1.0
CLIFF_REWARD = -100.0


class CliffWalking(TabularEnv):
    """The standard 4x12 cliff.

    Args:
        rows, cols: grid shape. The defaults are the textbook figure.
        slip: probability of moving perpendicular to the intended direction.
            0 is the textbook version. Raising it makes SARSA's caution pay off
            more visibly, because exploration is no longer the only way to fall.
        max_steps: truncation limit.
        obs_mode: "index", "onehot", or "xy".

    Note that falling off the cliff does NOT terminate the episode -- it returns
    the agent to the start. Only reaching the goal terminates. Implementations
    that treat the fall as terminal learn a different, easier problem, and it is
    a common enough mistake to be worth stating.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        slip: float = 0.0,
        max_steps: int = 200,
        obs_mode: str = "index",
    ):
        if rows < 2 or cols < 3:
            raise ValueError(f"need at least a 2x3 grid, got {rows}x{cols}")
        if not 0.0 <= slip <= 1.0:
            raise ValueError(f"slip must be in [0, 1], got {slip}")

        self.rows = int(rows)
        self.cols = int(cols)
        self.slip = float(slip)
        self.start = (self.rows - 1, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.cliff = frozenset((self.rows - 1, c) for c in range(1, self.cols - 1))

        n_states = self.rows * self.cols
        P = self._build_transitions(n_states)
        start_distribution = np.zeros(n_states)
        start_distribution[self.index(self.start)] = 1.0

        super().__init__(
            n_states=n_states,
            n_actions=4,
            P=P,
            start_distribution=start_distribution,
            max_steps=max_steps,
            obs_mode=obs_mode,
        )

    # ------------------------------------------------------------------
    def index(self, cell) -> int:
        return cell[0] * self.cols + cell[1]

    def cell(self, state: int):
        return divmod(int(state), self.cols)

    def _in_bounds(self, cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _move(self, cell, delta):
        target = (cell[0] + delta[0], cell[1] + delta[1])
        return target if self._in_bounds(target) else cell

    def _outcome_distribution(self, action: int):
        if self.slip == 0.0:
            return [(ACTION_DELTAS[action], 1.0)]
        perpendicular = [(action + 1) % 4, (action - 1) % 4]
        return [
            (ACTION_DELTAS[action], 1.0 - self.slip),
            (ACTION_DELTAS[perpendicular[0]], self.slip / 2.0),
            (ACTION_DELTAS[perpendicular[1]], self.slip / 2.0),
        ]

    def _build_transitions(self, n_states):
        P = {s: {a: [] for a in range(4)} for s in range(n_states)}
        goal_index = self.index(self.goal)
        start_index = self.index(self.start)

        for s in range(n_states):
            cell = self.cell(s)
            for a in range(4):
                if s == goal_index:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if cell in self.cliff:
                    # Unreachable as a resting state -- the fall is resolved
                    # inside the transition below. Defined for completeness.
                    P[s][a] = [(1.0, start_index, 0.0, False)]
                    continue

                merged: dict[tuple[int, float, bool], float] = {}
                for delta, prob in self._outcome_distribution(a):
                    landed = self._move(cell, delta)
                    if landed in self.cliff:
                        key = (start_index, CLIFF_REWARD, False)
                    else:
                        key = (self.index(landed), STEP_REWARD, landed == self.goal)
                    merged[key] = merged.get(key, 0.0) + prob

                P[s][a] = [
                    (prob, next_state, reward, terminated)
                    for (next_state, reward, terminated), prob in merged.items()
                ]
        return P

    # ------------------------------------------------------------------
    def observation(self, state: int):
        if self.obs_mode == "xy":
            row, col = self.cell(state)
            return np.array(
                [row / max(self.rows - 1, 1), col / max(self.cols - 1, 1)], dtype=np.float32
            )
        return super().observation(state)

    def render(self) -> str:
        rows = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if self.state is not None and (r, c) == self.cell(self.state):
                    row.append("A")
                elif (r, c) in self.cliff:
                    row.append("C")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) == self.start:
                    row.append("S")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def render_policy(self, policy) -> str:
        policy = np.asarray(policy)
        if policy.ndim == 2:
            policy = policy.argmax(axis=1)
        rows = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.cliff:
                    row.append("C")
                elif (r, c) == self.goal:
                    row.append("G")
                else:
                    row.append(ACTION_ARROWS[int(policy[self.index((r, c))])])
            rows.append(" ".join(row))
        return "\n".join(rows)


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - SARSA and Q-learning learn the same policy: epsilon is 0, or SARSA is using
#   max(Q[s']) somewhere instead of Q[s', a'] for the action it actually took.
#   That is the single most common SARSA bug and it turns SARSA into Q-learning
#   with no error and a plausible-looking curve.
# - Q-learning's training return looks worse than SARSA's: correct, that is the
#   result. Evaluate the GREEDY policy separately (epsilon=0) and Q-learning wins.
# - Returns around -13 for the optimal path on the default 4x12 grid; SARSA's
#   safe path is around -17. If you see -100s persisting late in training with a
#   decayed epsilon, exploration is not decaying.
