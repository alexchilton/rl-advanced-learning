"""GridWorld: the workhorse environment.

Small enough to solve exactly, configurable enough to reproduce most of the ways
reinforcement learning goes wrong.

The reward modes are the reason this file exists:

    sparse        +1 for entering the goal. Nothing else. Honest and hard.
    dense         a per-step progress signal. Easy, and it changes the problem.
    shaped        potential-based shaping (Ng et al. 1999). Provably leaves the
                  optimal policy unchanged -- the tests assert exactly that.
    misspecified  a one-sided progress bonus. Looks reasonable. The optimal
                  policy is to oscillate next to the goal forever and never
                  enter it. This is reward hacking, in four lines.

The difference between `shaped` and `misspecified` is one `max(0, ...)`. That is
the entire lesson.
"""

from __future__ import annotations

import numpy as np

from envs.tabular import TabularEnv

# (row, column) deltas. Row 0 is the top.
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_ARROWS = ("^", ">", "v", "<")

REWARD_MODES = ("sparse", "dense", "shaped", "misspecified")

STEP_PENALTY = 0.01  # dense mode only
PROGRESS_SCALE = 0.1  # dense mode
POTENTIAL_SCALE = 0.1  # shaped mode
# misspecified mode. Larger than the others so that the hack wins at ordinary
# discounts. It has to beat reaching the goal, and that contest is decided by
# HACK_BONUS / (1 - gamma^2) against roughly 1 + HACK_BONUS -- see the note at
# the bottom of this file. At 0.1 the hack only wins above gamma ~= 0.96, which
# is a real and unpleasant property of misspecified rewards rather than a
# property of this grid.
HACK_BONUS = 0.3


class GridWorld(TabularEnv):
    """An n x n grid. Reach the goal.

    Args:
        size: grid is size x size.
        start: (row, col) of the start cell.
        goal: (row, col) of the goal cell. Defaults to the bottom-right corner.
        walls: iterable of (row, col) cells that cannot be entered.
        slip: probability of moving perpendicular to the intended direction,
            split evenly between the two perpendiculars. 0 gives a deterministic
            grid; 0.2 is enough to make greedy policies visibly risk-averse.
        reward_mode: one of REWARD_MODES.
        gamma: only used by `shaped`, which needs the discount to build a
            policy-invariant potential. Pass the same gamma the agent uses --
            a mismatch quietly breaks the invariance guarantee.
        max_steps: truncation limit. Reaching it sets `truncated`, never
            `terminated`.
        obs_mode: "index", "onehot", or "xy".
    """

    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        walls=(),
        slip: float = 0.0,
        reward_mode: str = "sparse",
        gamma: float = 0.99,
        max_steps: int = 100,
        obs_mode: str = "index",
    ):
        if size < 2:
            raise ValueError(f"size must be at least 2, got {size}")
        if not 0.0 <= slip <= 1.0:
            raise ValueError(f"slip must be in [0, 1], got {slip}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")

        self.size = int(size)
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (self.size - 1, self.size - 1)
        self.walls = frozenset(tuple(w) for w in walls)
        self.slip = float(slip)
        self.reward_mode = reward_mode
        self.gamma = float(gamma)

        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not self._in_bounds(cell):
                raise ValueError(f"{name} {cell} is outside a {size}x{size} grid")
            if cell in self.walls:
                raise ValueError(f"{name} {cell} is a wall")
        if self.start == self.goal:
            raise ValueError("start and goal are the same cell")

        n_states = self.size * self.size
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
    # Geometry
    # ------------------------------------------------------------------
    def index(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.size + cell[1]

    def cell(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.size)

    def _in_bounds(self, cell) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def _move(self, cell, delta):
        """Where you end up. Walls and edges leave you where you were."""
        target = (cell[0] + delta[0], cell[1] + delta[1])
        if not self._in_bounds(target) or target in self.walls:
            return cell
        return target

    def distance(self, cell) -> int:
        """Manhattan distance to the goal. Ignores walls, which is the point --
        a shaping signal built from an approximate distance is the realistic case."""
        return abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _outcome_distribution(self, action: int):
        """(delta, probability) pairs after slip."""
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

        for s in range(n_states):
            cell = self.cell(s)
            for a in range(4):
                # The goal is absorbing and terminal, so its own value is 0.
                # Walls are unreachable; they self-loop only so that every state
                # in the index space has a well-formed row in P.
                if s == goal_index:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if cell in self.walls:
                    P[s][a] = [(1.0, s, 0.0, False)]
                    continue

                merged: dict[int, float] = {}
                for delta, prob in self._outcome_distribution(a):
                    next_cell = self._move(cell, delta)
                    next_state = self.index(next_cell)
                    merged[next_state] = merged.get(next_state, 0.0) + prob

                for next_state, prob in merged.items():
                    next_cell = self.cell(next_state)
                    reward = self._reward(cell, next_cell)
                    terminated = next_cell == self.goal
                    P[s][a].append((prob, next_state, reward, terminated))
        return P

    def _reward(self, cell, next_cell) -> float:
        reached_goal = next_cell == self.goal
        base = 1.0 if reached_goal else 0.0
        d, d_next = self.distance(cell), self.distance(next_cell)

        if self.reward_mode == "sparse":
            return base

        if self.reward_mode == "dense":
            # Two-sided progress plus a cost of living. Learns fast. Note that it
            # is NOT policy-invariant: the step penalty makes a shorter path
            # strictly better, which happens to agree with the sparse optimum
            # here, but would not on a grid where the goal is optional.
            return base - STEP_PENALTY + PROGRESS_SCALE * (d - d_next)

        if self.reward_mode == "shaped":
            # Potential-based shaping: F = gamma * Phi(s') - Phi(s).
            # Phi(terminal) MUST be 0 or the invariance guarantee is void. That
            # single line is the most commonly omitted detail in the paper.
            potential = -POTENTIAL_SCALE * d
            potential_next = 0.0 if reached_goal else -POTENTIAL_SCALE * d_next
            return base + self.gamma * potential_next - potential

        if self.reward_mode == "misspecified":
            # One-sided: reward progress, do not penalise regress. Now stepping
            # towards the goal and back again nets +HACK_BONUS every two steps,
            # forever, and entering the goal ends the gravy train.
            return base + HACK_BONUS * max(0.0, d - d_next)

        raise AssertionError(f"unhandled reward_mode {self.reward_mode!r}")

    # ------------------------------------------------------------------
    def observation(self, state: int):
        if self.obs_mode == "xy":
            row, col = self.cell(state)
            scale = max(self.size - 1, 1)
            return np.array([row / scale, col / scale], dtype=np.float32)
        return super().observation(state)

    # ------------------------------------------------------------------
    # Rendering. Cheap to write, and it has caught more bugs than any plot.
    # ------------------------------------------------------------------
    def render(self) -> str:
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) in self.walls:
                    row.append("#")
                elif (r, c) == self.goal:
                    row.append("G")
                elif self.state is not None and (r, c) == self.cell(self.state):
                    row.append("A")
                elif (r, c) == self.start:
                    row.append("s")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def render_policy(self, policy) -> str:
        """Arrows per cell. The fastest way to see that a policy is nonsense."""
        policy = np.asarray(policy)
        if policy.ndim == 2:
            policy = policy.argmax(axis=1)
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) in self.walls:
                    row.append("#")
                elif (r, c) == self.goal:
                    row.append("G")
                else:
                    row.append(ACTION_ARROWS[int(policy[self.index((r, c))])])
            rows.append(" ".join(row))
        return "\n".join(rows)

    def render_values(self, values, width: int = 7, decimals: int = 2) -> str:
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            values = values.max(axis=1)
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) in self.walls:
                    row.append("#".rjust(width))
                else:
                    row.append(f"{values[self.index((r, c))]:{width}.{decimals}f}")
            rows.append(" ".join(row))
        return "\n".join(rows)


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Agent learns fast on `dense` and not at all on `sparse`: that is not a bug.
#   With a 5x5 grid and epsilon-greedy, a random walk reaches the goal often
#   enough; at size 12 it does not. If sparse never learns, the problem is
#   exploration, not the update rule. See diagnostics/ and Tier 5 (HER, RND).
# - Learned Q looks plausible but the policy is wrong: compare against
#   `true_q(gamma)` cell by cell, not by eyeballing the return curve.
# - `shaped` gives a different optimal policy from `sparse`: the gamma passed to
#   the env does not match the gamma the agent uses, or Phi(terminal) is not 0.
# - `misspecified` agent gets high reward and never reaches the goal: working as
#   intended. That is the demo.
#
# The misspecified reward is worth doing the arithmetic on, because it is the
# part people assume is obvious.
#
# Oscillating between a cell at distance 2 and one at distance 1 collects
# HACK_BONUS every second step, forever:
#
#     V_hack = HACK_BONUS / (1 - gamma^2)
#
# Entering the goal from distance 1 collects the bonus once plus the goal reward,
# and then the episode is over:
#
#     V_honest = 1 + HACK_BONUS
#
# So the hack wins exactly when HACK_BONUS / (1 - gamma^2) > 1 + HACK_BONUS. At
# HACK_BONUS = 0.1 that needs gamma > 0.96; at 0.3 it needs gamma > 0.87.
#
# Which is the uncomfortable part. The same misspecified reward is harmless at
# one discount and catastrophic at another, so a reward bug can sit dormant in a
# codebase and surface the day someone raises gamma to solve a longer task. The
# bug did not change. The horizon did.
#
# `tests/test_gridworld.py::test_the_hack_depends_on_the_discount` pins both sides.
