"""Racetrack (Sutton & Barto, exercise 5.12).

This environment exists for one reason: it is the smallest thing here on which
behaviour cloning fails for the RIGHT reason.

Measured on the other environments, BC does not fail properly. On a deterministic
GridWorld with a fixed start it reaches 1.00 success from a single demonstration,
because the expert visits 14 of 64 states and the clone never needs the rest. On
PointMass a linear least-squares clone fits the PD expert to a training error of
2e-17 and matches its step count exactly, because the expert IS a clipped linear
function. Neither shows compounding error; one shows missing data and the other
shows nothing at all.

Compounding error needs three things at once, and Racetrack is the smallest
environment with all three.

1. **The good region is narrow.** The track is four cells wide through the
   corners. A single wrong acceleration puts the car into a wall.
2. **Velocity is part of the state, so errors compound in TIME, not just space.**
   A small mistake at speed 4 costs four cells, and the next mistake is made from
   a state further from anything the expert demonstrated.
3. **The expert never demonstrates recovery.** The optimal policy races the line
   and brakes for exactly one corner. It is never seen approaching a wall too
   fast, so the demonstrations contain no example of what to do about it -- which
   is precisely the gap DAgger closes by querying the expert on the LEARNER's
   states rather than on its own.

Everything stays finite, so the exact-ground-truth guarantee survives: the optimal
policy, the exact value of any policy, and the exact value of a behaviour-cloned
policy all come from `envs.solvers` rather than from sampling.

State is (cell, vertical speed, horizontal speed). Actions are the nine
combinations of accelerating each velocity component by -1, 0 or +1.
"""

from __future__ import annotations

import numpy as np

from envs.tabular import TabularEnv

# Track characters. '#' is drivable, ' ' is wall, 'S' is the start line,
# 'F' is the finish line. Row 0 is the top.
#
# Velocity components are non-negative, as in Sutton & Barto, so the car can only
# travel up and to the right. A track therefore has to be a staircase. This is not
# a limitation to work around -- it is what makes the corners unavoidable, and the
# corners are where imitation breaks down.
#
# Three corners rather than one, because compounding error needs time to compound.
# The original one-corner track here was finished in seven steps and a clone from a
# single demonstration lost only 0.4 of a return unit; over eighteen steps and three
# corners the same clone has room to drift somewhere the expert never went.
DEFAULT_TRACK = (
    "          ########F",
    "          ########F",
    "          ########F",
    "     ############# ",
    "     ############# ",
    "     ############# ",
    "     #####         ",
    "     #####         ",
    "     #####         ",
    "  ########         ",
    "  ########         ",
    "  ########         ",
    "  ####             ",
    "  ####             ",
    "  SSSS             ",
)

# The textbook single-corner track, kept for reproducing chapter 5.
SIMPLE_TRACK = (
    "  ##########F",
    "  ##########F",
    "  ##########F",
    "  ####       ",
    "  ####       ",
    "  ####       ",
    "  ####       ",
    "  ####       ",
    "  ####       ",
    "  SSSS       ",
)

MAX_SPEED = 4
STEP_REWARD = -1.0

# (delta vertical speed, delta horizontal speed). Vertical speed is upward.
ACCELERATIONS = tuple((dv, dh) for dv in (-1, 0, 1) for dh in (-1, 0, 1))
ACTION_NAMES = tuple(f"dv={dv:+d},dh={dh:+d}" for dv, dh in ACCELERATIONS)


class Racetrack(TabularEnv):
    """Drive from the start line to the finish line without hitting a wall.

    Args:
        track: rows of the track as strings. Defaults to a corridor with one
            corner, which is enough to separate a racing line from a safe line.
        max_speed: velocity components are integers in [0, max_speed]. Larger
            values make the corner harder and the state space bigger.
        noise: probability that the acceleration is ignored and the velocity
            carries over unchanged (Sutton & Barto use 0.1). This is what stops a
            purely memorised action sequence from working, and it spreads the
            expert's demonstrations slightly, which is the realistic case.
        crash_penalty: extra reward on hitting a wall, on top of the step cost.
            0 matches the textbook, where crashing is punished only by the time
            it wastes.
        on_crash: what a wall does. "stop" leaves the car at the last drivable
            cell on its path with zero velocity. "restart" returns it to a random
            start cell, which is what Sutton & Barto specify.
        max_steps: truncation limit.
        obs_mode: "index", "onehot", or "features" (row, col, vertical speed,
            horizontal speed, each scaled to roughly [0, 1]).

    Neither crash mode ends the episode -- same convention as the cliff, and for
    the same reason: recovery has to be part of the problem or there is nothing
    for DAgger to teach.

    The default is "stop" rather than the textbook "restart", and the reason is
    measured. Under "restart", behaviour cloning from a SINGLE demonstration
    reaches a 1.00 success rate, because a crash teleports the learner back to
    the start line -- which is on the expert's own state distribution. The
    environment hands the clone a free correction every time it errs, so the
    covariate shift this environment exists to demonstrate never accumulates.

    Under "stop" the car is stranded mid-track at zero velocity, in a state no
    demonstration contains, and has to drive out of it. That is the situation
    DAgger was invented for. "restart" is kept for textbook fidelity and for
    reproducing the Monte Carlo control results in chapter 5.
    """

    def __init__(
        self,
        track=DEFAULT_TRACK,
        max_speed: int = MAX_SPEED,
        noise: float = 0.1,
        crash_penalty: float = 0.0,
        on_crash: str = "stop",
        max_steps: int = 200,
        obs_mode: str = "index",
    ):
        if max_speed < 1:
            raise ValueError(f"max_speed must be at least 1, got {max_speed}")
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {noise}")
        if on_crash not in ("stop", "restart"):
            raise ValueError(f"on_crash must be 'stop' or 'restart', got {on_crash!r}")
        self.on_crash = on_crash

        self.track = tuple(track)
        self.rows = len(self.track)
        self.cols = max(len(row) for row in self.track)
        self.max_speed = int(max_speed)
        self.noise = float(noise)
        self.crash_penalty = float(crash_penalty)
        self.speeds = self.max_speed + 1

        self.drivable, self.start_cells, self.finish_cells = self._parse_track()
        if not self.start_cells:
            raise ValueError("track has no start line ('S')")
        if not self.finish_cells:
            raise ValueError("track has no finish line ('F')")

        # One extra absorbing state for "finished", so that every transition has
        # a well-defined next state even though the episode ends there.
        self.n_cells = self.rows * self.cols
        self.finished_state = self.n_cells * self.speeds * self.speeds
        n_states = self.finished_state + 1

        P = self._build_transitions(n_states)
        start_distribution = np.zeros(n_states)
        for cell in self.start_cells:
            start_distribution[self.state_index(cell, 0, 0)] = 1.0 / len(self.start_cells)

        super().__init__(
            n_states=n_states,
            n_actions=len(ACCELERATIONS),
            P=P,
            start_distribution=start_distribution,
            max_steps=max_steps,
            obs_mode=obs_mode,
        )

    # ------------------------------------------------------------------
    # Track geometry
    # ------------------------------------------------------------------
    def _parse_track(self):
        drivable, start_cells, finish_cells = set(), [], set()
        for r, row in enumerate(self.track):
            for c, char in enumerate(row):
                cell = (r, c)
                if char == "#":
                    drivable.add(cell)
                elif char == "S":
                    drivable.add(cell)
                    start_cells.append(cell)
                elif char == "F":
                    drivable.add(cell)
                    finish_cells.add(cell)
                elif char != " ":
                    raise ValueError(f"unknown track character {char!r} at row {r}, col {c}")
        return frozenset(drivable), start_cells, frozenset(finish_cells)

    def cell_index(self, cell) -> int:
        return cell[0] * self.cols + cell[1]

    def state_index(self, cell, vertical: int, horizontal: int) -> int:
        return (self.cell_index(cell) * self.speeds + vertical) * self.speeds + horizontal

    def decode(self, state: int):
        """(cell, vertical speed, horizontal speed). Raises on the finished state."""
        state = int(state)
        if state == self.finished_state:
            raise ValueError("the finished state has no position or velocity")
        horizontal = state % self.speeds
        state //= self.speeds
        vertical = state % self.speeds
        cell_index = state // self.speeds
        return divmod(cell_index, self.cols), vertical, horizontal

    def _path(self, cell, vertical: int, horizontal: int):
        """Cells swept while moving by (-vertical, +horizontal), start excluded.

        Checking only the destination would let the car jump a wall at speed 4,
        which turns the corner into a shortcut and quietly makes the task easier.
        """
        steps = max(vertical, horizontal)
        if steps == 0:
            return []
        swept, previous = [], cell
        for step in range(1, steps + 1):
            fraction = step / steps
            point = (
                cell[0] - int(round(vertical * fraction)),
                cell[1] + int(round(horizontal * fraction)),
            )
            if point != previous:
                swept.append(point)
                previous = point
        return swept

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _resolve(self, cell, vertical, horizontal, action):
        """(next_state, reward, terminated) for one deterministic outcome."""
        delta_vertical, delta_horizontal = ACCELERATIONS[action]
        vertical = int(np.clip(vertical + delta_vertical, 0, self.max_speed))
        horizontal = int(np.clip(horizontal + delta_horizontal, 0, self.max_speed))

        for point in self._path(cell, vertical, horizontal):
            if point in self.finish_cells:
                return self.finished_state, STEP_REWARD, True
            if point not in self.drivable:
                # Crash. `cell` is still the last drivable cell on the path.
                reward = STEP_REWARD + self.crash_penalty
                if self.on_crash == "restart":
                    return None, reward, False
                return self.state_index(cell, 0, 0), reward, False
            cell = point

        return self.state_index(cell, vertical, horizontal), STEP_REWARD, False

    def _build_transitions(self, n_states):
        P = {s: {a: [] for a in range(len(ACCELERATIONS))} for s in range(n_states)}
        crash_outcomes = [
            (1.0 / len(self.start_cells), self.state_index(cell, 0, 0))
            for cell in self.start_cells
        ]

        for state in range(n_states):
            if state == self.finished_state:
                for action in range(len(ACCELERATIONS)):
                    P[state][action] = [(1.0, state, 0.0, True)]
                continue

            cell, vertical, horizontal = self.decode(state)
            if cell not in self.drivable:
                # Unreachable. Defined so every row of P is well formed.
                for action in range(len(ACCELERATIONS)):
                    P[state][action] = [(1.0, state, 0.0, False)]
                continue

            for action in range(len(ACCELERATIONS)):
                merged: dict[tuple[int, float, bool], float] = {}
                # With probability `noise` the acceleration is ignored. Index 4 is
                # (0, 0), the do-nothing acceleration.
                outcomes = [(action, 1.0 - self.noise), (4, self.noise)]
                for effective, probability in outcomes:
                    if probability == 0.0:
                        continue
                    next_state, reward, terminated = self._resolve(
                        cell, vertical, horizontal, effective
                    )
                    if next_state is None:  # crash, scattered over the start line
                        for crash_probability, crash_state in crash_outcomes:
                            key = (crash_state, reward, False)
                            merged[key] = merged.get(key, 0.0) + probability * crash_probability
                    else:
                        key = (next_state, reward, terminated)
                        merged[key] = merged.get(key, 0.0) + probability

                P[state][action] = [
                    (probability, next_state, reward, terminated)
                    for (next_state, reward, terminated), probability in merged.items()
                ]
        return P

    # ------------------------------------------------------------------
    def observation(self, state: int):
        if self.obs_mode == "features":
            if state == self.finished_state:
                return np.zeros(4, dtype=np.float32)
            (row, col), vertical, horizontal = self.decode(state)
            return np.array(
                [
                    row / max(self.rows - 1, 1),
                    col / max(self.cols - 1, 1),
                    vertical / self.max_speed,
                    horizontal / self.max_speed,
                ],
                dtype=np.float32,
            )
        return super().observation(state)

    # ------------------------------------------------------------------
    # Rendering. Note these take a STATE, not a cell -- the position alone does
    # not determine what the car should do, which is the whole difficulty.
    # ------------------------------------------------------------------
    def render(self) -> str:
        if self.state is None or self.state == self.finished_state:
            car, vertical, horizontal = None, 0, 0
        else:
            car, vertical, horizontal = self.decode(self.state)

        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == car:
                    row.append("A")
                elif cell in self.finish_cells:
                    row.append("F")
                elif cell in self.start_cells:
                    row.append("S")
                elif cell in self.drivable:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        lines.append(f"speed: up {vertical}, right {horizontal}")
        return "\n".join(lines)

    def render_trajectory(self, states) -> str:
        """Draw the cells a trajectory passed through, marked by speed."""
        speed_at: dict[tuple[int, int], int] = {}
        for state in states:
            if int(state) == self.finished_state:
                continue
            cell, vertical, horizontal = self.decode(state)
            speed_at[cell] = max(vertical, horizontal)

        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell in speed_at:
                    row.append(str(speed_at[cell]))
                elif cell in self.finish_cells:
                    row.append("F")
                elif cell in self.drivable:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)

    def cell_values(self, values):
        """Collapse a per-state quantity to a per-cell grid by taking the best
        velocity in each cell. For eyeballing a value function on the track."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            values = values.max(axis=1)
        grid = np.full((self.rows, self.cols), np.nan)
        for cell in self.drivable:
            best = max(
                values[self.state_index(cell, v, h)]
                for v in range(self.speeds)
                for h in range(self.speeds)
            )
            grid[cell] = best
        return grid

    def demonstration_coverage(self, states) -> float:
        """Fraction of reachable states a set of demonstrations actually visited.

        Print this before concluding anything about behaviour cloning. If the
        expert covered 3% of the state space, the clone's failures off that 3%
        are not a modelling problem.
        """
        reachable = sum(
            1
            for cell in self.drivable
            for _ in range(self.speeds * self.speeds)
            if cell not in self.finish_cells
        )
        return len({int(s) for s in states}) / max(reachable, 1)


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The optimal policy takes the corner at full speed and never crashes: check
#   `noise`. At 0 the track is a deterministic puzzle and the racing line is
#   exact. At 0.1 the optimal policy has to leave margin, which is what makes the
#   expert's behaviour worth imitating rather than memorising.
# - Value iteration will not converge at gamma=1: with `noise` there are states
#   from which the finish is not reachable in bounded time. Use gamma < 1.
# - A cloned policy crashes immediately: expected, and the point. Report
#   `demonstration_coverage` alongside, so the reader can see whether the failure
#   is compounding error or simply no data. The two need different fixes and look
#   identical in the return.
# - The car appears to jump a wall: `_path` is checking only the destination
#   cell. At speed 4 the car moves four cells per step and the corner becomes a
#   shortcut.
