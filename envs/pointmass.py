"""PointMass: continuous control, deliberately badly scaled.

A point mass in 2D. Push it, reach the target, stop there. Observation is
`[x, y, vx, vy]`, action is a force `[fx, fy]` in `[-1, 1]^2`.

Two things about this environment are on purpose.

**The scaling is bad.** Position is bounded at 1. Velocity, in the units used
here, is not: measured over 30 episodes, a uniformly random policy peaks around 8
and constant full thrust reaches 24. So the four numbers the agent sees differ in
magnitude by roughly an order of magnitude, and the two that vary most are the
two that matter least at the start of training.

A network fed this raw has to learn first-layer weights ten times smaller on two
of its four inputs before it can learn anything else, and with one learning rate
for all of them it mostly does not. Wrap the observation in a running mean/std
normaliser and the same agent with the same hyperparameters learns. That one-line
difference is the demo in `experiments/obs_normalisation.py`, and it is the most
common reason a correct implementation does nothing at all.

**The action bound matters.** Actions are clipped to `[-1, 1]`. A Gaussian policy
that ignores the bound puts most of its probability mass outside it, so the
gradient it computes is for actions the environment never executed. A tanh-squashed
policy fixes that, but only with the log-probability correction -- and forgetting
that correction is `broken/sac_missing_tanh_correction.py`.

There is no exact solver here: the state space is continuous. Ground truth comes
instead from a hand-written proportional-derivative controller, `expert_action`,
which is near-optimal and doubles as the demonstrator for behaviour cloning and
DAgger in Tier 4.
"""

from __future__ import annotations

import numpy as np

# Units are chosen so that velocity is numerically far larger than position. That
# is the whole point -- do not "fix" it here. Fix it with a normaliser, in the
# agent. `test_pointmass_observation_scales_differ_by_more_than_an_order_of_magnitude`
# asserts the gap so that a future tweak cannot quietly remove the demo.
#
# A full-thrust step adds DT * FORCE_SCALE = 1.0 to the velocity, and moves the
# mass DT * v. Crossing the box under constant thrust therefore peaks at about
# sqrt(2 * 2 / DT) ~= 28 velocity units against a position that never leaves
# [-1, 1]. An earlier version used DT=0.05, where the wall arrives after seven
# steps and the peak speed is 7 -- the scale gap was in the comment but not in
# the numbers.
DT = 0.005
FORCE_SCALE = 200.0
DAMPING = 0.02
VELOCITY_LIMIT = 40.0
POSITION_LIMIT = 1.0

GOAL_RADIUS = 0.1
GOAL_SPEED = 2.0  # must also be slow enough to count as "arrived"

# Proportional-derivative gains for `expert_action`, chosen for a critically
# damped approach that settles in roughly 60 steps. Note how small the velocity
# gain has to be: velocity is ~30x position, so equal treatment would saturate
# the actuator on the velocity term alone. That asymmetry is exactly what a
# network has to discover for itself when the observation is not normalised.
EXPERT_KP = 0.9
EXPERT_KD = 0.14

REWARD_MODES = ("dense", "sparse")


class PointMass:
    """Push a point mass to the origin and keep it there.

    Args:
        reward_mode: "dense" gives -distance each step, which any policy-gradient
            method solves quickly. "sparse" gives +1 only on arrival and 0
            otherwise, which none of them solve without help -- that is the case
            HER and RND exist for.
        max_steps: truncation limit. Reaching it sets `truncated`, never
            `terminated`. On this environment the distinction is worth real
            performance: the mass is still moving when the clock runs out, so the
            value of the final state is genuinely non-zero.
        start_radius: the mass starts at a uniformly random angle and a radius
            drawn uniformly from [0.4 * start_radius, start_radius].
        reward_scale: multiplies every reward. Set it to 1000 to reproduce the
            gradient explosion in `experiments/reward_normalisation.py`. The
            optimal policy is unchanged; the learning dynamics are not.
        seed: seeds the start state sampling.
    """

    n_actions = 2
    obs_dim = 4

    def __init__(
        self,
        reward_mode: str = "dense",
        max_steps: int = 200,
        start_radius: float = 0.8,
        reward_scale: float = 1.0,
        seed: int | None = None,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        if not 0.0 < start_radius <= POSITION_LIMIT:
            raise ValueError(f"start_radius must be in (0, {POSITION_LIMIT}], got {start_radius}")

        self.reward_mode = reward_mode
        self.max_steps = int(max_steps)
        self.start_radius = float(start_radius)
        self.reward_scale = float(reward_scale)

        self._rng = np.random.default_rng(seed)
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.t = 0
        self._started = False

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Angle and radius both vary. A fixed radius would make the expert's
        # trajectory identical every episode up to rotation, which is fine for
        # control and useless as a demonstration dataset -- behaviour cloning
        # would have nothing to generalise over.
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        radius = self._rng.uniform(0.4 * self.start_radius, self.start_radius)
        self.position = radius * np.array([np.cos(angle), np.sin(angle)])
        self.velocity = np.zeros(2)
        self.t = 0
        self._started = True
        return self.observation(), {"steps": 0, "distance": self.distance()}

    def step(self, action):
        if not self._started:
            raise RuntimeError("step() before reset()")

        action = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"action contains non-finite values: {action}")
        # Clipping here rather than trusting the agent. An agent whose actions are
        # routinely clipped is computing gradients for actions that never ran --
        # `info['clipped']` reports the fraction, and a value above ~0.1 means the
        # policy's scale is wrong.
        clipped = np.clip(action, -1.0, 1.0)
        clip_fraction = float(np.mean(np.abs(action - clipped) > 1e-9))

        self.velocity += DT * (FORCE_SCALE * clipped - DAMPING * self.velocity)
        self.velocity = np.clip(self.velocity, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        self.position += DT * self.velocity

        # Walls are inelastic: hit one and the velocity into it is killed.
        for axis in range(2):
            if abs(self.position[axis]) > POSITION_LIMIT:
                self.position[axis] = np.clip(self.position[axis], -POSITION_LIMIT, POSITION_LIMIT)
                self.velocity[axis] = 0.0

        self.t += 1
        arrived = self.arrived()
        reward = self._reward(arrived)
        truncated = bool(not arrived and self.t >= self.max_steps)
        info = {
            "steps": self.t,
            "distance": self.distance(),
            "speed": float(np.linalg.norm(self.velocity)),
            "clipped": clip_fraction,
        }
        return self.observation(), reward, bool(arrived), truncated, info

    def observation(self):
        return np.concatenate([self.position, self.velocity]).astype(np.float32)

    # ------------------------------------------------------------------
    def distance(self) -> float:
        return float(np.linalg.norm(self.position))

    def arrived(self) -> bool:
        """Close to the target AND slow. Position alone is not enough -- a mass
        flying through the target at full speed has not solved anything, and an
        agent rewarded for that learns to oscillate."""
        return self.distance() < GOAL_RADIUS and float(np.linalg.norm(self.velocity)) < GOAL_SPEED

    def _reward(self, arrived: bool) -> float:
        if self.reward_mode == "dense":
            reward = -self.distance()
            if arrived:
                reward += 1.0
        else:
            reward = 1.0 if arrived else 0.0
        return float(reward * self.reward_scale)

    # ------------------------------------------------------------------
    def expert_action(self, noise: float = 0.0):
        """A proportional-derivative controller. Near-optimal, and the
        demonstrator for behaviour cloning and DAgger.

        `noise` makes it an imperfect expert, which is what the offline dataset
        quality axis needs: expert, medium and random data from one knob.
        """
        force = -(EXPERT_KP * self.position + EXPERT_KD * self.velocity)
        if noise > 0.0:
            force = force + self._rng.normal(0.0, noise, size=2)
        return np.clip(force, -1.0, 1.0)

    def render(self) -> str:
        return (
            f"t={self.t:3d}  pos=({self.position[0]:+.3f}, {self.position[1]:+.3f})  "
            f"vel=({self.velocity[0]:+7.2f}, {self.velocity[1]:+7.2f})  "
            f"d={self.distance():.3f}{'  ARRIVED' if self.arrived() else ''}"
        )

    def observation_scales(self):
        """Rough magnitude of each observation component.

        Printing this next to your network's first-layer weights is the fastest
        way to see the scaling problem this environment exists to demonstrate.
        """
        return np.array([POSITION_LIMIT, POSITION_LIMIT, VELOCITY_LIMIT, VELOCITY_LIMIT])


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Nothing learns, no error, loss decreases slightly: check observation scaling
#   before anything else. Print `observation_scales()`. Normalise, rerun.
# - `info['clipped']` is above 0.1: the policy's action distribution is wider than
#   the action space. Squash with tanh, or shrink the initial log-std.
# - Reward improves then collapses: with `reward_scale` large, the TD error and
#   therefore the gradient scale with it. The maths is unchanged; the optimiser
#   is not. Scale the reward, or normalise the return.
# - Agent reaches the target and immediately leaves: `arrived()` requires low
#   speed as well as low distance. If a variant drops the speed condition, flying
#   through at speed collects the bonus, so the optimal policy becomes a loop.
# - Episode ends at exactly `max_steps` every time and the value function looks
#   pessimistic near the end: `truncated` is being treated as `terminated`, so the
#   bootstrap is cut off while the mass is still moving.
