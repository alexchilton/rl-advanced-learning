"""k-armed bandit: exploration with nothing else attached.

One state, k actions, a reward drawn from a distribution per arm. No transitions,
no bootstrapping, no credit assignment across time. Everything that goes wrong
here is purely an exploration problem, which is why it is the right place to start
and the right place to return to when an exploration bonus misbehaves in a real
environment.

Not a `TabularEnv`. The rewards are continuous random variables, so they do not
fit the finite `(probability, next_state, reward, terminated)` table -- and the
ground truth we want here is not a value function anyway. It is `true_means`, from
which regret follows exactly.

Regret, not reward, is the metric. Reward depends on how good the arms happen to
be; regret measures only the decisions.
"""

from __future__ import annotations

import numpy as np


class Bandit:
    """A k-armed bandit with Gaussian rewards.

    Args:
        k: number of arms.
        mean_spread: arm means are drawn from N(0, mean_spread^2) at construction.
            Smaller values make the problem harder -- the arms are closer together
            so more pulls are needed to tell them apart.
        noise: standard deviation of the reward around each arm's mean. This is
            the other half of the difficulty. `mean_spread / noise` is the signal
            to noise ratio and is the number that actually predicts how hard the
            instance is.
        drift: per-step random walk applied to every arm mean. 0 is the standard
            stationary bandit. Non-zero makes it non-stationary, which breaks
            sample-average value estimates and is why a constant step size exists.
        seed: seeds the arm means AND the reward noise.
    """

    def __init__(
        self,
        k: int = 10,
        mean_spread: float = 1.0,
        noise: float = 1.0,
        drift: float = 0.0,
        seed: int | None = None,
    ):
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")

        self.k = int(k)
        self.mean_spread = float(mean_spread)
        self.noise = float(noise)
        self.drift = float(drift)

        self._rng = np.random.default_rng(seed)
        self.true_means = self._rng.normal(0.0, self.mean_spread, size=self.k)
        self.t = 0

    # ------------------------------------------------------------------
    # Gymnasium-style interface, so the same agent loop works here.
    # A bandit episode is one step: every step terminates.
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return 0, {"step": self.t}

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.k:
            raise ValueError(f"arm {action} outside [0, {self.k})")

        reward = float(self._rng.normal(self.true_means[action], self.noise))
        info = {"regret": self.regret(action), "step": self.t}

        if self.drift:
            self.true_means += self._rng.normal(0.0, self.drift, size=self.k)
        self.t += 1

        # terminated=True: the bandit episode is exactly one step long. An agent
        # that bootstraps here is bootstrapping through a terminal transition,
        # which is the same bug as anywhere else -- and on a bandit it is easy
        # to spot, because the value estimate should equal the mean reward.
        return 0, reward, True, False, info

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.true_means))

    @property
    def best_mean(self) -> float:
        return float(np.max(self.true_means))

    def regret(self, action: int) -> float:
        """Expected reward given up by pulling `action` instead of the best arm.

        Always non-negative, and exactly zero for the best arm. Cumulative regret
        is the curve worth plotting: a good algorithm's curve flattens, a bad
        one's stays straight.
        """
        return float(self.best_mean - self.true_means[int(action)])

    def gap(self) -> float:
        """Difference between the best and second-best arm.

        The instance-difficulty parameter in every bandit regret bound. A tiny gap
        means even an optimal algorithm needs many pulls, so a flat learning curve
        is not evidence of a bug -- check this first.
        """
        top_two = np.sort(self.true_means)[-2:]
        return float(top_two[1] - top_two[0])

    def summary(self) -> str:
        lines = [
            f"k={self.k}  noise={self.noise}  drift={self.drift}",
            f"best arm {self.best_arm} with mean {self.best_mean:.3f}, gap to second {self.gap():.3f}",
            "means: " + " ".join(f"{m:+.2f}" for m in self.true_means),
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Cumulative regret grows linearly forever: the agent stopped exploring before
#   it identified the best arm, or epsilon never decays. A linear regret curve is
#   the signature of a policy that has locked on to the wrong arm.
# - Regret is negative: the agent is being scored against a stale `best_mean`
#   while `drift` is non-zero. Recompute regret at the time of the pull, which is
#   what `step` returns in `info`.
# - Sample-average estimates track well and then fall apart: `drift` is non-zero.
#   Sample averages weight every observation equally, so a 10000-pull-old reward
#   still counts. Switch to a constant step size.
# - Every algorithm looks identical: `gap()` is too small, or `noise` too large.
#   Compare on several seeded instances, not one.
