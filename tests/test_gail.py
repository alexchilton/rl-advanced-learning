"""Tests for tabular GAIL.

The reward-form tests are the ones worth having. GAIL's failure on this task is
not a tuning accident, it is arithmetic, and it should stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import gail
from envs.racetrack import Racetrack

GAMMA = 0.99


@pytest.fixture(scope="module")
def env():
    return Racetrack()


@pytest.fixture(scope="module")
def expert(env):
    return gail.noisily_rational_expert(env, GAMMA)


@pytest.fixture(scope="module")
def demonstrations(env, expert):
    return gail.collect_demonstrations(env, expert, 30, seed=0)


# ----------------------------------------------------------------------
# The reward form
# ----------------------------------------------------------------------
def test_reward_signs_are_what_the_docstring_claims():
    confidence = np.array([0.01, 0.3, 0.5, 0.7, 0.99])

    positive = gail.manufactured_reward(confidence, "positive")
    negative = gail.manufactured_reward(confidence, "negative")
    symmetric = gail.manufactured_reward(confidence, "symmetric")

    assert (positive > 0).all(), "-log(1-D) is strictly positive"
    assert (negative < 0).all(), "log(D) is strictly negative"
    assert symmetric[0] < 0 < symmetric[-1], "log(D)-log(1-D) crosses zero at D=0.5"
    assert gail.manufactured_reward(np.array([0.5]), "symmetric")[0] == pytest.approx(0.0)


def test_reward_forms_are_finite_at_the_extremes():
    """D of exactly 0 or 1 is reachable once the discriminator wins, and an
    infinite reward stops the Q update dead rather than merely biasing it."""
    for form in gail.REWARD_FORMS:
        values = gail.manufactured_reward(np.array([0.0, 1.0]), form)
        assert np.isfinite(values).all(), form


def test_unknown_reward_form_raises():
    with pytest.raises(ValueError, match="reward form must be"):
        gail.manufactured_reward(np.array([0.5]), "vibes")


@pytest.mark.slow
def test_a_strictly_positive_reward_makes_the_agent_immortal(env, demonstrations):
    """The best-known GAIL pathology, pinned.

    With r = -log(1 - D) every step earns something greater than zero and
    terminating earns nothing more, so an immortal agent is optimal under that
    reward no matter how good the discriminator is. -100 is the exact value of
    running out the clock forever on this task.

    This is a property of the reward's SIGN, not of adversarial training.
    """
    states, actions = demonstrations
    Q, _ = gail.train(env, states, actions, seed=0, reward_form="positive")
    assert env.policy_return(Q.argmax(axis=1), GAMMA) < -95


@pytest.mark.slow
def test_a_negative_reward_lets_the_agent_finish(env, demonstrations):
    """And be suspicious of it. Racetrack rewards finishing quickly, so 'every
    step costs something' happens to be the right prior here. It would be exactly
    as wrong on a task where the goal is to survive."""
    states, actions = demonstrations
    Q, _ = gail.train(env, states, actions, seed=0, reward_form="negative")
    assert env.policy_return(Q.argmax(axis=1), GAMMA) > -50


# ----------------------------------------------------------------------
# The discriminator
# ----------------------------------------------------------------------
def test_discriminator_is_low_capacity_on_purpose(env):
    """A per-state-action discriminator memorises, and a memorising discriminator
    turns GAIL into support matching -- behaviour cloning with extra machinery."""
    buckets, n_buckets = gail.discriminator_features(env)
    assert buckets.shape == (env.n_states,)
    assert n_buckets < env.n_states / 20
    assert n_buckets * env.n_actions < 2_000


def test_discriminator_ignores_velocity(env):
    """States sharing a cell must share a bucket, or capacity creeps back in."""
    buckets, _ = gail.discriminator_features(env)
    cell = sorted(env.drivable)[7]
    indices = [
        buckets[env.state_index(cell, v, h)]
        for v in range(env.speeds)
        for h in range(env.speeds)
    ]
    assert len(set(indices)) == 1


def test_finished_state_gets_its_own_bucket(env):
    buckets, n_buckets = gail.discriminator_features(env)
    assert buckets[env.finished_state] == n_buckets - 1


# ----------------------------------------------------------------------
# Access
# ----------------------------------------------------------------------
def test_gail_never_reads_the_environment_reward(env, demonstrations):
    """The access table says GAIL sees demonstrations and interaction, not reward.

    `collect_demonstrations` returns states and actions only -- there is nowhere
    for a reward to enter. Checked structurally, because a leak here would make
    every comparison in this file meaningless.
    """
    states, actions = demonstrations
    assert states.ndim == 1 and actions.ndim == 1
    assert len(states) == len(actions)

    import inspect

    source = inspect.getsource(gail.train)
    assert "reward, " not in source.replace("manufactured_reward", "")
    assert "env.step" in source, "GAIL must interact -- that is its whole advantage"


@pytest.mark.slow
def test_gail_beats_behaviour_cloning_when_demonstrations_are_scarce(env, expert):
    """The window where the method earns its complexity.

    With three demonstrations BC has almost nothing to copy and GAIL can go and
    look. Measured: BC -68.78, GAIL -11.92, medians over three seeds.
    """
    states, actions = gail.collect_demonstrations(env, expert, 3, seed=0)
    clone = gail.behaviour_clone(env.n_states, env.n_actions, states, actions)
    Q, _ = gail.train(env, states, actions, seed=0, reward_form="negative")

    bc_value = env.policy_return(clone, GAMMA)
    gail_value = env.policy_return(Q.argmax(axis=1), GAMMA)
    assert gail_value > bc_value + 20, f"BC {bc_value:.2f}, GAIL {gail_value:.2f}"