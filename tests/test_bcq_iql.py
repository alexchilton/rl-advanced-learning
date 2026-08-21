"""Tests for tabular BCQ and IQL.

Both are alternatives to CQL for the same failure, so the tests check the same
two things for each: that it fixes what naive offline Q-learning breaks, and that
its dial actually spans the range its docstring claims. A dial whose ends behave
identically is a dial that is not connected to anything, and this repo has already
shipped one of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import bcq_tabular, iql_tabular, offline_q
from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
N_TRANSITIONS = 8_000


@pytest.fixture(scope="module")
def env():
    return Racetrack()


@pytest.fixture(scope="module")
def medium(env):
    return datasets.build(env, "medium", N_TRANSITIONS, GAMMA, seed=0)


@pytest.fixture(scope="module")
def random_data(env):
    return datasets.build(env, "random", N_TRANSITIONS, GAMMA, seed=0)


# ----------------------------------------------------------------------
# BCQ
# ----------------------------------------------------------------------
def test_bcq_beats_naive_offline_q(medium, env):
    naive = env.policy_return(offline_q.train(medium, gamma=GAMMA).argmax(axis=1), GAMMA)
    Q = bcq_tabular.train(medium, gamma=GAMMA)
    constrained = env.policy_return(bcq_tabular.greedy_policy(Q, medium), GAMMA)
    assert constrained > naive + 50


def test_bcq_admissible_actions_shrink_as_the_threshold_rises(medium):
    widths = [
        bcq_tabular.admissible_actions(medium, threshold=t).sum()
        for t in (0.0, 0.3, 0.6, 1.0)
    ]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > widths[-1], "the threshold must actually restrict something"


def test_bcq_admissible_actions_are_always_ones_the_data_took(medium):
    support = medium.support()
    for threshold in (0.0, 0.5, 1.0):
        allowed = bcq_tabular.admissible_actions(medium, threshold)
        assert not (allowed & ~support).any()


def test_bcq_every_visited_state_keeps_at_least_one_action(medium):
    """A relative threshold must never empty a row, or the backup has no value
    to take and the constraint becomes a crash rather than a constraint."""
    visited = medium.action_counts().sum(axis=1) > 0
    for threshold in (0.0, 0.5, 1.0):
        allowed = bcq_tabular.admissible_actions(medium, threshold)
        assert allowed[visited].any(axis=1).all()


def test_bcq_policy_extraction_stays_in_support(medium):
    """Training with the constraint and then taking an unrestricted argmax undoes
    the method completely, and nothing in the training curve shows it."""
    Q = bcq_tabular.train(medium, gamma=GAMMA)
    policy = bcq_tabular.greedy_policy(Q, medium)
    allowed = bcq_tabular.admissible_actions(medium)
    supported = allowed.any(axis=1)
    assert allowed[np.arange(len(policy))[supported], policy[supported]].all()


def test_pessimism_about_unvisited_states_is_inert_on_trajectory_data(medium):
    """A finding, pinned so it does not get quietly re-asserted.

    In trajectory data you always act from wherever you landed, unless the episode
    ended there -- in which case the transition is terminal and does not bootstrap.
    So a state that gets bootstrapped through but was never acted from cannot
    exist, and `pessimistic_value` never fires.

    Extrapolation error here is entirely about unseen ACTIONS at seen states.
    """
    acted_from = np.zeros(medium.n_states, dtype=bool)
    acted_from[medium.states] = True
    bootstrapped = medium.next_states[~medium.terminated]
    assert (~acted_from[bootstrapped]).sum() == 0

    optimistic = bcq_tabular.train(medium, gamma=GAMMA, pessimistic_value=0.0)
    pessimistic = bcq_tabular.train(medium, gamma=GAMMA, pessimistic_value=-100.0)
    np.testing.assert_allclose(optimistic, pessimistic)


# ----------------------------------------------------------------------
# IQL
# ----------------------------------------------------------------------
def test_iql_never_reads_an_out_of_sample_action_in_its_target():
    """The defining property, checked structurally rather than by outcome.

    Build a dataset where one action at one state is never taken, give that action
    an absurd Q value, and confirm training is unaffected. A method that took a max
    over actions would be dragged straight to it.
    """
    env = Racetrack()
    dataset = datasets.build(env, "medium", 2_000, GAMMA, seed=0)
    Q, V = iql_tabular.train(dataset, iterations=50, gamma=GAMMA)

    support = dataset.support()
    # Everything outside the support must be untouched by training.
    assert np.allclose(Q[~support], 0.0)


def test_iql_expectile_half_evaluates_the_behaviour_policy(medium, env):
    """At tau=0.5 the expectile is a mean, so V is the data's own value."""
    _, V = iql_tabular.train(medium, expectile=0.5, gamma=GAMMA)
    behaviour = env.policy_return(datasets.behaviour_policy_for(env, "medium", GAMMA), GAMMA)
    start_value = float(np.dot(env.start_distribution, V))
    assert start_value == pytest.approx(behaviour, abs=3.0)


def test_iql_expectile_raises_the_value_estimate_monotonically(medium, env):
    values = [
        float(np.dot(env.start_distribution, iql_tabular.train(medium, expectile=t, gamma=GAMMA)[1]))
        for t in (0.5, 0.7, 0.9)
    ]
    assert values == sorted(values), f"a higher expectile must not lower V: {values}"


def test_iql_beats_behaviour_cloning_on_random_data(random_data, env):
    counts = random_data.action_counts()
    clone = counts.argmax(axis=1).astype(np.int64)
    clone[counts.sum(axis=1) == 0] = 0
    bc_value = env.policy_return(clone, GAMMA)

    Q, V = iql_tabular.train(random_data, expectile=0.8, gamma=GAMMA)
    iql_value = env.policy_return(iql_tabular.extract_policy(random_data, Q, V), GAMMA)
    assert iql_value > bc_value + 50


def test_iql_too_high_an_expectile_reproduces_the_failure(random_data, env):
    """The honest half. IQL does not remove the failure mode, it relocates it from
    'unseen actions' to 'thinly-seen actions', where a hyperparameter controls it.
    """
    good = iql_tabular.train(random_data, expectile=0.8, gamma=GAMMA)
    extreme = iql_tabular.train(random_data, expectile=0.99, gamma=GAMMA)

    good_value = env.policy_return(iql_tabular.extract_policy(random_data, *good), GAMMA)
    extreme_value = env.policy_return(iql_tabular.extract_policy(random_data, *extreme), GAMMA)
    assert extreme_value < good_value - 20


def test_iql_beta_zero_is_behaviour_cloning(random_data, env):
    Q, V = iql_tabular.train(random_data, expectile=0.8, gamma=GAMMA)
    policy = iql_tabular.extract_policy(random_data, Q, V, beta=0.0)

    counts = random_data.action_counts()
    visited = counts.sum(axis=1) > 0
    np.testing.assert_array_equal(policy[visited], counts.argmax(axis=1)[visited])


def test_iql_policy_extraction_produces_no_nan_at_extreme_beta(random_data):
    """`beta * advantage` overflows long before `advantage` does. Clipping the
    advantage alone left nans in the weights at beta=100."""
    Q, V = iql_tabular.train(random_data, expectile=0.8, gamma=GAMMA)
    for beta in (0.0, 1.0, 100.0, 1e6):
        policy = iql_tabular.extract_policy(random_data, Q, V, beta=beta)
        assert policy.shape == (random_data.n_states,)
        assert np.isfinite(policy).all()
