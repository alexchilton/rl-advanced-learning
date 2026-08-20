"""Tests for offline datasets, offline Q-learning and tabular CQL.

The last three are the ones that matter. They pin the claims the offline material
is built on: that naive offline Q-learning really does fail, that the conservative
penalty really does fix it, and that behaviour cloning really does collapse on
data with no good trajectory to copy. If any of those stops being true the
experiments are measuring nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import cql_tabular, offline_q
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


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def test_rejects_unknown_quality(env):
    with pytest.raises(ValueError, match="quality must be one of"):
        datasets.behaviour_policy_for(env, "excellent", GAMMA)


def test_behaviour_policies_are_valid_distributions(env):
    for quality in datasets.QUALITIES:
        policy = datasets.behaviour_policy_for(env, quality, GAMMA)
        assert policy.shape == (env.n_states, env.n_actions)
        np.testing.assert_allclose(policy.sum(axis=1), 1.0)
        assert (policy >= 0).all()


def test_dataset_has_the_requested_size_and_valid_indices(medium, env):
    assert len(medium) == N_TRANSITIONS
    assert medium.states.min() >= 0 and medium.states.max() < env.n_states
    assert medium.actions.min() >= 0 and medium.actions.max() < env.n_actions


def test_dataset_exposes_no_environment(medium):
    """Offline means offline. A dataset that can be stepped is not a dataset."""
    assert not hasattr(medium, "step")
    assert not hasattr(medium, "reset")


def test_dataset_arrays_must_match_in_length():
    with pytest.raises(ValueError, match="mismatched lengths"):
        datasets.Dataset([0, 1], [0], [0.0], [1], [False], 2, 2)


def test_support_and_coverage_agree(medium):
    support = medium.support()
    assert support[medium.states, medium.actions].all()
    assert medium.coverage() == pytest.approx(support.mean())


def test_expert_data_has_narrower_coverage_than_random(env):
    expert = datasets.build(env, "expert", N_TRANSITIONS, GAMMA, seed=0)
    random_data = datasets.build(env, "random", N_TRANSITIONS, GAMMA, seed=0)
    assert expert.coverage() < random_data.coverage()


def test_behaviour_policy_recovers_the_empirical_distribution(medium):
    policy = medium.behaviour_policy()
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    counts = medium.action_counts()
    visited = counts.sum(axis=1) > 0
    np.testing.assert_allclose(
        policy[visited], counts[visited] / counts[visited].sum(axis=1, keepdims=True)
    )


# ----------------------------------------------------------------------
# The failure, and the fix
# ----------------------------------------------------------------------
def test_naive_offline_q_learning_fails(medium, env):
    """Extrapolation error, pinned.

    `max_a'` selects actions the log never took, whose Q is still at its zero
    initialisation -- which on this all-negative-reward task is the most
    optimistic number in the table. The resulting policy never finishes.
    """
    Q = offline_q.train(medium, gamma=GAMMA)
    _, actual = offline_q.value_gap(Q, env, medium, GAMMA)
    assert actual < -50, f"naive offline Q should fail badly here, got {actual:.2f}"


def test_the_agent_believes_its_own_broken_estimate(medium, env):
    """The reason this failure is dangerous rather than merely bad.

    The corrupted quantity IS the agent's self-estimate, so it cannot be used to
    detect the corruption. Any offline result validated against the agent's own Q
    is worthless.
    """
    Q = offline_q.train(medium, gamma=GAMMA)
    predicted, actual = offline_q.value_gap(Q, env, medium, GAMMA)
    assert predicted > actual + 50, (
        f"expected a large optimism gap: believes {predicted:.2f}, gets {actual:.2f}"
    )


def test_cql_fixes_it(medium, env):
    Q = cql_tabular.train(medium, alpha=1.0, gamma=GAMMA)
    _, actual = offline_q.value_gap(Q, env, medium, GAMMA)
    optimal = env.optimal_return(GAMMA)
    assert actual > optimal - 2.0, f"CQL should land near optimal, got {actual:.2f}"


def test_cql_with_alpha_zero_is_exactly_naive_offline_q(medium):
    """The two files must be one edit apart, or the comparison is not about the
    penalty."""
    naive = offline_q.train(medium, gamma=GAMMA)
    zero_alpha = cql_tabular.train(medium, alpha=0.0, gamma=GAMMA)
    np.testing.assert_allclose(naive, zero_alpha)


def test_conservatism_reduces_out_of_support_actions(medium):
    naive = offline_q.out_of_support_fraction(offline_q.train(medium, gamma=GAMMA), medium)
    conservative = offline_q.out_of_support_fraction(
        cql_tabular.train(medium, alpha=1.0, gamma=GAMMA), medium
    )
    assert conservative < naive


def test_out_of_support_is_weighted_by_where_the_data_goes(medium):
    """An unweighted count over all states sits near 0.87 regardless of the
    policy, because most states are unreachable and their greedy action is
    arbitrary forever. The metric has to be measured where the agent goes."""
    Q = cql_tabular.train(medium, alpha=1.0, gamma=GAMMA)
    weighted = offline_q.out_of_support_fraction(Q, medium)

    support = medium.support()
    greedy = Q.argmax(axis=1)
    unweighted = 1.0 - support[np.arange(len(greedy)), greedy].mean()

    assert weighted < 0.2
    assert unweighted > 0.5, "the unweighted version should be uninformative here"


def test_too_much_conservatism_hurts(env, medium):
    """The dial has two ends. A large penalty makes Q stop being a value."""
    good = cql_tabular.train(medium, alpha=1.0, gamma=GAMMA)
    excessive = cql_tabular.train(medium, alpha=100.0, gamma=GAMMA)
    good_value = env.policy_return(good.argmax(axis=1), GAMMA)
    excessive_value = env.policy_return(excessive.argmax(axis=1), GAMMA)
    assert excessive_value < good_value - 5.0

    # And its predicted values become impossible: no return here can exceed 0.
    predicted = float(np.dot(env.start_distribution, excessive.max(axis=1)))
    assert predicted > 0, "expected the over-conservative Q to stop meaning anything"


# ----------------------------------------------------------------------
# The row that justifies the field
# ----------------------------------------------------------------------
def test_behaviour_cloning_collapses_on_random_data_and_cql_does_not(env):
    """Same log, same access, opposite outcome.

    Random behaviour contains no good trajectory to copy, so BC has nothing to
    imitate. It contains the reward on every transition, so a method that reads
    the reward can assemble a good policy from pieces of bad ones.
    """
    dataset = datasets.build(env, "random", N_TRANSITIONS, GAMMA, seed=0)

    counts = dataset.action_counts()
    clone = counts.argmax(axis=1).astype(np.int64)
    unseen = counts.sum(axis=1) == 0
    clone[unseen] = np.random.default_rng(0).integers(0, env.n_actions, int(unseen.sum()))

    bc_value = env.policy_return(clone, GAMMA)
    cql_value = env.policy_return(
        cql_tabular.train(dataset, alpha=1.0, gamma=GAMMA).argmax(axis=1), GAMMA
    )
    assert bc_value < -50, f"BC should collapse on random data, got {bc_value:.2f}"
    assert cql_value > bc_value + 50, (
        f"CQL should recover a usable policy: BC {bc_value:.2f}, CQL {cql_value:.2f}"
    )


def test_offline_methods_cannot_exceed_the_optimal_policy(env, medium):
    """The ceiling is real. Nothing offline can reach a state the log never had."""
    optimal = env.optimal_return(GAMMA)
    for alpha in (0.5, 1.0, 2.0):
        Q = cql_tabular.train(medium, alpha=alpha, gamma=GAMMA)
        assert env.policy_return(Q.argmax(axis=1), GAMMA) <= optimal + 1e-6
