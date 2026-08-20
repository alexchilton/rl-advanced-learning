"""Tests for the exact solvers.

These matter more than they look. Everything else in the repo is checked against
`value_iteration`, so if it is wrong, every later "the agent learned the wrong
thing" conclusion is wrong too. It is the one thing here with no safety net above
it, so it gets checked against hand-computed answers and against an independent
second method.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs import solvers
from envs.gridworld import GridWorld

GAMMA = 0.9


def two_state_chain():
    """s0 --a0--> s1 (r=0), s1 --any--> terminal (r=1).

    Small enough to write the answer down: V(s1) = 1, V(s0) = gamma.
    """
    P = {
        0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, 0.0, False)]},
        1: {0: [(1.0, 1, 1.0, True)], 1: [(1.0, 1, 1.0, True)]},
    }
    return P, 2, 2


def test_value_iteration_matches_hand_computed_values():
    P, n_states, n_actions = two_state_chain()
    V, Q, pi = solvers.value_iteration(P, n_states, n_actions, GAMMA)

    assert V[1] == pytest.approx(1.0)
    assert V[0] == pytest.approx(GAMMA)
    assert pi[0] == 0  # only a0 makes progress


def test_terminal_transitions_do_not_bootstrap():
    """The whole point of masking: a terminal transition's value is r, not r + gamma*V."""
    P = {0: {0: [(1.0, 0, 1.0, True)]}}
    V, Q, _ = solvers.value_iteration(P, 1, 1, GAMMA)
    assert Q[0, 0] == pytest.approx(1.0)

    # Same transition, not marked terminal: now it bootstraps into itself and the
    # value becomes the geometric sum. This is bug #1 in the catalogue, and this
    # is exactly how much it changes the answer.
    P_unmasked = {0: {0: [(1.0, 0, 1.0, False)]}}
    V_unmasked, _, _ = solvers.value_iteration(P_unmasked, 1, 1, GAMMA)
    assert V_unmasked[0] == pytest.approx(1.0 / (1.0 - GAMMA))


def test_policy_iteration_agrees_with_value_iteration():
    env = GridWorld(size=5, slip=0.15, reward_mode="sparse")
    V_vi, Q_vi, pi_vi = solvers.value_iteration(env.P, env.n_states, env.n_actions, GAMMA)
    V_pi, Q_pi, pi_pi = solvers.policy_iteration(env.P, env.n_states, env.n_actions, GAMMA)

    np.testing.assert_allclose(V_vi, V_pi, atol=1e-8)
    np.testing.assert_allclose(Q_vi, Q_pi, atol=1e-8)


def test_policy_evaluation_of_optimal_policy_equals_optimal_value():
    env = GridWorld(size=5, slip=0.1, reward_mode="sparse")
    V, _, pi = solvers.value_iteration(env.P, env.n_states, env.n_actions, GAMMA)
    V_pi = solvers.policy_evaluation(env.P, env.n_states, env.n_actions, pi, GAMMA)
    np.testing.assert_allclose(V, V_pi, atol=1e-8)


def test_optimal_policy_beats_every_other_deterministic_policy():
    """A strong check: no policy can have a higher value in any state."""
    env = GridWorld(size=4, slip=0.2, reward_mode="sparse")
    V_star, _, _ = solvers.value_iteration(env.P, env.n_states, env.n_actions, GAMMA)

    rng = np.random.default_rng(0)
    for _ in range(20):
        pi = rng.integers(0, env.n_actions, size=env.n_states)
        V = solvers.policy_evaluation(env.P, env.n_states, env.n_actions, pi, GAMMA)
        assert np.all(V <= V_star + 1e-8)


def test_value_iteration_raises_instead_of_returning_garbage():
    """gamma=1 on a task that need not terminate does not converge. It must say so.

    A solver that quietly returned its last iterate here would poison every
    ground-truth comparison downstream.
    """
    P = {0: {0: [(1.0, 0, 1.0, False)]}}
    with pytest.raises(RuntimeError, match="did not converge"):
        solvers.value_iteration(P, 1, 1, gamma=1.0, max_iters=500)


def test_epsilon_greedy_policy_rows_sum_to_one():
    Q = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    pi = solvers.epsilon_greedy_policy(Q, epsilon=0.1)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    assert pi[0, 3] == pytest.approx(0.9 + 0.1 / 4)
    assert pi[1, 0] == pytest.approx(0.9 + 0.1 / 4)


def test_epsilon_one_is_uniform():
    Q = np.array([[0.0, 5.0]])
    pi = solvers.epsilon_greedy_policy(Q, epsilon=1.0)
    np.testing.assert_allclose(pi, 0.5)


def test_as_stochastic_rejects_rows_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        solvers.as_stochastic(np.array([[0.5, 0.2]]), 1, 2)


def test_policy_return_matches_value_at_start_state():
    env = GridWorld(size=4, reward_mode="sparse")
    _, _, pi = solvers.value_iteration(env.P, env.n_states, env.n_actions, GAMMA)
    V = solvers.policy_evaluation(env.P, env.n_states, env.n_actions, pi, GAMMA)
    expected = V[env.index(env.start)]
    assert env.policy_return(pi, GAMMA) == pytest.approx(expected)
