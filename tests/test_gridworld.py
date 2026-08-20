"""Tests for GridWorld, and for the claims its docstring makes.

The reward-mode tests are the interesting ones. They are not testing code so much
as testing a theorem: potential-based shaping preserves the optimal policy, and
the one-sided variant does not. If `test_potential_shaping_preserves_optimal_policy`
ever fails, the shaping is wrong and every experiment built on it is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs import solvers
from envs.gridworld import ACTION_DELTAS, HACK_BONUS, GridWorld

GAMMA = 0.95


# ----------------------------------------------------------------------
# Construction and dynamics
# ----------------------------------------------------------------------
def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="size must be"):
        GridWorld(size=1)
    with pytest.raises(ValueError, match="slip must be"):
        GridWorld(slip=1.5)
    with pytest.raises(ValueError, match="reward_mode must be"):
        GridWorld(reward_mode="vibes")
    with pytest.raises(ValueError, match="start .* is a wall"):
        GridWorld(size=3, start=(0, 0), walls=[(0, 0)])
    with pytest.raises(ValueError, match="same cell"):
        GridWorld(size=3, start=(2, 2), goal=(2, 2))


def test_index_and_cell_round_trip():
    env = GridWorld(size=6)
    for s in range(env.n_states):
        assert env.index(env.cell(s)) == s


def test_edges_and_walls_leave_the_agent_in_place():
    env = GridWorld(size=3, walls=[(1, 1)])
    # Up from the top row.
    assert env._move((0, 1), ACTION_DELTAS[0]) == (0, 1)
    # Into a wall.
    assert env._move((0, 1), ACTION_DELTAS[2]) == (0, 1)
    # Ordinary move.
    assert env._move((0, 0), ACTION_DELTAS[1]) == (0, 1)


def test_reset_and_step_follow_the_gymnasium_contract():
    env = GridWorld(size=4, max_steps=10)
    obs, info = env.reset(seed=0)
    assert obs == env.index(env.start)
    assert info["steps"] == 0

    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_before_reset_raises():
    env = GridWorld(size=3)
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(0)


def test_invalid_action_raises():
    env = GridWorld(size=3)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="outside"):
        env.step(4)


# ----------------------------------------------------------------------
# terminated vs truncated -- the distinction the whole repo leans on
# ----------------------------------------------------------------------
def test_truncation_is_not_termination():
    """Walk into a wall until the step budget runs out."""
    env = GridWorld(size=4, max_steps=5)
    env.reset(seed=0)
    for step in range(5):
        _, _, terminated, truncated, _ = env.step(0)  # up, from the top row
        assert not terminated
        assert truncated == (step == 4)


def test_reaching_the_goal_terminates_and_is_never_truncated():
    """Termination wins over the step budget, even when they coincide.

    An episode that is both `terminated` and `truncated` would make the bootstrap
    ambiguous, and the ambiguity would be resolved differently by every algorithm
    that consumes it.
    """
    env = GridWorld(size=2, max_steps=1, start=(0, 0), goal=(0, 1))
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(1)  # right, straight into the goal
    assert terminated
    assert not truncated  # even though t == max_steps
    assert reward == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Slip
# ----------------------------------------------------------------------
def test_slip_zero_is_deterministic():
    env = GridWorld(size=4, slip=0.0)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            assert len(env.P[s][a]) == 1


def test_slip_splits_probability_between_perpendiculars():
    env = GridWorld(size=5, slip=0.2)
    s = env.index((2, 2))
    outcomes = env.P[s][1]  # right
    probs = sorted(t[0] for t in outcomes)
    assert probs == pytest.approx([0.1, 0.1, 0.8])


def test_slip_lowers_the_optimal_return():
    """Sanity check that slip actually reaches the value function."""
    deterministic = GridWorld(size=5, slip=0.0, reward_mode="sparse")
    slippery = GridWorld(size=5, slip=0.3, reward_mode="sparse")
    v_det = deterministic.optimal_return(GAMMA)
    v_slip = slippery.optimal_return(GAMMA)
    assert v_slip < v_det  # slipping wastes steps, so the discounted return falls


# ----------------------------------------------------------------------
# Reward modes -- the reason this environment exists
# ----------------------------------------------------------------------
def optimal_policy_reaches_goal(env, gamma, max_steps=200):
    """Follow the greedy policy from the start and report whether it terminates."""
    pi = env.optimal_policy(gamma)
    env.reset(seed=0)
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(int(pi[env.state]))
        if terminated:
            return True
        if truncated:
            return False
    return False


@pytest.mark.parametrize("size", [4, 5, 7])
def test_potential_shaping_preserves_optimal_policy(size):
    """Ng et al. 1999, checked numerically.

    The shaped and sparse problems have different rewards and different value
    functions, but the same optimal policy -- everywhere it is uniquely optimal.
    Ties are excluded because argmax tie-breaking is arbitrary and not part of
    the claim.
    """
    sparse = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
    shaped = GridWorld(size=size, reward_mode="shaped", gamma=GAMMA)

    q_sparse = sparse.true_q(GAMMA)
    q_shaped = shaped.true_q(GAMMA)

    best_sparse = q_sparse.max(axis=1, keepdims=True)
    unique = (np.abs(q_sparse - best_sparse) < 1e-9).sum(axis=1) == 1

    assert unique.sum() > 0, "no uniquely-optimal states to compare"
    np.testing.assert_array_equal(
        q_sparse.argmax(axis=1)[unique],
        q_shaped.argmax(axis=1)[unique],
    )


def test_potential_shaping_changes_the_value_function_it_does_not_change_the_policy():
    """Guards against a shaping term that is accidentally zero everywhere --
    which would make the test above pass for the wrong reason."""
    sparse = GridWorld(size=5, reward_mode="sparse", gamma=GAMMA)
    shaped = GridWorld(size=5, reward_mode="shaped", gamma=GAMMA)
    assert not np.allclose(sparse.true_q(GAMMA), shaped.true_q(GAMMA))


def test_shaped_advantage_is_the_potential_difference():
    """The exact identity: Q_shaped(s,a) = Q_sparse(s,a) - Phi(s)."""
    sparse = GridWorld(size=5, reward_mode="sparse", gamma=GAMMA)
    shaped = GridWorld(size=5, reward_mode="shaped", gamma=GAMMA)
    q_sparse, q_shaped = sparse.true_q(GAMMA), shaped.true_q(GAMMA)

    goal_index = sparse.index(sparse.goal)
    for s in range(sparse.n_states):
        if s == goal_index or sparse.cell(s) in sparse.walls:
            continue
        potential = -0.1 * sparse.distance(sparse.cell(s))
        np.testing.assert_allclose(q_shaped[s], q_sparse[s] - potential, atol=1e-6)


def test_sparse_and_dense_optimal_policies_both_reach_the_goal():
    for mode in ("sparse", "dense", "shaped"):
        env = GridWorld(size=6, reward_mode=mode, gamma=GAMMA)
        assert optimal_policy_reaches_goal(env, GAMMA), f"{mode} failed to reach the goal"


def test_misspecified_reward_is_hacked():
    """The demo, asserted.

    Under the one-sided progress bonus the optimal policy never enters the goal.
    It oscillates next to it, collecting the bonus forever. Nothing is broken --
    the agent is doing exactly what it was asked to do.
    """
    env = GridWorld(size=6, reward_mode="misspecified", gamma=GAMMA)
    assert not optimal_policy_reaches_goal(env, GAMMA)

    # And it is worth more than solving the task honestly.
    sparse = GridWorld(size=6, reward_mode="sparse", gamma=GAMMA)
    honest_policy = sparse.optimal_policy(GAMMA)
    assert env.policy_return(env.optimal_policy(GAMMA), GAMMA) > env.policy_return(
        honest_policy, GAMMA
    )


def test_the_hack_depends_on_the_discount():
    """The uncomfortable half of the reward-hacking lesson.

    Farming the bonus is worth HACK_BONUS / (1 - gamma^2). Reaching the goal is
    worth 1 + HACK_BONUS and then stops. Which one wins depends on the discount,
    so the same misspecified reward is harmless at gamma=0.8 and catastrophic at
    gamma=0.99.

    The bug does not change. The horizon does. That is how a reward bug sits
    dormant until someone raises gamma to solve a longer task.
    """
    env = GridWorld(size=6, reward_mode="misspecified")

    assert optimal_policy_reaches_goal(env, 0.80), "expected the hack to lose at gamma=0.80"
    assert not optimal_policy_reaches_goal(env, 0.99), "expected the hack to win at gamma=0.99"

    # And the crossover is where the arithmetic says it is, not somewhere else.
    def hack_wins(gamma):
        return HACK_BONUS / (1 - gamma**2) > 1 + HACK_BONUS

    for gamma in (0.80, 0.85, 0.90, 0.95, 0.99):
        assert hack_wins(gamma) == (not optimal_policy_reaches_goal(env, gamma)), (
            f"prediction and value iteration disagree at gamma={gamma}"
        )


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------
def test_obs_modes():
    env = GridWorld(size=3, obs_mode="index")
    assert env.observation(4) == 4
    assert env.obs_dim == 1

    env = GridWorld(size=3, obs_mode="onehot")
    obs = env.observation(4)
    assert obs.shape == (9,) and obs[4] == 1.0 and obs.sum() == 1.0
    assert env.obs_dim == 9

    env = GridWorld(size=3, obs_mode="xy")
    np.testing.assert_allclose(env.observation(4), [0.5, 0.5])
    assert env.obs_dim == 2

    env = GridWorld(size=3, obs_mode="nonsense")
    with pytest.raises(ValueError, match="unknown obs_mode"):
        env.observation(0)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_same_seed_gives_the_same_trajectory():
    """Until this holds, no debugging session can conclude anything."""

    def rollout(seed):
        env = GridWorld(size=5, slip=0.3, max_steps=50)
        env.reset(seed=seed)
        rng = np.random.default_rng(123)
        trace = []
        for _ in range(50):
            obs, reward, terminated, truncated, _ = env.step(int(rng.integers(0, 4)))
            trace.append((obs, reward, terminated, truncated))
            if terminated or truncated:
                break
        return trace

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_rendering_runs_and_has_the_right_shape():
    env = GridWorld(size=4, walls=[(1, 1)])
    env.reset(seed=0)
    assert len(env.render().splitlines()) == 4
    assert "A" in env.render()

    pi = env.optimal_policy(GAMMA)
    assert len(env.render_policy(pi).splitlines()) == 4
    assert len(env.render_values(env.true_v(GAMMA)).splitlines()) == 4
