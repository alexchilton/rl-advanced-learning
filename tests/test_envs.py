"""Tests for CliffWalking, Chain, Baird, Bandit and PointMass.

Plus one contract test that runs over every finite environment at once, because
the guarantee "the solver and the environment cannot disagree" is worth checking
rather than asserting in a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs import solvers
from envs.bandit import Bandit
from envs.chain import Baird, Chain
from envs.cliff import CliffWalking
from envs.gridworld import GridWorld
from envs.pointmass import PointMass

GAMMA = 0.95


def finite_envs():
    return [
        GridWorld(size=4),
        GridWorld(size=4, slip=0.2, reward_mode="dense"),
        CliffWalking(),
        CliffWalking(slip=0.1),
        Chain(n=6),
        Chain(n=6, slip=0.2, small_reward=0.02),
        Baird(),
    ]


# ----------------------------------------------------------------------
# The contract every finite environment must satisfy
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env", finite_envs(), ids=lambda e: type(e).__name__)
def test_transition_table_is_well_formed(env):
    """`TabularEnv._validate` runs at construction, so reaching here is the test.

    Restated explicitly so a future change that weakens validation gets caught.
    """
    for s in range(env.n_states):
        for a in range(env.n_actions):
            total = sum(t[0] for t in env.P[s][a])
            assert total == pytest.approx(1.0), f"P[{s}][{a}] sums to {total}"


@pytest.mark.parametrize("env", finite_envs(), ids=lambda e: type(e).__name__)
def test_sampled_transitions_match_the_transition_table(env):
    """The guarantee, checked empirically.

    Sample many transitions from one state-action and compare the empirical next-state
    distribution to `P`. If `step()` ever stops sampling from `P`, ground truth
    silently becomes fiction, and every "the agent learned the wrong thing"
    conclusion in the repo becomes unsound.
    """
    rng = np.random.default_rng(0)
    n_samples = 4000

    for _ in range(6):
        s = int(rng.integers(0, env.n_states))
        a = int(rng.integers(0, env.n_actions))

        expected = np.zeros(env.n_states)
        for prob, s2, _, _ in env.P[s][a]:
            expected[s2] += prob

        counts = np.zeros(env.n_states)
        for _ in range(n_samples):
            env.reset(seed=None)
            env.state = s
            env.t = 0
            env.step(a)
            counts[env.state] += 1

        np.testing.assert_allclose(counts / n_samples, expected, atol=0.03)


@pytest.mark.parametrize("env", finite_envs(), ids=lambda e: type(e).__name__)
def test_value_iteration_converges_on_every_env(env):
    V, Q, pi = solvers.value_iteration(env.P, env.n_states, env.n_actions, GAMMA)
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(Q))
    assert pi.shape == (env.n_states,)


@pytest.mark.parametrize("env", finite_envs(), ids=lambda e: type(e).__name__)
def test_never_terminated_and_truncated_at_once(env):
    rng = np.random.default_rng(1)
    for episode in range(20):
        env.reset(seed=episode)
        for _ in range(env.max_steps + 5):
            _, _, terminated, truncated, _ = env.step(int(rng.integers(0, env.n_actions)))
            assert not (terminated and truncated)
            if terminated or truncated:
                break


# ----------------------------------------------------------------------
# CliffWalking
# ----------------------------------------------------------------------
def test_cliff_fall_returns_to_start_without_terminating():
    env = CliffWalking()
    env.reset(seed=0)
    assert env.cell(env.state) == (3, 0)

    _, reward, terminated, truncated, _ = env.step(1)  # right, off the cliff
    assert reward == pytest.approx(-100.0)
    assert not terminated, "falling off the cliff is not the end of the episode"
    assert env.cell(env.state) == (3, 0), "the fall returns the agent to the start"


def test_cliff_optimal_path_hugs_the_edge():
    """The Q-learning answer: along row 2, one above the cliff."""
    env = CliffWalking()
    pi = env.optimal_policy(gamma=1.0)
    env.reset(seed=0)

    visited = [env.cell(env.state)]
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(int(pi[env.state]))
        visited.append(env.cell(env.state))
        if terminated or truncated:
            break

    assert visited[-1] == env.goal, f"optimal policy did not reach the goal: {visited}"
    assert all(cell[0] == 2 for cell in visited[1:-1]), f"optimal path left row 2: {visited}"


def test_cliff_optimal_return_is_the_textbook_number():
    """13 steps at -1 each on the default 4x12 grid, undiscounted.

    gamma=1 converges here because every optimal path terminates, so value
    iteration reaches a fixed point after about as many sweeps as the path is long.
    """
    env = CliffWalking()
    assert env.optimal_return(gamma=1.0) == pytest.approx(-13.0)


def test_cliff_safe_path_is_better_under_a_stochastic_policy():
    """Why SARSA differs from Q-learning, stated as an exact fact rather than a curve.

    Evaluate the optimal (cliff-edge) policy and the row-1 safe policy under the
    same epsilon-greedy exploration. The safe policy wins, because exploration
    occasionally steps off the edge and the edge is expensive.
    """
    env = CliffWalking()
    gamma = 0.999
    epsilon = 0.1

    Q = env.true_q(gamma)
    risky = solvers.epsilon_greedy_policy(Q, epsilon)

    # A hand-built safe policy: climb to row 1, run right, drop onto the goal.
    # 15 steps instead of the optimal 13, two rows clear of the cliff.
    safe = np.zeros(env.n_states, dtype=np.int64)
    for r in range(env.rows):
        for c in range(env.cols):
            s = env.index((r, c))
            if c == env.cols - 1:
                safe[s] = 2  # down the last column onto the goal
            elif r > 1:
                safe[s] = 0  # up, away from the cliff
            else:
                safe[s] = 1  # right along row 1
    safe_stochastic = np.full((env.n_states, env.n_actions), epsilon / env.n_actions)
    safe_stochastic[np.arange(env.n_states), safe] += 1.0 - epsilon

    risky_return = env.policy_return(risky, gamma)
    safe_return = env.policy_return(safe_stochastic, gamma)
    assert safe_return > risky_return, (
        f"expected the safe path to win under exploration: "
        f"safe={safe_return:.2f} risky={risky_return:.2f}"
    )


# ----------------------------------------------------------------------
# Chain
# ----------------------------------------------------------------------
def test_chain_left_returns_to_start():
    env = Chain(n=6)
    env.reset(seed=0)
    for _ in range(3):
        env.step(1)  # right
    assert env.state == 3
    env.step(0)  # left
    assert env.state == 0


def test_chain_only_the_far_end_pays():
    env = Chain(n=5, small_reward=0.0)
    env.reset(seed=0)
    rewards = []
    for _ in range(4):
        _, reward, terminated, _, _ = env.step(1)
        rewards.append(reward)
    assert rewards[:-1] == [0.0, 0.0, 0.0]
    assert rewards[-1] == pytest.approx(1.0)
    assert terminated


def test_chain_random_walk_probability_is_exponentially_small():
    assert Chain(n=6).random_walk_success_probability() == pytest.approx(0.5**5)
    assert Chain(n=21).random_walk_success_probability() < 1e-6


def test_chain_small_reward_creates_a_local_optimum():
    """The trap: a consolation prize a myopic agent never leaves.

    Note what the contest actually is, because it is easy to get wrong. "left" is
    a RENEWABLE reward -- it pays every single step, forever, and never ends the
    episode. "right" is a ONE-OFF reward eleven steps away that terminates. So:

        V_left(0)  = small_reward / (1 - gamma)
        V_right(0) = gamma^10 * goal_reward

    At small_reward=0.05 the left-hand side is 5.0 at gamma=0.99, which beats the
    goal at every discount. The consolation prize has to be genuinely small before
    a far-sighted agent will walk past it.
    """
    env = Chain(n=12, small_reward=0.005, goal_reward=1.0)

    assert env.optimal_policy(0.5)[0] == 0, "myopic agent should take the prize"
    assert env.optimal_policy(0.99)[0] == 1, "far-sighted agent should go the distance"

    # The crossover is where the arithmetic says it is.
    for gamma in (0.5, 0.6, 0.8, 0.9, 0.99):
        prize_wins = 0.005 / (1 - gamma) > gamma**10 * 1.0
        assert (env.optimal_policy(gamma)[0] == 0) == prize_wins, (
            f"prediction and value iteration disagree at gamma={gamma}"
        )


def test_chain_a_renewable_small_reward_always_beats_a_one_off_large_one():
    """Stated on its own because it is the part that surprises people.

    No discount saves you. A reward that pays forever dominates a reward that pays
    once, and the only fix is to make it smaller or to make it stop.
    """
    env = Chain(n=12, small_reward=0.05, goal_reward=1.0)
    for gamma in (0.5, 0.9, 0.99, 0.999):
        assert env.optimal_policy(gamma)[0] == 0, f"prize should still win at gamma={gamma}"


# ----------------------------------------------------------------------
# Baird
# ----------------------------------------------------------------------
def test_baird_true_value_is_zero_and_representable():
    env = Baird()
    V = env.true_v(GAMMA)
    np.testing.assert_allclose(V, 0.0, atol=1e-9)

    # w = 0 represents it exactly, so approximation error is not the reason
    # off-policy TD diverges here.
    np.testing.assert_allclose(env.features @ np.zeros(8), 0.0)


def test_baird_policies_are_valid_and_different():
    behaviour, target = Baird.behaviour_policy(), Baird.target_policy()
    np.testing.assert_allclose(behaviour.sum(axis=1), 1.0)
    np.testing.assert_allclose(target.sum(axis=1), 1.0)
    assert not np.allclose(behaviour, target), "no off-policy-ness, no divergence"


def test_baird_features_have_the_textbook_shape():
    features = Baird.features
    assert features.shape == (7, 8)
    assert features[0, 0] == 2.0 and features[0, 7] == 1.0
    assert features[6, 6] == 1.0 and features[6, 7] == 2.0
    assert np.linalg.matrix_rank(features) == 7  # more features than states


# ----------------------------------------------------------------------
# Bandit
# ----------------------------------------------------------------------
def test_bandit_regret_is_zero_for_the_best_arm_and_positive_otherwise():
    bandit = Bandit(k=5, seed=0)
    assert bandit.regret(bandit.best_arm) == pytest.approx(0.0)
    for arm in range(bandit.k):
        assert bandit.regret(arm) >= 0.0


def test_bandit_rewards_centre_on_the_true_mean():
    bandit = Bandit(k=3, noise=1.0, seed=0)
    bandit.reset(seed=1)
    arm = 1
    samples = [bandit.step(arm)[1] for _ in range(20000)]
    assert np.mean(samples) == pytest.approx(bandit.true_means[arm], abs=0.05)


def test_bandit_episode_is_one_step():
    bandit = Bandit(k=3, seed=0)
    bandit.reset(seed=0)
    _, _, terminated, truncated, _ = bandit.step(0)
    assert terminated and not truncated


def test_bandit_drift_moves_the_means():
    stationary = Bandit(k=4, drift=0.0, seed=0)
    before = stationary.true_means.copy()
    for _ in range(100):
        stationary.step(0)
    np.testing.assert_allclose(stationary.true_means, before)

    drifting = Bandit(k=4, drift=0.1, seed=0)
    before = drifting.true_means.copy()
    for _ in range(100):
        drifting.step(0)
    assert not np.allclose(drifting.true_means, before)


def test_bandit_rejects_a_bad_arm():
    bandit = Bandit(k=3, seed=0)
    bandit.reset(seed=0)
    with pytest.raises(ValueError, match="outside"):
        bandit.step(3)


# ----------------------------------------------------------------------
# PointMass
# ----------------------------------------------------------------------
def test_pointmass_observation_scales_differ_by_more_than_an_order_of_magnitude():
    """The scaling problem, asserted. If this ever stops being true, the
    normalisation demo stops demonstrating anything."""
    env = PointMass(seed=0)
    scales = env.observation_scales()
    assert scales.max() / scales.min() > 10


def test_pointmass_reaches_large_velocities_in_practice():
    env = PointMass(seed=0)
    env.reset(seed=0)
    speeds = []
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step([1.0, 0.0])
        speeds.append(abs(float(obs[2])))
        if terminated or truncated:
            break
    assert max(speeds) > 10.0, "velocity never gets large, so scaling is not a problem here"


def test_pointmass_expert_solves_the_task():
    env = PointMass(reward_mode="dense", seed=0)
    for episode in range(5):
        env.reset(seed=episode)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, _ = env.step(env.expert_action())
            if terminated or truncated:
                break
        assert terminated, f"expert failed on episode {episode}: {env.render()}"


def test_pointmass_arrival_needs_low_speed_as_well_as_low_distance():
    env = PointMass(seed=0)
    env.reset(seed=0)
    env.position = np.zeros(2)
    env.velocity = np.array([30.0, 0.0])
    assert not env.arrived()
    env.velocity = np.zeros(2)
    assert env.arrived()


def test_pointmass_reports_clipping():
    env = PointMass(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step([5.0, 0.0])
    assert info["clipped"] == pytest.approx(0.5)
    _, _, _, _, info = env.step([0.5, 0.5])
    assert info["clipped"] == pytest.approx(0.0)


def test_pointmass_reward_scale_does_not_change_the_expert():
    """Scaling the reward leaves the problem identical and the gradients 1000x bigger.
    The environment must not quietly change anything else."""
    plain = PointMass(reward_scale=1.0, seed=0)
    scaled = PointMass(reward_scale=1000.0, seed=0)
    plain.reset(seed=3)
    scaled.reset(seed=3)
    for _ in range(20):
        _, r_plain, _, _, _ = plain.step([0.3, -0.2])
        _, r_scaled, _, _, _ = scaled.step([0.3, -0.2])
        assert r_scaled == pytest.approx(r_plain * 1000.0)


def test_pointmass_truncates_rather_than_terminates_when_the_clock_runs_out():
    env = PointMass(reward_mode="sparse", max_steps=5, seed=0)
    env.reset(seed=0)
    for step in range(5):
        _, _, terminated, truncated, _ = env.step([0.0, 0.0])
        assert not terminated
        assert truncated == (step == 4)


def test_pointmass_rejects_non_finite_actions():
    env = PointMass(seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="non-finite"):
        env.step([np.nan, 0.0])


def test_pointmass_step_before_reset_raises():
    env = PointMass(seed=0)
    with pytest.raises(RuntimeError, match="before reset"):
        env.step([0.0, 0.0])
