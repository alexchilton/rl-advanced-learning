"""Tests for MaxEnt IRL.

Two of these pin findings that cost real time to discover, and both would be easy
to reintroduce: demonstrations must be padded to the horizon, and the environment
has to be big enough for IRL to have an advantage at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import max_ent_irl as irl
from envs import solvers
from envs.gridworld import GridWorld
from envs.racetrack import Racetrack

GAMMA = 0.95


@pytest.fixture(scope="module")
def grid():
    return GridWorld(size=6, reward_mode="sparse", walls=[(2, 2), (3, 3)])


@pytest.fixture(scope="module")
def grid_model(grid):
    return solvers.flatten(grid.P, grid.n_states, grid.n_actions)


# ----------------------------------------------------------------------
# Sparse machinery
# ----------------------------------------------------------------------
def test_soft_value_iteration_returns_a_valid_policy(grid, grid_model):
    reward = np.zeros(grid.n_states)
    reward[grid.index(grid.goal)] = 1.0
    policy = irl.soft_value_iteration(grid_model, reward, GAMMA)

    assert policy.shape == (grid.n_states, grid.n_actions)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    assert (policy > 0).all(), "a max-entropy policy gives every action non-zero mass"


STEP_COST = 10.0


def shortest_path_reward(env, step_cost=STEP_COST):
    """-step_cost per state, 0 at the goal.

    The correct state-reward encoding of a "reach the goal" task, and NOT the
    obvious one. See `test_a_reward_on_a_terminal_state_can_never_be_collected`.

    The magnitude matters too, which is easy to miss. See
    `test_entropy_competes_with_the_reward_scale`.
    """
    reward = np.full(env.n_states, -step_cost)
    reward[env.index(env.goal)] = 0.0
    return reward


def optimal_actions(env, model, reward, gamma=GAMMA):
    """Every action tied for best under hard value iteration.

    Needed because a symmetric grid has genuine ties -- from the top-left corner
    of a square room, right and down are exactly as good -- and asserting against
    one arbitrary argmax would make the test depend on tie-breaking order.
    """
    V = np.zeros(model.n_states)
    for _ in range(500):
        Q = irl._q_from_v(model, reward, V, gamma)
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < 1e-12:
            break
        V = V_new
    Q = irl._q_from_v(model, reward, V, gamma)
    return np.abs(Q - Q.max(axis=1, keepdims=True)) < 1e-9


def test_soft_policy_prefers_the_better_action(grid, grid_model):
    reward = shortest_path_reward(grid)
    policy = irl.soft_value_iteration(grid_model, reward, GAMMA)
    best = optimal_actions(grid, grid_model, reward)

    start = grid.index(grid.start)
    assert best[start, policy[start].argmax()], (
        f"soft argmax {policy[start].argmax()} is not among the optimal actions "
        f"{np.flatnonzero(best[start])}"
    )
    assert policy[start][best[start]].sum() > 0.9, "optimal actions should hold most of the mass"


def test_entropy_competes_with_the_reward_scale(grid, grid_model):
    """MaxEnt's entropy bonus is fixed at one nat per action choice. The reward is
    not, and if it is too small the entropy wins.

    With four actions the bonus is log(4) = 1.386 per step. Against a step cost of
    1.0 the soft-optimal policy is very nearly uniform -- it would rather keep its
    options open forever than finish, because finishing ends the entropy stream.
    Measured at the start state:

        step cost  1.0   [0.251 0.249 0.249 0.251]   <- prefers the two WALLS
        step cost  5.0   [0.041 0.459 0.459 0.041]
        step cost 50.0   [0.    0.5   0.5   0.   ]

    This is why hand-writing a reward for `soft_value_iteration` is fiddly, and
    why `train` does not care: it learns theta, and theta absorbs the scale.
    """
    start = grid.index(grid.start)
    best = optimal_actions(grid, grid_model, shortest_path_reward(grid))

    weak = irl.soft_value_iteration(grid_model, shortest_path_reward(grid, 1.0), GAMMA)
    strong = irl.soft_value_iteration(grid_model, shortest_path_reward(grid, 50.0), GAMMA)

    assert weak[start][best[start]].sum() < 0.55, "at step cost 1 the entropy should dominate"
    assert strong[start][best[start]].sum() > 0.99, "at step cost 50 the reward should dominate"


def test_soft_value_iteration_does_not_overflow_at_high_gamma(grid, grid_model):
    """logsumexp without subtracting the row maximum saturates exp() as gamma
    approaches 1."""
    policy = irl.soft_value_iteration(grid_model, shortest_path_reward(grid), gamma=0.999)
    assert np.isfinite(policy).all()


def test_a_reward_on_a_terminal_state_can_never_be_collected(grid, grid_model):
    """A trap that costs an afternoon if you meet it in the wild.

    The obvious encoding of "reach the goal" as a STATE reward is +1 at the goal
    and 0 elsewhere. It does not work, and it fails silently.

    Transitions into the goal carry `terminated=True`, so `continues` is 0 and
    V(goal) never propagates back to anything. The +1 sits there, unreachable, and
    every policy has value 0 -- so the planner returns an arbitrary argmax and the
    agent wanders.

    The working encoding is -1 per step with 0 at the goal: reaching the goal is
    valuable because it STOPS the cost, not because it pays. Every "reach the
    goal" environment in this repo is really a shortest-path problem, and the
    reward has to say so.
    """
    goal = grid.index(grid.goal)

    naive = np.zeros(grid.n_states)
    naive[goal] = 1.0
    assert grid.policy_return(irl.plan(grid_model, naive, GAMMA), GAMMA) == pytest.approx(0.0)

    correct = shortest_path_reward(grid)
    assert grid.policy_return(irl.plan(grid_model, correct, GAMMA), GAMMA) == pytest.approx(
        grid.optimal_return(GAMMA)
    )


def test_expected_visitation_conserves_mass(grid, grid_model):
    """Each timestep is a probability distribution, so the total over a horizon of
    T must be T. A leak here is a silent bug that biases every gradient."""
    reward = np.zeros(grid.n_states)
    policy = irl.soft_value_iteration(grid_model, reward, GAMMA)
    for horizon in (1, 5, 20):
        total = irl.expected_visitation(grid_model, policy, grid.start_distribution, horizon)
        assert total.sum() == pytest.approx(horizon, rel=1e-9)


def test_plan_recovers_the_optimal_policy_from_the_true_reward(grid, grid_model):
    planned = irl.plan(grid_model, shortest_path_reward(grid), GAMMA)
    # Compare returns rather than action tables: ties break arbitrarily.
    assert grid.policy_return(planned, GAMMA) >= grid.optimal_return(GAMMA) - 1e-6


# ----------------------------------------------------------------------
# Demonstrations
# ----------------------------------------------------------------------
def test_demonstrations_are_padded_to_the_horizon(grid):
    """The bug that made the goal repulsive.

    `expected_visitation` runs the learner for the whole horizon and terminal
    states are absorbing, so a good policy accumulates mass in the goal. If
    demonstrations stop at termination, the expert appears to visit the goal once
    against the learner's thirty, and the gradient pushes the goal's reward DOWN.
    """
    expert = irl.noisily_rational_expert(grid, GAMMA)
    demonstrations = irl.collect_demonstrations(grid, expert, 20, seed=0, horizon=25)
    assert all(len(trajectory) == 25 for trajectory in demonstrations)


def test_padding_repeats_the_final_state_with_no_action(grid):
    expert = irl.noisily_rational_expert(grid, GAMMA)
    trajectory = irl.collect_demonstrations(grid, expert, 1, seed=0, horizon=30)[0]
    padded = [step for step in trajectory if step[1] is None]
    assert padded, "at least the final entry carries no action"
    assert len({state for state, _ in padded}) == 1, "padding repeats one state"


def test_unpadded_demonstrations_would_invert_the_goal_reward(grid):
    """The failure, reproduced deliberately so the fix cannot be undone silently.

    Truncating the padding is exactly the earlier bug, and the recovered reward at
    the goal should come out lower than the average cell rather than higher.
    """
    expert = irl.noisily_rational_expert(grid, GAMMA)
    demonstrations = irl.collect_demonstrations(grid, expert, 200, seed=0)
    features = irl.one_hot_features(grid)

    padded_reward, _ = irl.train(grid, demonstrations, features, iterations=60)
    stripped = [[step for step in t if step[1] is not None] for t in demonstrations]
    broken_reward, _ = irl.train(grid, stripped, features, iterations=60)

    goal = grid.index(grid.goal)
    assert padded_reward[goal] > padded_reward.mean()
    assert broken_reward[goal] < padded_reward[goal]


# ----------------------------------------------------------------------
# Features and identifiability
# ----------------------------------------------------------------------
def test_recovered_reward_is_mean_centred(grid):
    """Rewards are identified only up to an additive constant."""
    expert = irl.noisily_rational_expert(grid, GAMMA)
    demonstrations = irl.collect_demonstrations(grid, expert, 100, seed=0)
    _, theta = irl.train(grid, demonstrations, irl.one_hot_features(grid), iterations=40)
    assert theta.mean() == pytest.approx(0.0, abs=1e-9)


def test_racetrack_features_are_position_only():
    """Velocity is deliberately excluded, which is what makes 7,126 states share
    about a hundred parameters."""
    env = Racetrack()
    features = irl.racetrack_features(env)
    assert features.shape[0] == env.n_states

    cell = sorted(env.drivable)[5]
    rows = [features[env.state_index(cell, v, h)] for v in range(env.speeds) for h in range(env.speeds)]
    for row in rows[1:]:
        np.testing.assert_array_equal(row, rows[0])


def test_racetrack_features_are_indicators():
    env = Racetrack()
    features = irl.racetrack_features(env)
    assert set(np.unique(features)) <= {0.0, 1.0}
    assert features.sum(axis=1).max() == 1.0


# ----------------------------------------------------------------------
# The claims the demo makes
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_irl_beats_behaviour_cloning_on_a_large_state_space():
    """The reason this file moved off the gridworld.

    With 7,126 states and 10 demonstrations, per-state counting has almost nothing
    to count. A hundred-parameter reward plus a transition model does.
    """
    env = Racetrack()
    model = solvers.flatten(env.P, env.n_states, env.n_actions)
    expert = irl.noisily_rational_expert(env, GAMMA)
    demonstrations = irl.collect_demonstrations(env, expert, 10, seed=0)

    clone = irl.behaviour_clone(env.n_states, env.n_actions, demonstrations)
    reward, _ = irl.train(env, demonstrations, irl.racetrack_features(env), iterations=60)

    bc_value = env.policy_return(clone, GAMMA)
    irl_value = env.policy_return(irl.plan(model, reward), GAMMA)
    assert irl_value > bc_value + 3.0, f"BC {bc_value:.2f}, IRL {irl_value:.2f}"


@pytest.mark.slow
def test_irl_beats_the_demonstrator_it_learned_from():
    """The access table's last column. A method whose output is a policy cannot do
    this; a method whose output is a reward can, because a reward can be optimised."""
    env = Racetrack()
    model = solvers.flatten(env.P, env.n_states, env.n_actions)
    expert = irl.noisily_rational_expert(env, GAMMA)
    demonstrations = irl.collect_demonstrations(env, expert, 100, seed=0)
    reward, _ = irl.train(env, demonstrations, irl.racetrack_features(env), iterations=60)

    assert env.policy_return(irl.plan(model, reward), GAMMA) > env.policy_return(expert, GAMMA)


@pytest.mark.slow
def test_recovered_reward_replans_to_the_optimum():
    """Weaker than a transfer claim, and it is what has actually been measured.

    A transfer experiment -- learn the reward in one room, re-plan it in a room
    whose walls have moved -- is the strongest argument for recovering a reward
    at all, and it belongs here. It is NOT here yet.

    An earlier gridworld version produced clone 0.0000 against replanned +0.3138,
    but that was at gamma=0.9 with different training settings; under the current
    ones it produced 0.0000 against 0.0000. Racetrack is the right home for it,
    since two tracks of the same size share a reward feature space, but that has
    not been run. No assertion until it has.
    """
    env = GridWorld(size=7, reward_mode="sparse", walls=[(2, 2), (2, 3), (3, 2)])
    model = solvers.flatten(env.P, env.n_states, env.n_actions)
    expert = irl.noisily_rational_expert(env, GAMMA, rationality=0.55)
    demonstrations = irl.collect_demonstrations(env, expert, 200, seed=0)

    reward, _ = irl.train(env, demonstrations, irl.one_hot_features(env))
    replanned = env.policy_return(irl.plan(model, reward, GAMMA), GAMMA)

    assert replanned >= env.optimal_return(GAMMA) - 1e-6
    assert replanned > env.policy_return(expert, GAMMA)
