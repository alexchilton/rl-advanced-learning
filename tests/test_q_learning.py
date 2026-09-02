"""Tests for tabular Q-learning and the sparse-vs-dense claims.

The tie-breaking test pins a bug that made an entire experiment wrong, and the
shaping-margin test pins the identity that explains the most surprising result in
the repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import q_learning
from envs.gridworld import POTENTIAL_SCALE, GridWorld

GAMMA = 0.95


@pytest.fixture(scope="module")
def small():
    return GridWorld(size=6, reward_mode="dense", gamma=GAMMA, max_steps=200)


# ----------------------------------------------------------------------
# Tie-breaking
# ----------------------------------------------------------------------
def test_argmax_breaks_ties_at_random():
    """np.argmax returns the FIRST maximal entry. A Q table of zeros is entirely
    ties, so a deterministic argmax makes the agent pick action 0 in every
    unvisited state -- driving into the same wall until epsilon rescues it.

    Measured before this was fixed: sparse-reward Q-learning on an 8x8 grid
    scored 0.0000 against an optimum of 0.5133 and never improved.
    """
    rng = np.random.default_rng(0)
    zeros = np.zeros(4)
    picks = {q_learning.argmax_random_ties(zeros, rng) for _ in range(200)}
    assert picks == {0, 1, 2, 3}, f"all-tied values must reach every action, got {picks}"


def test_argmax_still_picks_the_maximum():
    rng = np.random.default_rng(0)
    values = np.array([0.0, 5.0, 1.0, 5.0])
    picks = {q_learning.argmax_random_ties(values, rng) for _ in range(200)}
    assert picks == {1, 3}, "only the tied maxima are eligible"

    assert q_learning.argmax_random_ties(np.array([1.0, 9.0, 2.0]), rng) == 1


def test_epsilon_zero_is_greedy(small):
    rng = np.random.default_rng(0)
    Q = np.zeros((small.n_states, small.n_actions))
    Q[0] = [0.0, 7.0, 0.0, 0.0]
    assert all(q_learning.epsilon_greedy(Q, 0, 0.0, rng) == 1 for _ in range(50))


def test_epsilon_one_is_uniform(small):
    rng = np.random.default_rng(0)
    Q = np.zeros((small.n_states, small.n_actions))
    Q[0] = [0.0, 7.0, 0.0, 0.0]
    picks = {q_learning.epsilon_greedy(Q, 0, 1.0, rng) for _ in range(200)}
    assert picks == set(range(small.n_actions))


# ----------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------
def test_q_learning_solves_a_small_grid(small):
    _, curve = q_learning.train(small, gamma=GAMMA, n_steps=30_000, eval_every=5_000, seed=0)
    assert max(v for _, v in curve) >= small.optimal_return(GAMMA) - 1e-6


def test_q_is_accurate_where_the_agent_goes_and_wrong_where_it_does_not(small):
    """The check a reward curve cannot give you -- and the answer is two-sided.

    After 60,000 steps on a 6x6 grid the greedy policy is exactly optimal, and
    **17 of 36 states still have a Q error above 1.2**. Q-learning learned the
    decision, not the value function.

    That is correct behaviour, not a bug: epsilon-greedy converges onto its own
    trajectory, so states off that path are visited rarely and their estimates stay
    near the initialisation. It is worth pinning because it breaks a common
    assumption -- that a converged policy implies a converged Q -- and because any
    method that later reads Q away from the visited states (offline RL, planning,
    reward inference) is reading numbers that were never learned.
    """
    Q, _ = q_learning.train(small, gamma=GAMMA, n_steps=60_000, eval_every=60_000, seed=0)
    truth = small.true_q(GAMMA)

    # Where the greedy policy actually goes.
    policy = Q.argmax(axis=1)
    visited = set()
    for episode in range(30):
        small.reset(seed=episode)
        visited.add(small.state)
        for _ in range(small.max_steps):
            _, _, terminated, truncated, _ = small.step(int(policy[small.state]))
            visited.add(small.state)
            if terminated or truncated:
                break
    on_path = [s for s in sorted(visited) if small.cell(s) != small.goal]
    off_path = [s for s in range(small.n_states) if s not in visited]

    on_error = np.abs(Q[on_path] - truth[on_path]).max()
    off_error = np.abs(Q[off_path] - truth[off_path]).max()

    # The policy is exactly optimal.
    assert small.policy_return(policy, GAMMA) >= small.optimal_return(GAMMA) - 1e-6

    # The values are not, anywhere. Measured: 0.689 on the visited path and 1.278
    # off it. With a CONSTANT learning rate Q never converges even where it is
    # visited -- it tracks, bouncing by roughly alpha times the TD error forever.
    # An absolute threshold here would be wrong, which is how the first version of
    # this test failed.
    assert off_error > 1.5 * on_error, (
        f"expected values to be clearly worse off the visited path: "
        f"on {on_error:.3f}, off {off_error:.3f}"
    )


def test_curve_starts_at_step_zero_and_uses_the_eval_env(small):
    scorer = GridWorld(size=6, reward_mode="sparse", gamma=GAMMA, max_steps=200)
    _, curve = q_learning.train(
        small, gamma=GAMMA, n_steps=4_000, eval_every=2_000, eval_env=scorer, seed=0
    )
    assert [s for s, _ in curve] == [0, 2000, 4000]
    # Scored on the sparse objective, so the untrained policy is worth 0, not the
    # dense reward's larger number.
    assert curve[0][1] == pytest.approx(0.0, abs=1e-9)


def test_steps_to_threshold():
    curve = [(0, -5.0), (100, -2.0), (200, 1.0), (300, 3.0)]
    assert q_learning.steps_to_threshold(curve, 1.0) == 200
    assert q_learning.steps_to_threshold(curve, 99.0) is None


def test_training_does_not_mutate_the_environments_dynamics(small):
    before = {s: {a: list(small.P[s][a]) for a in range(small.n_actions)}
              for s in range(small.n_states)}
    q_learning.train(small, gamma=GAMMA, n_steps=2_000, eval_every=2_000, seed=0)
    assert small.P == before


# ----------------------------------------------------------------------
# The sparse / dense / shaped claims
# ----------------------------------------------------------------------
def test_shaping_margin_equals_the_sparse_return():
    """The identity that explains why potential-based shaping learns slowly.

        V_shaped(s) = V_sparse(s) - Phi(s)

    and standing still forever is worth exactly -Phi(s), because a stationary
    transition pays (gamma - 1) * Phi each step. So the margin between solving the
    task and loitering is V_sparse(s) -- unchanged -- while the values themselves
    grow by -Phi.

    Shaping inflates the numbers and leaves the decision the same size. That is a
    signal-to-noise problem, and it is invisible in the theorem.
    """
    for size in (6, 10, 14):
        sparse = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
        shaped = GridWorld(size=size, reward_mode="shaped", gamma=GAMMA)
        potential = POTENTIAL_SCALE * sparse.distance(sparse.start)

        margin = shaped.optimal_return(GAMMA) - potential
        assert margin == pytest.approx(sparse.optimal_return(GAMMA), abs=1e-6)


def test_the_margin_shrinks_as_a_fraction_of_the_values():
    """Which is the part that hurts: the same decision, larger numbers."""
    ratios = []
    for size in (6, 10, 14):
        sparse = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
        shaped = GridWorld(size=size, reward_mode="shaped", gamma=GAMMA)
        ratios.append(sparse.optimal_return(GAMMA) / shaped.optimal_return(GAMMA))
    assert ratios == sorted(ratios, reverse=True), f"ratio should fall with size: {ratios}"


@pytest.mark.slow
def test_dense_reward_learns_faster_than_sparse_on_a_large_grid():
    size = 12
    scorer = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA, max_steps=300)
    optimal = scorer.optimal_return(GAMMA)

    hits = {}
    for mode in ("sparse", "dense"):
        env = GridWorld(size=size, reward_mode=mode, gamma=GAMMA, max_steps=300)
        reached = []
        for seed in (0, 1, 2):
            _, curve = q_learning.train(
                env, gamma=GAMMA, n_steps=120_000, eval_every=5_000,
                eval_env=scorer, seed=seed,
            )
            step = q_learning.steps_to_threshold(curve, optimal * 0.95)
            if step is not None:
                reached.append(step)
        assert len(reached) >= 2, f"{mode} failed to solve {size}x{size}"
        hits[mode] = float(np.median(reached))

    assert hits["dense"] < hits["sparse"], hits


@pytest.mark.slow
def test_the_misspecified_reward_never_reaches_the_goal():
    scorer = GridWorld(size=8, reward_mode="sparse", gamma=GAMMA, max_steps=300)
    env = GridWorld(size=8, reward_mode="misspecified", gamma=GAMMA, max_steps=300)
    _, curve = q_learning.train(
        env, gamma=GAMMA, n_steps=60_000, eval_every=10_000, eval_env=scorer, seed=0
    )
    assert max(v for _, v in curve) == pytest.approx(0.0, abs=1e-9)
