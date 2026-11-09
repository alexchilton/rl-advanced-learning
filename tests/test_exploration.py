"""Exploration and curriculum: the claims the tier rests on, checked on small instances."""

import numpy as np
import torch

from algos import count_based_exploration as counts
from algos import curriculum_reverse, curriculum_size, her
from envs.bit_flip import BitFlip
from envs.chain import Chain
from envs.gridworld import GridWorld

torch.set_num_threads(1)


def test_distance_to_goal_on_chain_and_grid():
    np.testing.assert_array_equal(curriculum_reverse.distance_to_goal(Chain(n=6))[:5], [5, 4, 3, 2, 1])
    grid = GridWorld(size=4, reward_mode="sparse")
    distance = curriculum_reverse.distance_to_goal(grid)
    assert distance[grid.index(grid.start)] == 6, "Manhattan distance from (0,0) to (3,3)"


def test_directed_exploration_beats_epsilon_greedy_on_a_long_chain():
    env = Chain(n=20, max_steps=80)
    first = {name: counts.train(env, n_steps=40_000, eval_every=40_000, seed=0, **kw)[2]
             for name, kw in (("eps", {}), ("optimistic", dict(q_init=1.0)), ("bonus", dict(bonus=0.1)))}
    assert first["eps"] is None, "epsilon-greedy should not reach the end of Chain(20) in 40,000 steps"
    assert first["optimistic"] is not None and first["bonus"] is not None


def test_size_curriculum_ends_with_the_optimal_chain_policy():
    make = lambda n: Chain(n=n, max_steps=4 * n)
    Q, total, _ = curriculum_size.train(make, (5, 10, 15, 20), seed=0)
    env = make(20)
    assert env.policy_return(Q.argmax(1), 0.99) >= 0.95 * env.optimal_return(0.99)
    assert total < 5_000


def test_reverse_curriculum_reaches_the_real_start():
    env = Chain(n=15, max_steps=60)
    Q, curve, stages = curriculum_reverse.train(env, n_steps=30_000, seed=0)
    assert stages[-1][1] == 14
    assert curve[-1][1] >= 0.95 * env.optimal_return(0.99)


def test_bit_flip_reward_and_termination():
    env = BitFlip(4)
    env.reset(seed=0)
    env.goal = env.state.copy()
    env.goal[2] = 1 - env.goal[2]
    _, reward, terminated, _, _ = env.step(2)
    assert reward == 0.0 and terminated


def test_her_solves_bits_plain_dqn_cannot():
    make = lambda: BitFlip(10)
    rates = {}
    for k in (0, 4):
        _, log = her.train(make(), n_episodes=600, k=k, evaluate=her.success_rate(make, episodes=30),
                           eval_every=600, seed=0)
        rates[k] = log[-1]["success"]
    assert rates[4] >= 0.9 and rates[0] <= 0.1, rates


def test_teacher_ignores_solved_and_hopeless_tasks_and_picks_the_frontier():
    """LP measured as the success rate itself would pick the solved task forever."""
    from algos.curriculum_automatic import ProgressTeacher

    teacher = ProgressTeacher(3, window=10, epsilon=0.0)
    for _ in range(10):
        teacher.update(0, 1.0)          # solved
        teacher.update(2, 0.0)          # hopeless
    for outcome in [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]:
        teacher.update(1, outcome)      # improving
    assert teacher.progress().tolist() == [0.0, 1.0, 0.0]
    assert teacher.choose(np.random.default_rng(0)) == 1


def test_automatic_curriculum_buckets_cover_every_state_including_the_real_start():
    from algos.curriculum_automatic import buckets

    env = Chain(n=30, max_steps=120)
    groups = buckets(env)
    covered = np.sort(np.concatenate(groups))
    real_start = int(np.argmax(env.start_distribution))
    assert real_start in groups[-1]
    assert len(covered) == len(set(covered.tolist())) == env.n_states - 1  # every state but the absorbing goal


def test_goal_grid_puts_the_goal_in_the_observation():
    """Without the goal in the observation the stages conflict and forgetting is the right answer."""
    from algos.curriculum_failure import GOALS, GoalGrid

    env = GoalGrid()
    seen = set()
    for goal in GOALS:
        env.set_goal(goal)
        obs, _ = env.reset(seed=0)
        seen.add(tuple(np.round(obs[2:], 3)))
    assert len(seen) == len(GOALS)


def test_the_absorbing_goal_cell_is_never_a_start():
    """Its self-loop is terminal: a start there 'succeeds' in one step with zero reward."""
    from algos.curriculum_automatic import buckets

    grid = GridWorld(size=6, reward_mode="sparse")
    goal = grid.index(grid.goal)
    assert curriculum_reverse.distance_to_goal(grid)[goal] >= 10**6
    assert goal not in np.concatenate(buckets(grid))
