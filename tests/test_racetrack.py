"""Tests for Racetrack, and for the claims that justify its existence.

The interesting ones are at the bottom. This environment was added because
behaviour cloning does NOT fail properly on the others, and those tests pin that
reasoning down so a future change cannot quietly remove the demonstration while
leaving the file in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.racetrack import ACCELERATIONS, SIMPLE_TRACK, Racetrack

GAMMA = 0.99


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="max_speed"):
        Racetrack(max_speed=0)
    with pytest.raises(ValueError, match="noise must be"):
        Racetrack(noise=2.0)
    with pytest.raises(ValueError, match="on_crash must be"):
        Racetrack(on_crash="explode")
    with pytest.raises(ValueError, match="unknown track character"):
        Racetrack(track=("  ##X#F", "  SS  ."))
    with pytest.raises(ValueError, match="no start line"):
        Racetrack(track=("###F",))
    with pytest.raises(ValueError, match="no finish line"):
        Racetrack(track=("SS##",))


def test_state_encoding_round_trips():
    env = Racetrack(track=SIMPLE_TRACK)
    for cell in list(env.drivable)[:20]:
        for vertical in range(env.speeds):
            for horizontal in range(env.speeds):
                s = env.state_index(cell, vertical, horizontal)
                assert env.decode(s) == (cell, vertical, horizontal)


def test_finished_state_has_no_position():
    env = Racetrack(track=SIMPLE_TRACK)
    with pytest.raises(ValueError, match="no position"):
        env.decode(env.finished_state)


def test_starts_on_the_start_line_at_rest():
    env = Racetrack(track=SIMPLE_TRACK)
    for seed in range(10):
        env.reset(seed=seed)
        cell, vertical, horizontal = env.decode(env.state)
        assert cell in env.start_cells
        assert (vertical, horizontal) == (0, 0)


def test_path_sweeps_every_cell_it_crosses():
    """Checking only the destination would let the car jump a wall at speed 4,
    turning a corner into a shortcut and quietly making the task easier."""
    env = Racetrack(track=SIMPLE_TRACK)
    path = env._path((9, 2), vertical=4, horizontal=0)
    assert path == [(8, 2), (7, 2), (6, 2), (5, 2)]

    diagonal = env._path((9, 2), vertical=2, horizontal=2)
    assert diagonal[-1] == (7, 4)
    assert len(diagonal) == 2

    assert env._path((5, 5), vertical=0, horizontal=0) == []


def test_speed_is_clamped_to_the_limit():
    env = Racetrack(track=SIMPLE_TRACK, noise=0.0, max_speed=2)
    cell = (8, 3)
    # Accelerate up repeatedly; vertical speed must stop at max_speed.
    accelerate_up = ACCELERATIONS.index((1, 0))
    state = env.state_index(cell, 2, 0)
    next_state, _, _ = env._resolve(cell, 2, 0, accelerate_up)
    assert next_state is not None
    _, vertical, _ = env.decode(next_state)
    assert vertical == 2

    decelerate = ACCELERATIONS.index((-1, 0))
    next_state, _, _ = env._resolve(cell, 0, 0, decelerate)
    _, vertical, _ = env.decode(next_state)
    assert vertical == 0


# ----------------------------------------------------------------------
# Crash behaviour -- the reason this environment has two modes
# ----------------------------------------------------------------------
def test_crash_stop_leaves_the_car_on_the_track_at_rest():
    env = Racetrack(track=SIMPLE_TRACK, noise=0.0, on_crash="stop")
    # (3, 5) is the top-right of the vertical corridor; driving right leaves it.
    cell = (5, 5)
    right = ACCELERATIONS.index((0, 1))
    next_state, reward, terminated = env._resolve(cell, 0, 0, right)
    landed, vertical, horizontal = env.decode(next_state)
    assert not terminated
    assert landed in env.drivable, "the car must stop ON the track, not inside a wall"
    assert (vertical, horizontal) == (0, 0), "a crash removes all velocity"
    assert reward == pytest.approx(-1.0)


def test_crash_restart_returns_to_the_start_line():
    env = Racetrack(track=SIMPLE_TRACK, noise=0.0, on_crash="restart")
    cell = (5, 5)
    right = ACCELERATIONS.index((0, 1))
    next_state, _, terminated = env._resolve(cell, 0, 0, right)
    assert next_state is None, "restart mode signals a crash for P to scatter over starts"
    assert not terminated


def test_crash_mode_changes_where_a_crashed_car_ends_up():
    """The two modes must actually behave differently, or the default is pointless."""
    stop = Racetrack(track=SIMPLE_TRACK, noise=0.0, on_crash="stop")
    restart = Racetrack(track=SIMPLE_TRACK, noise=0.0, on_crash="restart")
    start_states = {stop.state_index(c, 0, 0) for c in stop.start_cells}

    crashing_state = stop.state_index((5, 5), 0, 0)
    right = ACCELERATIONS.index((0, 1))

    stop_next = {t[1] for t in stop.P[crashing_state][right]}
    restart_next = {t[1] for t in restart.P[crashing_state][right]}
    assert restart_next <= start_states
    assert not (stop_next <= start_states)


def test_crash_never_terminates_the_episode():
    """Recovery has to be part of the problem, or there is nothing for DAgger
    to teach."""
    for mode in ("stop", "restart"):
        env = Racetrack(track=SIMPLE_TRACK, on_crash=mode)
        for state in range(env.n_states):
            if state == env.finished_state:
                continue
            for action in range(env.n_actions):
                for _, next_state, _, terminated in env.P[state][action]:
                    if terminated:
                        assert next_state == env.finished_state


def test_crossing_the_finish_line_terminates():
    env = Racetrack(track=SIMPLE_TRACK, noise=0.0)
    # Row 1 of SIMPLE_TRACK is drivable to column 11, finish at column 12.
    cell = (1, 11)
    right = ACCELERATIONS.index((0, 1))
    next_state, _, terminated = env._resolve(cell, 0, 0, right)
    assert terminated
    assert next_state == env.finished_state


def test_the_finished_state_is_absorbing():
    env = Racetrack(track=SIMPLE_TRACK)
    for action in range(env.n_actions):
        outcomes = env.P[env.finished_state][action]
        assert outcomes == [(1.0, env.finished_state, 0.0, True)]


# ----------------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------------
def test_exactly_solvable_and_the_expert_finishes():
    env = Racetrack()
    policy = env.optimal_policy(GAMMA)
    assert np.all(np.isfinite(env.true_v(GAMMA)))

    finished = 0
    for seed in range(20):
        env.reset(seed=seed)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, _ = env.step(int(policy[env.state]))
            if terminated or truncated:
                break
        finished += terminated
    assert finished == 20, "the exact optimal policy must finish every episode"


def test_noise_makes_the_task_harder():
    clean = Racetrack(noise=0.0).optimal_return(GAMMA)
    noisy = Racetrack(noise=0.3).optimal_return(GAMMA)
    assert noisy < clean


# ----------------------------------------------------------------------
# The reason this environment exists
# ----------------------------------------------------------------------
def nearest_neighbour_clone(env, states, actions):
    scale = np.array([env.rows, env.cols, env.max_speed, env.max_speed], dtype=float)

    def feat(s):
        if int(s) == env.finished_state:
            return np.zeros(4)
        (r, c), v, h = env.decode(int(s))
        return np.array([r, c, v, h], dtype=float) / scale

    demonstrated = np.array([feat(s) for s in states])
    table = np.zeros(env.n_states, dtype=np.int64)
    for s in range(env.n_states):
        table[s] = actions[int(np.argmin(np.linalg.norm(demonstrated - feat(s), axis=1)))]
    return table


def collect(env, expert, n_labels, seed=1):
    states, actions = [], []
    episode = 0
    while len(states) < n_labels:
        env.reset(seed=seed * 1000 + episode)
        episode += 1
        for _ in range(env.max_steps):
            states.append(env.state)
            actions.append(int(expert[env.state]))
            _, _, terminated, truncated, _ = env.step(int(expert[env.state]))
            if terminated or truncated or len(states) >= n_labels:
                break
    return np.array(states[:n_labels]), np.array(actions[:n_labels])


def test_behaviour_cloning_fails_from_few_demonstrations():
    """The demonstration this environment was added for.

    A clone fitted to a handful of expert labels must be substantially worse than
    the expert. If this ever stops holding, the BC and DAgger material has quietly
    lost its subject and `experiments/bc_vs_dagger.py` is measuring nothing.
    """
    env = Racetrack(on_crash="stop")
    expert = env.optimal_policy(GAMMA)
    optimal = env.optimal_return(GAMMA)

    states, actions = collect(env, expert, n_labels=20)
    clone = nearest_neighbour_clone(env, states, actions)
    cloned_value = env.policy_return(clone, GAMMA)

    assert cloned_value < optimal - 2.0, (
        f"BC should lose real return here: clone {cloned_value:.2f} vs optimal {optimal:.2f}"
    )


def test_a_free_reset_would_destroy_the_demonstration():
    """Why the default is `stop` rather than the textbook `restart`.

    Under `restart` a crash teleports the learner back to the start line, which is
    ON the expert's own state distribution. The environment hands the clone a free
    correction every time it errs, so the covariate shift never accumulates and BC
    looks far better than it is.
    """
    expert_labels = 20
    results = {}
    for mode in ("stop", "restart"):
        env = Racetrack(on_crash=mode)
        expert = env.optimal_policy(GAMMA)
        states, actions = collect(env, expert, expert_labels)
        clone = nearest_neighbour_clone(env, states, actions)
        results[mode] = env.optimal_return(GAMMA) - env.policy_return(clone, GAMMA)

    assert results["stop"] > results["restart"], (
        f"expected the free reset to flatter the clone: "
        f"stop gap {results['stop']:.2f}, restart gap {results['restart']:.2f}"
    )


def test_the_expert_covers_almost_none_of_the_state_space():
    """Context for any BC result here. Report coverage alongside the return, or a
    reader cannot tell compounding error from simply having no data."""
    env = Racetrack()
    expert = env.optimal_policy(GAMMA)
    states, _ = collect(env, expert, n_labels=100)
    assert env.demonstration_coverage(states) < 0.05


def test_rendering_runs():
    env = Racetrack(track=SIMPLE_TRACK)
    env.reset(seed=0)
    assert len(env.render().splitlines()) == env.rows + 1
    assert "A" in env.render()

    policy = env.optimal_policy(GAMMA)
    states = [env.state]
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(int(policy[env.state]))
        states.append(env.state)
        if terminated or truncated:
            break
    assert len(env.render_trajectory(states).splitlines()) == env.rows

    grid = env.cell_values(env.true_v(GAMMA))
    assert grid.shape == (env.rows, env.cols)
