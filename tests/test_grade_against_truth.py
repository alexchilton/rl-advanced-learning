"""Tests for the grading bridge, and for the parity claim it rests on.

The parity test is the important one. Everything `grade_cliff_walking` says is
only meaningful if our CliffWalking and Gymnasium's are the same environment, and
that is a claim about two codebases rather than about one.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.cliff import CliffWalking
from experiments import grade_against_truth as grader

GAMMA = 1.0
gymnasium = pytest.importorskip("gymnasium")


# ----------------------------------------------------------------------
# Parity
# ----------------------------------------------------------------------
def test_our_cliff_walking_matches_gymnasiums():
    """Values, not transitions.

    The transition tables legitimately differ on eleven states -- the ten cliff
    cells and the goal -- because neither environment can ever BE in a cliff cell
    and Gymnasium handles termination in `step()` rather than in `P`. Comparing
    raw tables would report a difference that means nothing. Comparing exact
    values is the check that matters.
    """
    parity = grader.verify_cliff_parity(GAMMA)

    assert parity["differing_states"] == parity["cliff_and_goal"]
    assert parity["max_value_difference"] == pytest.approx(0.0, abs=1e-9)
    assert parity["policies_agree"]
    assert parity["start_value_theirs"] == pytest.approx(-13.0)
    assert parity["start_value_ours"] == pytest.approx(-13.0)


# ----------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------
def test_exact_q_grades_perfectly():
    env = CliffWalking()
    grade = grader.grade_cliff_walking(env.true_q(GAMMA), GAMMA)

    assert grade["max_error"] == pytest.approx(0.0)
    assert grade["policy_agreement"] == pytest.approx(1.0)
    assert grade["exact_return"] == pytest.approx(-13.0)


def test_a_policy_that_never_finishes_scores_minus_infinity():
    """At gamma=1 with a cost per step, a looping policy really is worth -inf.

    `policy_evaluation` declines to converge rather than returning its last
    iterate, which is correct, and the grader has to turn that into an answer
    instead of an exception. Reporting a large negative number would suggest the
    agent is nearly there. It never arrives.
    """
    env = CliffWalking()
    grade = grader.grade_cliff_walking(np.zeros((env.n_states, env.n_actions)), GAMMA)
    assert np.isneginf(grade["exact_return"])
    assert "never reaches the goal" in grader.report(grade)


def test_high_policy_agreement_can_still_never_reach_the_goal():
    """The reason this file exists.

    'Mostly right' is not a property policies have. A Q table agreeing with the
    optimal policy in most states can still circle forever, and no aggregate
    score over states reveals it.
    """
    env = CliffWalking()
    rng = np.random.default_rng(0)
    grade = grader.grade_cliff_walking(env.true_q(GAMMA) + rng.normal(0, 2.0, (48, 4)), GAMMA)

    assert grade["policy_agreement"] > 0.8
    assert np.isneginf(grade["exact_return"])


def test_unreachable_states_are_not_graded():
    """Corrupting only the cliff cells must change nothing. Grading states no
    policy can occupy would report noise as a finding."""
    env = CliffWalking()
    Q = env.true_q(GAMMA).copy()
    for cell in env.cliff:
        Q[env.index(cell)] = 999.0

    grade = grader.grade_cliff_walking(Q, GAMMA)
    assert grade["max_error"] == pytest.approx(0.0)
    assert grade["policy_agreement"] == pytest.approx(1.0)


def test_tied_optimal_actions_count_as_correct():
    """Punishing an arbitrary choice between two equally good actions would make
    the agreement number meaningless."""
    env = CliffWalking()
    truth = env.true_q(GAMMA)
    tied = np.abs(truth - truth.max(axis=1, keepdims=True)) < 1e-9
    assert tied.sum(axis=1).max() > 1, "expected at least one state with tied actions"

    grade = grader.grade_cliff_walking(truth, GAMMA)
    assert grade["policy_agreement"] == pytest.approx(1.0)


def test_rejects_a_wrongly_shaped_q():
    with pytest.raises(ValueError, match="expected Q of shape"):
        grader.grade_cliff_walking(np.zeros((10, 4)), GAMMA)


def test_worst_states_are_ordered_and_actionable():
    env = CliffWalking()
    Q = env.true_q(GAMMA).copy()
    Q[0] += 50.0
    grade = grader.grade_cliff_walking(Q, GAMMA, worst=3)

    errors = [entry["error"] for entry in grade["worst_states"]]
    assert errors == sorted(errors, reverse=True)
    assert grade["worst_states"][0]["state"] == 0
    assert "cell" in grade["worst_states"][0]
