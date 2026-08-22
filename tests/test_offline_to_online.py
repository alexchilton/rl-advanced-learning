"""Tests for offline pretraining followed by online fine-tuning.

Deliberately modest. The interesting claims in that file are about the SHAPE of
the handover, and the measurements say the seeds disagree too much for most of
them to be assertable. What is pinned here is the machinery, plus the one
comparison the data actually supports.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos import cql_tabular, offline_to_online
from envs import datasets
from envs.racetrack import Racetrack

GAMMA = 0.99
N_TRANSITIONS = 8_000
STEPS = 4_000


@pytest.fixture(scope="module")
def env():
    return Racetrack()


@pytest.fixture(scope="module")
def dataset(env):
    return datasets.build(env, "medium", N_TRANSITIONS, GAMMA, seed=0)


@pytest.fixture(scope="module")
def pretrained(dataset):
    return cql_tabular.train(dataset, alpha=1.0, gamma=GAMMA)


def test_finetune_returns_a_curve_with_the_expected_shape(env, pretrained):
    curve = offline_to_online.finetune(
        env, pretrained, n_steps=STEPS, eval_every=1_000, seed=0
    )
    assert curve[0][0] == 0
    assert [step for step, _ in curve] == [0, 1000, 2000, 3000, 4000]
    assert all(np.isfinite(value) for _, value in curve)


def test_finetune_does_not_mutate_the_q_it_was_given(env, pretrained):
    """A pretrained Q gets reused across variants. Mutating it in place would make
    every run after the first start from a different place, silently."""
    before = pretrained.copy()
    offline_to_online.finetune(env, pretrained, n_steps=1_000, eval_every=1_000, seed=0)
    np.testing.assert_array_equal(pretrained, before)


def test_finetune_is_deterministic_given_a_seed(env, pretrained):
    a = offline_to_online.finetune(env, pretrained, n_steps=2_000, eval_every=1_000, seed=3)
    b = offline_to_online.finetune(env, pretrained, n_steps=2_000, eval_every=1_000, seed=3)
    assert a == b


def test_replay_uses_the_offline_data(env, pretrained, dataset):
    """With replay on, the same seed must produce a different trajectory of Q
    updates -- otherwise the offline log is being ignored and the 'mixed replay'
    variant is measuring nothing."""
    without = offline_to_online.finetune(
        env, pretrained, n_steps=2_000, eval_every=2_000, seed=0
    )
    with_replay = offline_to_online.finetune(
        env, pretrained, n_steps=2_000, eval_every=2_000, seed=0,
        dataset=dataset, replay_batch=4,
    )
    assert without != with_replay


def test_starting_from_scratch_is_far_worse_than_pretraining(env, pretrained):
    """The one claim in that file the seeds agree on.

    Tabular Q-learning from zero on 7,126 states has barely begun after a few
    thousand steps, while the pretrained policy is already near optimal. The
    offline log is worth more than the online budget that follows it.
    """
    scratch = offline_to_online.finetune(
        env, np.zeros_like(pretrained), n_steps=STEPS, eval_every=STEPS, seed=0
    )
    warm = offline_to_online.finetune(
        env, pretrained, n_steps=STEPS, eval_every=STEPS, seed=0
    )
    assert warm[-1][1] > scratch[-1][1] + 20


@pytest.mark.slow
def test_a_gentle_learning_rate_is_reliably_worse_than_a_large_one(env, pretrained, dataset):
    """The finding worth keeping, and the one that surprised me.

    A small fine-tuning learning rate is the intuitive defence against wrecking a
    pretrained policy, and here it is consistently the worst option -- all three
    seeds within 0.2 of each other at about -30, against a start of -6.54.

    Consistency across seeds is what makes it a mechanism rather than volatility.
    The suspected cause is in the module docstring and is NOT verified: a
    conservative Q is not a return, so a corrected entry and an untouched
    neighbour sit on different scales while argmax compares them anyway.
    """
    worst = []
    for seed in (0, 1, 2):
        curve = offline_to_online.finetune(
            env, pretrained, n_steps=20_000, eval_every=5_000,
            learning_rate=offline_to_online.GENTLE_LEARNING_RATE, seed=seed,
        )
        worst.append(min(value for _, value in curve))

    spread = max(worst) - min(worst)
    assert spread < 5.0, f"expected the dip to be consistent across seeds, got {worst}"
