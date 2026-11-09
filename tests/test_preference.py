"""Preference tier: the toy task's definitions, and that each method moves the policy the right way."""

import numpy as np
import pytest
import torch

from algos import dpo, reward_model, rlhf, sequence_policy as sp
from envs import preference_toy as toy

torch.set_num_threads(1)


def test_true_reward_and_labelers():
    assert toy.true_reward(3, [3, 3, 1, toy.EOS]) == pytest.approx(2 - 0.3)
    long, short = [3, 3, 1, 1, 1, 1], [3, 3]          # true reward 0.8 vs 2.0
    assert toy.score("human", 3, short) > toy.score("human", 3, long)
    assert toy.score("length_bias", 3, long, bias=0.5) > toy.score("length_bias", 3, short, bias=0.5)
    # the constitution penalises the repetition the true reward wants
    assert toy.score("constitution", 3, [3, 1, 2]) > toy.score("constitution", 3, [3, 3, 3])
    assert toy.true_reward(3, [3, 3, 3]) > toy.true_reward(3, [3, 1, 2])


@pytest.fixture(scope="module")
def sft_policy():
    return sp.sft(n_steps=600)


def copy_of(policy):
    clone = sp.SequencePolicy()
    clone.load_state_dict(policy.state_dict())
    return clone


def test_log_prob_matches_sampling_distribution(sft_policy):
    """The summed log-probability of a one-token response equals log p(token) + log p(EOS | token)."""
    lp = sft_policy.log_prob([2], [[5, toy.EOS]])
    logits = sft_policy.logits(torch.tensor([2]), torch.tensor([[sp.START, 5]]))
    direct = torch.log_softmax(logits[0, 0], -1)[5] + torch.log_softmax(logits[0, 1], -1)[toy.EOS]
    assert float(lp) == pytest.approx(float(direct), abs=1e-5)


def test_reward_model_beats_chance():
    model = reward_model.train(toy.preference_data(np.random.default_rng(0), 500), n_epochs=10)
    assert reward_model.accuracy(model, n=500) > 0.7


def test_rlhf_on_the_true_reward_improves_it(sft_policy):
    before = sp.evaluate(sft_policy)["true_reward"]
    log = rlhf.train(copy_of(sft_policy), rlhf.true_reward_fn, n_steps=60, estimator="rloo", eval_every=60)
    assert log[-1]["true_reward"] > before + 3


def test_dpo_improves_the_true_reward(sft_policy):
    before = sp.evaluate(sft_policy)["true_reward"]
    pairs = toy.preference_data(np.random.default_rng(1), 1_000, temperature=0.1)
    log = dpo.train(copy_of(sft_policy), pairs, n_epochs=2, eval_every=10)
    assert log[-1]["true_reward"] > before + 1
