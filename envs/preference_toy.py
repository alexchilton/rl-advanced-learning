"""A toy language task with a KNOWN true reward, for preference-based RL.

Real RLHF cannot measure its own central failure: the true reward is a human's
judgement, so when the policy over-optimises the learned reward model nobody can
say by how much. Here the true reward is a formula, so every number can be checked.

    prompt     one of N_PROMPTS integers; prompt p asks for "token p"
    response   a sequence of up to MAX_LEN tokens from a vocabulary of VOCAB, ended by EOS
    true reward  r*(p, y) = (#tokens equal to p) - 0.3 * (#other tokens)

So a good response repeats the requested token, and every other token costs.
The best response is the requested token MAX_LEN times.

Labelers compare two responses and say which is better, Bradley-Terry style:
P(y1 preferred) = sigmoid((score(y1) - score(y2)) / temperature). They differ in
what they score:

    human        the true reward
    length_bias  the true reward plus `bias` per token: longer looks better
                 (Singhal et al. 2023 measured this in real preference data)
    constitution an AI labeler following a written rule instead of the true reward:
                 "prefer responses that mention the requested token, and penalise
                 repetition". It agrees with the human on most pairs and disagrees
                 systematically on one thing (Bai et al. 2022 use a model, not a rule;
                 the point here is only that the labeler's rule is not the goal)

`reference_sample` is the SFT starting point: a model of what people already
write, before any preference learning. Responses have random length 2-10, and a
token is the requested one 30% of the time.
"""

from __future__ import annotations

import numpy as np

VOCAB = 8
EOS = VOCAB              # token id VOCAB ends a response
N_PROMPTS = VOCAB
MAX_LEN = 12
OTHER_PENALTY = 0.3


def true_reward(prompt, response):
    response = [t for t in response if t != EOS]
    hits = sum(t == prompt for t in response)
    return hits - OTHER_PENALTY * (len(response) - hits)


def length(response):
    return sum(t != EOS for t in response)


def reference_sample(rng, prompt):
    n = int(rng.integers(2, 11))
    tokens = [int(prompt) if rng.random() < 0.3 else int(rng.integers(VOCAB)) for _ in range(n)]
    return tokens + [EOS]


def score(labeler, prompt, response, bias=0.5):
    if labeler == "human":
        return true_reward(prompt, response)
    if labeler == "length_bias":
        return true_reward(prompt, response) + bias * length(response)
    if labeler == "constitution":
        tokens = [t for t in response if t != EOS]
        mentions = float(prompt in tokens)
        repetition = len(tokens) - len(set(tokens))
        return 2.0 * mentions - 0.5 * repetition
    raise ValueError(f"unknown labeler {labeler!r}")


def label(rng, labeler, prompt, y1, y2, temperature=0.5, bias=0.5):
    """1 if y1 is preferred, else 0. Bradley-Terry noise with the given temperature."""
    diff = (score(labeler, prompt, y1, bias) - score(labeler, prompt, y2, bias)) / temperature
    return int(rng.random() < 1.0 / (1.0 + np.exp(-diff)))


def preference_data(rng, n_pairs, labeler="human", temperature=0.5, bias=0.5):
    """Pairs of reference responses to random prompts, with one label each.
    Returns a list of (prompt, chosen, rejected)."""
    data = []
    for _ in range(n_pairs):
        p = int(rng.integers(N_PROMPTS))
        y1, y2 = reference_sample(rng, p), reference_sample(rng, p)
        if label(rng, labeler, p, y1, y2, temperature, bias):
            data.append((p, y1, y2))
        else:
            data.append((p, y2, y1))
    return data
