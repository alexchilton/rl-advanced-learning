"""A tiny prompt-conditioned language model, and supervised fine-tuning (SFT).

Shared by the preference tier (`reward_model.py`, `rlhf.py`, `dpo.py`). Not an
algorithm from a paper: it is the "pretrained model" every RLHF method starts
from, small enough to train on a laptop in seconds.

    p(y | prompt) = prod_t p(y_t | prompt, y_<t)       a GRU over the response tokens,
                                                       the prompt as the first input

`log_prob` returns the summed log-probability of each whole response, which is
the quantity every method in this tier works with: the policy gradient weights
it, the KL penalty compares it with the reference, and DPO puts it in a sigmoid.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from envs.preference_toy import EOS, MAX_LEN, N_PROMPTS, VOCAB, reference_sample

HIDDEN = 64
START = VOCAB + 1          # input token that starts every response


class SequencePolicy(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.prompt = nn.Embedding(N_PROMPTS, hidden)
        self.token = nn.Embedding(VOCAB + 2, hidden)        # VOCAB tokens, EOS, START
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, VOCAB + 1)              # next token or EOS

    def logits(self, prompts, inputs):
        h0 = self.prompt(prompts)[None]
        out, _ = self.gru(self.token(inputs), h0)
        return self.head(out)

    def log_prob(self, prompts, responses):
        """Sum of log p over each response's tokens, EOS included if present.

        `responses` is a list of token lists. Returns a tensor of shape (batch,).
        """
        lengths = [len(r) for r in responses]
        width = max(lengths)
        targets = torch.full((len(responses), width), EOS, dtype=torch.long)
        for i, r in enumerate(responses):
            targets[i, :len(r)] = torch.as_tensor(r)
        inputs = torch.cat([torch.full((len(responses), 1), START), targets[:, :-1]], dim=1)
        logp = torch.log_softmax(self.logits(torch.as_tensor(prompts), inputs), -1)
        token_logp = logp.gather(2, targets[..., None]).squeeze(2)
        mask = torch.arange(width)[None] < torch.as_tensor(lengths)[:, None]
        return (token_logp * mask).sum(1)

    @torch.no_grad()
    def sample(self, prompts, generator=None):
        """One response per prompt, up to MAX_LEN tokens or EOS."""
        batch = len(prompts)
        h = self.prompt(torch.as_tensor(prompts))[None]
        token = torch.full((batch, 1), START)
        responses = [[] for _ in range(batch)]
        alive = torch.ones(batch, dtype=torch.bool)
        for _ in range(MAX_LEN):
            out, h = self.gru(self.token(token), h)
            probs = torch.softmax(self.head(out[:, -1]), -1)
            token = torch.multinomial(probs, 1, generator=generator)
            for i in torch.nonzero(alive).flatten().tolist():
                responses[i].append(int(token[i]))
            alive &= token[:, 0] != EOS
            if not alive.any():
                break
        return responses


def sft(n_steps=2_000, batch=128, learning_rate=3e-3, seed=0):
    """Fit the policy to reference responses by maximum likelihood. Returns the model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = SequencePolicy()
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        prompts = rng.integers(N_PROMPTS, size=batch)
        responses = [reference_sample(rng, p) for p in prompts]
        loss = -policy.log_prob(prompts, responses).mean()
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    return policy


def frozen(policy):
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    return reference


def evaluate(policy, n=512, seed=123):
    """Mean true reward, mean length, and how often the response is the requested token only."""
    from envs.preference_toy import length, true_reward

    generator = torch.Generator().manual_seed(seed)
    prompts = np.random.default_rng(seed).integers(N_PROMPTS, size=n)
    responses = policy.sample(prompts, generator)
    rewards = [true_reward(p, r) for p, r in zip(prompts, responses)]
    lengths = [length(r) for r in responses]
    return dict(true_reward=float(np.mean(rewards)), length=float(np.mean(lengths)))
