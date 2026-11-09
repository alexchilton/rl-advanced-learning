"""A reward model learned from pairwise preferences (Christiano et al. 2017; Stiennon et al. 2020).

    python algos/reward_model.py

People are bad at assigning a number to a response and good at saying which of
two is better. So collect comparisons, and fit a scalar r(prompt, response)
whose differences explain them under the Bradley-Terry model:

    P(y_w preferred over y_l) = sigmoid( r(x, y_w) - r(x, y_l) )
    loss = - log sigmoid( r(x, y_w) - r(x, y_l) )

The reward is only identified up to a constant per prompt, since only
differences enter the loss.

Accuracy is measured against the TRUE reward on fresh pairs, which real RLHF
cannot do: it is the fraction of pairs where the model orders the two responses
the same way the true reward does.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.sequence_policy import HIDDEN, START
from envs.preference_toy import EOS, N_PROMPTS, VOCAB, reference_sample, true_reward


class RewardModel(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.prompt = nn.Embedding(N_PROMPTS, hidden)
        self.token = nn.Embedding(VOCAB + 2, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, prompts, responses):
        """Scalar reward per (prompt, response). Reads the GRU state after the last real token."""
        lengths = [max(len(r), 1) for r in responses]
        width = max(lengths)
        tokens = torch.full((len(responses), width + 1), EOS, dtype=torch.long)
        tokens[:, 0] = START
        for i, r in enumerate(responses):
            if r:
                tokens[i, 1:len(r) + 1] = torch.as_tensor(r)
        out, _ = self.gru(self.token(tokens), self.prompt(torch.as_tensor(prompts))[None])
        last = out[torch.arange(len(responses)), torch.as_tensor(lengths)]
        return self.head(last).squeeze(-1)


def train(pairs, n_epochs=30, batch=128, learning_rate=3e-3, seed=0):
    """`pairs` is a list of (prompt, chosen, rejected). Returns the model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RewardModel()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(n_epochs):
        for index in np.array_split(rng.permutation(len(pairs)), max(len(pairs) // batch, 1)):
            p = [pairs[i][0] for i in index]
            margin = model(p, [pairs[i][1] for i in index]) - model(p, [pairs[i][2] for i in index])
            loss = -nn.functional.logsigmoid(margin).mean()
            optimiser.zero_grad(); loss.backward(); optimiser.step()
    return model


@torch.no_grad()
def accuracy(model, n=2_000, seed=99, sampler=None):
    """Fraction of fresh pairs ordered as the true reward orders them (ties skipped).

    `sampler(rng, prompt)` draws responses; default is the reference distribution.
    Pass a trained policy's sampler to measure accuracy off-distribution.
    """
    rng = np.random.default_rng(seed)
    sampler = sampler or reference_sample
    correct = total = 0
    for _ in range(n // 100):
        prompts = rng.integers(N_PROMPTS, size=100)
        y1 = [sampler(rng, p) for p in prompts]
        y2 = [sampler(rng, p) for p in prompts]
        predicted = (model(prompts, y1) - model(prompts, y2)).numpy()
        truth = np.array([true_reward(p, a) - true_reward(p, b) for p, a, b in zip(prompts, y1, y2)])
        keep = truth != 0
        correct += int((np.sign(predicted[keep]) == np.sign(truth[keep])).sum())
        total += int(keep.sum())
    return correct / total


def main():
    from envs.preference_toy import preference_data

    torch.set_num_threads(1)
    print("Reward-model accuracy against the true reward, on fresh reference pairs\n")
    print(f"{'pairs':>7} {'accuracy':>9}")
    for n_pairs in (100, 300, 1_000, 3_000, 10_000):
        data = preference_data(np.random.default_rng(0), n_pairs)
        print(f"{n_pairs:>7,} {accuracy(train(data)):>9.3f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - High accuracy on held-out pairs, and the policy still exploits it: the
#   held-out pairs come from the reference policy; the policy being optimised
#   produces responses the model has never seen. Measure accuracy on the
#   policy's own samples (`sampler=`), not only on the reference's.
# - The reward drifts by a constant between runs: identified only up to a
#   constant per prompt. Normalise before using it as a reward.
# - Accuracy stuck near 0.5: the labels are too noisy (temperature high) or the
#   model reads the wrong position (the last real token, not padding).
