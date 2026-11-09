"""Direct Preference Optimization (Rafailov et al. 2023).

    python algos/dpo.py

The KL-regularised RLHF objective has a closed-form optimum:

    pi*(y | x) = pi_ref(y | x) exp(r(x, y) / beta) / Z(x)

Solve it for r and the reward is a function of the policy itself:
r(x, y) = beta log(pi(y|x) / pi_ref(y|x)) + beta log Z(x). Put that into the
Bradley-Terry likelihood and Z(x) cancels, because only differences matter:

    loss = - log sigmoid( beta * [ log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l) ] )

No reward model, no sampling, no RL loop: supervised training on the preference
pairs. "Your language model is secretly a reward model."

What it gives up: it never sees responses outside the preference data. RLHF
samples from the current policy and scores them; DPO only pushes probability
between the pairs it was given.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.sequence_policy import evaluate, frozen

BETA = 0.1
LEARNING_RATE = 1e-3


def train(policy, pairs, n_epochs=3, beta=BETA, batch=64, learning_rate=LEARNING_RATE, seed=0, eval_every=50):
    """Fine-tune `policy` in place on `pairs` = [(prompt, chosen, rejected), ...].

    Returns a log: update, true reward, length, implicit-reward accuracy on the batch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reference = frozen(policy)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    log, update = [], 0
    for _ in range(n_epochs):
        for index in np.array_split(rng.permutation(len(pairs)), max(len(pairs) // batch, 1)):
            prompts = [pairs[i][0] for i in index]
            chosen = [pairs[i][1] for i in index]
            rejected = [pairs[i][2] for i in index]
            with torch.no_grad():
                ref_w, ref_l = reference.log_prob(prompts, chosen), reference.log_prob(prompts, rejected)
            # ---- the DPO loss
            margin = beta * ((policy.log_prob(prompts, chosen) - ref_w) - (policy.log_prob(prompts, rejected) - ref_l))
            loss = -nn.functional.logsigmoid(margin).mean()
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            update += 1
            if update % eval_every == 0:
                row = evaluate(policy)
                row.update(update=update, implicit_accuracy=float((margin > 0).float().mean()))
                log.append(row)
    return log


def main():
    from algos.sequence_policy import sft
    from envs.preference_toy import preference_data

    torch.set_num_threads(1)
    data = preference_data(np.random.default_rng(0), 3_000)
    print(f"SFT policy: {evaluate(sft())}")
    for beta in (0.03, 0.1, 0.3):
        log = train(sft(), data, beta=beta)
        print(f"beta {beta}: {log[-1]}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The chosen responses' likelihood goes DOWN during training: common and not a
#   bug. DPO only needs the margin to grow, and it can grow by pushing both down,
#   rejected faster. Watch the true reward, not log pi(y_w).
# - The reference model is the policy being trained: then every log-ratio is 0
#   and the gradient is the plain Bradley-Terry one on a policy that never moves
#   relative to itself. The reference must be frozen at the start.
# - Works on the training pairs, generalises poorly: DPO is supervised on fixed
#   pairs. Nothing pushes probability onto good responses nobody compared.
