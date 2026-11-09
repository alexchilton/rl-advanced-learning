"""RLHF: policy-gradient fine-tuning against a learned reward, with a KL leash.

    python algos/rlhf.py

Stiennon et al. 2020 and Ouyang et al. 2022 (InstructGPT) optimise

    max_pi  E_{y ~ pi}[ r_RM(x, y) ]  -  beta * KL( pi(. | x) || pi_ref(. | x) )

The reward model is only trustworthy near the data it was fitted on, which came
from the reference policy. The KL term keeps the policy near there. Without it
the policy finds whatever the reward model scores highest, which is usually not
what people wanted (Gao et al. 2022 measure exactly how fast that happens).

Three ways to turn sampled rewards into advantages, one argument:

    estimator="mean"  REINFORCE with the batch-mean baseline
    estimator="rloo"  REINFORCE leave-one-out (Ahmadian et al. 2024): k samples per
                      prompt, each one's baseline is the mean of the other k-1. Unbiased,
                      no value network.
    estimator="grpo"  Group Relative Policy Optimization (Shao et al. 2024): within each
                      prompt's group, (r - mean) / std. The KL is a separate loss term with
                      the k3 estimator rather than part of the reward.

PPO's clipped ratio is left out on purpose: each batch is used for one gradient
step, so the ratio is exactly 1 and the clip never acts. Ahmadian et al. argue
that in this regime the clip and the value network are dead weight; RLOO is that
argument in code.
"""

from __future__ import annotations

import numpy as np
import torch

from algos.sequence_policy import evaluate, frozen
from envs.preference_toy import N_PROMPTS, length, true_reward

BETA = 0.05
LEARNING_RATE = 1e-3
PROMPTS_PER_BATCH = 32
K = 4


def train(policy, reward_fn, n_steps=300, beta=BETA, estimator="rloo", k=K, prompts_per_batch=PROMPTS_PER_BATCH,
          learning_rate=LEARNING_RATE, seed=0, eval_every=25):
    """Fine-tune `policy` in place. `reward_fn(prompts, responses) -> tensor` is the reward
    (a reward model, or the true reward for an upper bound).

    Returns a log: step, proxy reward, true reward, KL to the reference, length.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reference = frozen(policy)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    log = []
    for step in range(1, n_steps + 1):
        prompts = np.repeat(rng.integers(N_PROMPTS, size=prompts_per_batch), k)
        responses = policy.sample(prompts)
        logp = policy.log_prob(prompts, responses)
        with torch.no_grad():
            ref_logp = reference.log_prob(prompts, responses)
            proxy = reward_fn(prompts, responses)
            kl = logp.detach() - ref_logp                       # per-sample log-ratio
            grouped = proxy.view(prompts_per_batch, k)
            if estimator == "grpo":
                advantage = ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-6)).flatten()
            else:
                total = (proxy - beta * kl).view(prompts_per_batch, k)
                if estimator == "rloo":
                    baseline = (total.sum(1, keepdim=True) - total) / (k - 1)
                else:
                    baseline = total.mean()
                advantage = (total - baseline).flatten()
        loss = -(advantage * logp).mean()
        if estimator == "grpo":
            # k3 estimator of KL(pi || ref), per sample, as its own term
            log_ratio = ref_logp - logp
            loss = loss + beta * (torch.exp(log_ratio) - log_ratio - 1).mean()
        optimiser.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimiser.step()

        if step % eval_every == 0:
            row = evaluate(policy)
            row.update(step=step, proxy=float(proxy.mean()), kl=float(kl.mean()))
            log.append(row)
    return log


def true_reward_fn(prompts, responses):
    return torch.tensor([true_reward(p, r) for p, r in zip(prompts, responses)], dtype=torch.float32)


def main():
    from algos import reward_model
    from algos.sequence_policy import sft
    from envs.preference_toy import preference_data

    torch.set_num_threads(1)
    rm = reward_model.train(preference_data(np.random.default_rng(0), 3_000))
    with torch.no_grad():
        rm_fn = lambda p, r: rm(p, r)
    print(f"SFT policy: {evaluate(sft())}")
    for estimator in ("mean", "rloo", "grpo"):
        log = train(sft(), rm_fn, estimator=estimator, n_steps=300)
        last = log[-1]
        print(f"{estimator:>5}: true reward {last['true_reward']:6.2f}, proxy {last['proxy']:6.2f}, "
              f"KL {last['kl']:5.2f}, length {last['length']:5.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Proxy reward climbs, true reward (or human rating) falls: reward-model
#   over-optimisation. Raise beta, or stop earlier; the KL is the x-axis on which
#   this happens (Gao et al. 2022).
# - KL goes negative: it is a sample estimate of a non-negative quantity, and the
#   plain log-ratio can be negative on a batch. The k3 estimator cannot.
# - GRPO with one sample per prompt: the group std is undefined and every
#   advantage is 0. k must be at least 2.
# - Responses get longer every update: the reward model prefers length. Measure
#   length alongside reward (Singhal et al. 2023).
