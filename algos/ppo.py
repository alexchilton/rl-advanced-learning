"""Proximal Policy Optimization, clipped (Schulman et al. 2017). PyTorch, discrete actions.

    python algos/ppo.py

A2C takes one gradient step per batch and throws the batch away. That wastes
data, but taking more steps is unsafe: after the first step the batch no longer
comes from the current policy. PPO reuses each batch for several epochs and
corrects with the probability ratio

    r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)

then stops the policy from moving far by clipping it:

    L = E[ min( r_t A_t,  clip(r_t, 1 - eps, 1 + eps) A_t ) ]

Outside [1 - eps, 1 + eps] the objective is flat in the direction that would
increase the improvement, so its gradient is zero there. That is a soft trust
region (TRPO's KL constraint) costing one `clamp`.

The rollout and GAE are `algos/a2c.py`'s. The update loop below is the difference.
A JAX/Flax version with a health panel and timing comparisons against SB3 is in
`notebooks/jax_01_lunar_lander.ipynb`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.a2c import ActorCritic, Rollout, collect, evaluate_gym, gae

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 3e-4
N_STEPS = 1_024
EPOCHS = 10
MINIBATCH = 64
CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.0
HIDDEN = 64
MAX_GRAD = 0.5


def train(env, total_steps=100_000, gamma=GAMMA, lam=LAMBDA, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
          epochs=EPOCHS, minibatch=MINIBATCH, clip=CLIP, value_coef=VALUE_COEF, entropy_coef=ENTROPY_COEF,
          hidden=HIDDEN, seed=0, evaluate=None, eval_every=10_000):
    """Returns (model, log). Log rows: step, score, approx KL, clip fraction, entropy, explained variance."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    rollout = Rollout(env, seed)
    log, step, next_eval = [], 0, eval_every
    while step < total_steps:
        batch = collect(rollout, model, n_steps)
        step += n_steps
        advantages, returns = gae(batch, gamma, lam)

        # ---- the PPO update: several epochs of minibatches on the same batch
        kls, clipped = [], []
        for _ in range(epochs):
            for index in np.array_split(rng.permutation(n_steps), n_steps // minibatch):
                index = torch.as_tensor(index)
                adv = advantages[index]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                dist = model.distribution(batch["obs"][index])
                logp = dist.log_prob(batch["actions"][index])
                ratio = torch.exp(logp - batch["logp"][index])
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
                value_loss = ((model.value(batch["obs"][index]) - returns[index]) ** 2).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * dist.entropy().mean()
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                optimiser.step()
                with torch.no_grad():
                    kls.append(float(((ratio - 1) - torch.log(ratio)).mean()))
                    clipped.append(float(((ratio - 1).abs() > clip).float().mean()))

        if evaluate is not None and step >= next_eval:
            next_eval += eval_every
            with torch.no_grad():
                values = model.value(batch["obs"])
                entropy = float(model.distribution(batch["obs"]).entropy().mean())
            explained = float(1 - (returns - values).var() / (returns.var() + 1e-8))
            log.append(dict(step=step, score=float(evaluate(model)), approx_kl=float(np.mean(kls)),
                            clip_fraction=float(np.mean(clipped)), entropy=entropy, explained_variance=explained))
    return model, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    _, log = train(make(), total_steps=100_000, evaluate=evaluate_gym(make), seed=0)
    print("CartPole-v1, PPO (PyTorch), 100,000 steps, seed 0\n")
    print(f"{'step':>8} {'greedy':>8} {'approx KL':>10} {'clip frac':>10} {'entropy':>8} {'critic EV':>10}")
    for row in log:
        print(f"{row['step']:>8,} {row['score']:>8.1f} {row['approx_kl']:>10.4f} {row['clip_fraction']:>10.3f} "
              f"{row['entropy']:>8.3f} {row['explained_variance']:>10.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Clip fraction ~0 and approx KL ~1e-4: the policy barely moves per update, so
#   the clip never binds and PPO is just slow A2C. More epochs or a larger
#   learning rate. The JAX notebook measures exactly this (32 vs 256 steps per batch).
# - Clip fraction > 0.3 and KL climbing: updates are too large; the clip is the
#   only thing holding the policy together. Fewer epochs or a smaller learning rate.
# - Ratio not 1 on the first minibatch of the first epoch: `logp` was stored from a
#   different network or after an update. The old log-probabilities must come from
#   the policy that collected the batch.
# - Advantage normalised over the whole batch vs per minibatch: both are common;
#   per-minibatch is used here. Not normalising at all makes the learning rate
#   depend on the reward scale.
