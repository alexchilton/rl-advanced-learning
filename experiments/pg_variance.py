"""How much variance does each policy-gradient trick remove? Measured at a fixed policy.

    python experiments/pg_variance.py

Freeze one CartPole policy (REINFORCE trained for 20,000 steps, so it is neither
random nor solved). Draw many independent one-episode gradient estimates from it,
under four weightings of log pi(a_t | s_t):

    whole return     G_0 for every action             Williams' original form
    reward-to-go     G_t                              drop rewards before the action
    minus mean       G_t - mean(G)                    a constant baseline
    minus V(s_t)     G_t - V(s_t)                     a learned state baseline

All four have the same expectation. They differ only in spread. Two numbers per
estimator:

    total variance   trace of the covariance of the gradient vector
    cosine to mean   average cosine between one estimate and the mean of all of
                     them: how often a single gradient step points the right way

V is fitted beforehand by regression on 300 separate episodes from the same
policy, so it is as good as this policy's data allows.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos import reinforce

N_ESTIMATES = 300
GAMMA = reinforce.GAMMA


def flat_grad(policy, obs, actions, weights):
    policy.zero_grad()
    loss = -(policy.distribution(obs).log_prob(actions) * weights).sum()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()]).numpy().copy()


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    policy, _ = reinforce.train(env, total_steps=20_000, seed=0)
    rng = np.random.default_rng(1)

    def episodes(n):
        return [reinforce.play(env, policy, rng) for _ in range(n)]

    # Fit V on separate data from the same frozen policy.
    fit = episodes(N_ESTIMATES)
    obs = torch.cat([e[0] for e in fit])
    g = torch.cat([torch.as_tensor(reinforce.returns_to_go(e[2], GAMMA)) for e in fit])
    value = nn.Sequential(nn.Linear(obs.shape[1], 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(value.parameters(), lr=3e-3)
    for _ in range(2_000):
        loss = ((value(obs).squeeze(1) - g) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        explained = float(1 - (g - value(obs).squeeze(1)).var() / g.var())
    mean_g = float(g.mean())

    data = episodes(N_ESTIMATES)
    lengths = [len(e[2]) for e in data]
    estimators = {"whole return": [], "reward-to-go": [], "minus mean": [], "minus V(s_t)": []}
    for o, a, rewards in data:
        rtg = torch.as_tensor(reinforce.returns_to_go(rewards, GAMMA))
        with torch.no_grad():
            v = value(o).squeeze(1)
        estimators["whole return"].append(flat_grad(policy, o, a, torch.full_like(rtg, float(rtg[0]))))
        estimators["reward-to-go"].append(flat_grad(policy, o, a, rtg))
        estimators["minus mean"].append(flat_grad(policy, o, a, rtg - mean_g))
        estimators["minus V(s_t)"].append(flat_grad(policy, o, a, rtg - v))

    print(f"CartPole-v1, one REINFORCE policy frozen after 20,000 steps; {N_ESTIMATES} one-episode estimates each")
    print(f"episode length under this policy: median {np.median(lengths):.0f}, range {min(lengths)}-{max(lengths)}")
    print(f"V(s) fitted on {N_ESTIMATES} other episodes: explained variance {explained:.2f}\n")
    print(f"{'estimator':>14} {'total variance':>15} {'relative':>9} {'cosine to mean':>15}")
    reference = None
    for name, grads in estimators.items():
        grads = np.array(grads)
        mean = grads.mean(0)
        total_var = float(grads.var(0).sum())
        cosine = float(np.mean(grads @ mean / (np.linalg.norm(grads, axis=1) * np.linalg.norm(mean) + 1e-12)))
        reference = reference or total_var
        print(f"{name:>14} {total_var:>15.1f} {total_var / reference:>9.3f} {cosine:>15.3f}")


if __name__ == "__main__":
    main()
