"""REINFORCE with a learned baseline (Williams, 1992, section 4; Sutton & Barto 13.4).

    python algos/reinforce_baseline.py

REINFORCE's gradient weights each log-probability by G_t. Any function of the
state alone, b(s_t), can be subtracted without changing the gradient's expectation:

    E[ grad log pi(a|s) * b(s) ] = b(s) * grad sum_a pi(a|s) = b(s) * grad 1 = 0

So subtract the best one available, a learned V(s_t). What is left, G_t - V(s_t),
is positive for actions that did better than usual from that state and negative
for worse ones. That is the signal; the shared, large, action-independent part of
G_t was only noise.

The critic is fitted to G_t by regression, in its own network with its own
optimiser. It does not bootstrap: this is still a Monte Carlo method. Bootstrapping
the critic is the step to actor-critic (`algos/a2c.py`).

The marked lines are the difference from `algos/reinforce.py`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.reinforce import (EPISODES_PER_UPDATE, GAMMA, HIDDEN, LEARNING_RATE, Policy,
                             evaluate_gym, play, returns_to_go)

VALUE_LEARNING_RATE = 1e-3


def train(env, total_steps=100_000, gamma=GAMMA, learning_rate=LEARNING_RATE,
          episodes_per_update=EPISODES_PER_UPDATE, hidden=HIDDEN, seed=0, evaluate=None, eval_every=10_000,
          value_learning_rate=VALUE_LEARNING_RATE):
    """Same signature and log as `reinforce.train`, plus the critic's explained variance."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    policy = Policy(obs_dim, env.action_space.n, hidden)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    # ---- baseline change 1: a separate value network
    value = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                          nn.Linear(hidden, 1))
    value_optimiser = torch.optim.Adam(value.parameters(), lr=value_learning_rate)

    log, batch, returns, grad, explained = [], [], [], 0.0, 0.0
    step, episode, next_eval = 0, 0, eval_every
    while step < total_steps:
        observations, actions, rewards = play(env, policy, rng)
        step += len(rewards)
        episode += 1
        batch.append((observations, actions, torch.as_tensor(returns_to_go(rewards, gamma))))
        returns.append(sum(rewards))

        if episode % episodes_per_update == 0:
            obs = torch.cat([b[0] for b in batch])
            act = torch.cat([b[1] for b in batch])
            g = torch.cat([b[2] for b in batch])
            # ---- baseline change 2: weight by G_t - V(s_t), then fit V to G_t
            v = value(obs).squeeze(1)
            advantage = g - v.detach()
            loss = -(policy.distribution(obs).log_prob(act) * advantage).mean()
            optimiser.zero_grad()
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(policy.parameters(), 1e6))
            optimiser.step()

            value_loss = ((v - g) ** 2).mean()
            value_optimiser.zero_grad()
            value_loss.backward()
            value_optimiser.step()
            explained = float(1 - (g - v.detach()).var() / (g.var() + 1e-8))
            batch = []

        if evaluate is not None and step >= next_eval:
            next_eval += eval_every
            log.append(dict(step=step, score=float(evaluate(policy)),
                            train_return=float(np.mean(returns[-20:])), grad=grad,
                            explained_variance=explained))
    return policy, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    _, log = train(make(), total_steps=100_000, evaluate=evaluate_gym(make), seed=0)
    print("CartPole-v1, REINFORCE with a learned baseline, 100,000 steps, seed 0\n")
    print(f"{'step':>8} {'greedy':>8} {'train return':>13} {'grad norm':>10} {'critic EV':>10}")
    for row in log:
        print(f"{row['step']:>8,} {row['score']:>8.1f} {row['train_return']:>13.1f} "
              f"{row['grad']:>10.2f} {row['explained_variance']:>10.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - No improvement over plain REINFORCE: check the critic's explained variance.
#   Near 0 or negative means V is not tracking G and is subtracting noise.
# - The policy gradient flows into the critic: `advantage` must use v.detach(),
#   or the actor loss trains V to make advantages large.
# - The baseline depends on the action: then it is no longer unbiased. b may use
#   the state, never the action that was taken.
