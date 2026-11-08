"""Soft Actor-Critic (Haarnoja et al. 2018), with the automatic temperature of the
2018 follow-up ("SAC: Algorithms and Applications").

    python algos/sac.py

Maximise reward AND the policy's entropy:

    J = E[ sum_t r_t + alpha * H(pi(. | s_t)) ]

The policy is stochastic: a Gaussian whose sample is squashed through tanh to
stay inside the action bound. The critic learns soft Q values, whose target
includes the entropy of the next action:

    y = r + gamma * ( min_i Q_target_i(s', a') - alpha * log pi(a' | s') ),   a' ~ pi(. | s')

and the actor minimises  alpha * log pi(a | s) - min_i Q_i(s, a)  using the
reparameterisation trick, so the gradient flows through the sampled action.

**The tanh correction.** Squashing changes the density. If u ~ N(mu, sigma) and
a = tanh(u), then log pi(a) = log N(u) - sum log(1 - tanh(u)^2). Leave the second
term out and the entropy the agent thinks it has is wrong near the bounds, where
it matters most. It is the marked line in `TanhGaussian.sample`.

**alpha** sets how much randomness is worth. Too small and the policy goes
deterministic early and stops exploring; too large and it never commits. With
`alpha=None` it is learned to hold the entropy near target_entropy = -act_dim.

The twin critics with a min are TD3's clipped double Q.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from algos.ddpg import (BATCH_SIZE, BUFFER_SIZE, GAMMA, HIDDEN, START_STEPS, TAU, ReplayBuffer,
                        critic_input, evaluate_gym, mlp, soft_update, spaces)

LEARNING_RATE = 3e-4
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class TanhGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def sample(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        u = mean + std * torch.randn_like(mean)                      # reparameterised
        action = torch.tanh(u)
        log_prob = torch.distributions.Normal(mean, std).log_prob(u)
        # ---- the tanh correction, in the numerically stable form
        log_prob = log_prob - 2 * (np.log(2) - u - nn.functional.softplus(-2 * u))
        return action, log_prob.sum(-1)

    def deterministic(self, obs):
        mean, _ = self.net(obs).chunk(2, dim=-1)
        return torch.tanh(mean)


def train(env, total_steps=20_000, gamma=GAMMA, tau=TAU, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, hidden=HIDDEN, start_steps=START_STEPS, alpha=None, seed=0, evaluate=None,
          eval_every=2_000):
    """`alpha=None` learns the temperature; a number fixes it. Returns (policy, log).

    Log rows: step, score, mean Q, alpha, entropy estimate (-mean log pi).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim, act_dim, scale = spaces(env)
    policy = TanhGaussian(obs_dim, act_dim, hidden)
    critics = nn.ModuleList(mlp(obs_dim + act_dim, 1, hidden) for _ in range(2))
    critics_target = copy.deepcopy(critics)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=learning_rate)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=learning_rate)
    target_entropy = -float(act_dim)
    buffer = ReplayBuffer(buffer_size, obs_dim, act_dim, seed)

    def current_alpha():
        return log_alpha.exp().detach() if alpha is None else torch.tensor(float(alpha))

    obs = np.asarray(env.reset(seed=seed)[0], np.float32)
    log, health = [], dict(q=0.0, alpha=float(current_alpha()), entropy=0.0)
    for step in range(1, total_steps + 1):
        if step <= start_steps:
            action = rng.uniform(-1, 1, act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                action = policy.sample(torch.as_tensor(obs)[None])[0][0].numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action * scale)
        next_obs = np.asarray(next_obs, np.float32)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = np.asarray(env.reset(seed=int(rng.integers(2**31 - 1)))[0], np.float32)

        if step > start_steps:
            o, a, r, o2, done = buffer.sample(batch_size)
            a_alpha = current_alpha()
            with torch.no_grad():
                a2, logp2 = policy.sample(o2)
                target_q = torch.stack([c(critic_input(o2, a2)).squeeze(1) for c in critics_target]).min(0).values
                y = r + gamma * (1 - done) * (target_q - a_alpha * logp2)
            qs = [c(critic_input(o, a)).squeeze(1) for c in critics]
            critic_loss = sum(((q - y) ** 2).mean() for q in qs)
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            new_a, logp = policy.sample(o)
            q_new = torch.stack([c(critic_input(o, new_a)).squeeze(1) for c in critics]).min(0).values
            policy_loss = (a_alpha * logp - q_new).mean()
            policy_opt.zero_grad(); policy_loss.backward(); policy_opt.step()

            if alpha is None:
                alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

            soft_update(critics_target, critics, tau)
            health = dict(q=float(qs[0].detach().mean()), alpha=float(current_alpha()),
                          entropy=float(-logp.detach().mean()))

        if evaluate is not None and step % eval_every == 0:
            log.append(dict(step=step, score=float(evaluate(policy)), **health))
    return policy, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("Pendulum-v1")
    _, log = train(make(), total_steps=20_000, evaluate=evaluate_gym(make), seed=0)
    print("Pendulum-v1, SAC (learned alpha), 20,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} return {row['score']:>8.1f}   mean Q {row['q']:>8.2f}   "
              f"alpha {row['alpha']:.3f}   entropy {row['entropy']:+.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Entropy estimate drifts far below target_entropy while alpha keeps rising: the
#   tanh correction is missing, so log pi is wrong near the bounds and the
#   temperature chases a quantity the policy cannot actually reach.
# - Actions saturate at ±1 and stay there: the same bug, seen from the action side.
# - alpha collapses to ~0 in the first few thousand steps: target_entropy is set
#   too low (or with the wrong sign), so the policy is "already random enough".
# - The policy loss uses the critic of the BUFFER action: it must use a fresh
#   reparameterised sample, or no gradient reaches the policy through Q.
