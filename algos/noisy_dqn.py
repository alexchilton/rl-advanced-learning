"""Noisy networks for exploration (Fortunato et al. 2017; one of Rainbow's six parts).

    python algos/noisy_dqn.py

Epsilon-greedy explores by acting at random on a schedule that knows nothing
about the task. A noisy network puts learnable noise in its weights instead:

    w = mu + sigma * eps,   eps ~ N(0, 1) resampled every step

so the greedy action of the CURRENT sample is already exploratory, and the
network learns how much noise to keep: gradient descent can shrink sigma where
it costs return and leave it where it does not. Epsilon is zero here.

Factorised Gaussian noise (the paper's cheaper variant): eps_ij = f(e_i) f(e_j)
with f(x) = sign(x) sqrt|x|, so a layer needs in + out random numbers, not in * out.
The last two layers are noisy; the first is plain (the paper makes every fully
connected layer noisy).

When the acting noise is resampled is `resample`:

    "step"     a new weight sample before every action, as in the paper's DQN agent
    "episode"  one sample held for the whole episode, so the perturbed policy is
               consistent: the same state gets the same perturbed action until the
               episode ends (parameter-space noise in the sense of Plappert et al.
               2017). The learning update always draws fresh noise.

Evaluation uses the mean weights (`net.eval()`), so the score is the policy the
noise is centred on. The health log records the mean |sigma|: if it falls to
zero, exploration has stopped, whatever the schedule would have said.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE, LEARNING_STARTS,
                       TARGET_UPDATE, ReplayBuffer, as_vector, evaluate_gym, greedy_action, value_vs_return)

SIGMA_0 = 0.5


class NoisyLinear(nn.Module):
    def __init__(self, n_in, n_out, sigma_0=SIGMA_0):
        super().__init__()
        bound = 1 / math.sqrt(n_in)
        self.weight_mu = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        self.bias_mu = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((n_out, n_in), sigma_0 * bound))
        self.bias_sigma = nn.Parameter(torch.full((n_out,), sigma_0 * bound))
        self.register_buffer("eps_in", torch.zeros(n_in))
        self.register_buffer("eps_out", torch.zeros(n_out))
        self.reset_noise()

    def reset_noise(self):
        f = lambda x: x.sign() * x.abs().sqrt()
        self.eps_in.copy_(f(torch.randn_like(self.eps_in)))
        self.eps_out.copy_(f(torch.randn_like(self.eps_out)))

    def forward(self, x):
        if not self.training:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * torch.outer(self.eps_out, self.eps_in)
        return nn.functional.linear(x, weight, self.bias_mu + self.bias_sigma * self.eps_out)


class NoisyQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN, sigma_0=SIGMA_0):
        super().__init__()
        self.noisy = nn.ModuleList([NoisyLinear(hidden, hidden, sigma_0), NoisyLinear(hidden, n_actions, sigma_0)])
        self.first = nn.Linear(obs_dim, hidden)

    def forward(self, obs):
        h = torch.relu(self.first(obs))
        return self.noisy[1](torch.relu(self.noisy[0](h)))

    def reset_noise(self):
        for layer in self.noisy:
            layer.reset_noise()

    def get_noise(self):
        return [(layer.eps_in.clone(), layer.eps_out.clone()) for layer in self.noisy]

    def set_noise(self, noise):
        for layer, (eps_in, eps_out) in zip(self.noisy, noise):
            layer.eps_in.copy_(eps_in); layer.eps_out.copy_(eps_out)

    def mean_sigma(self):
        return float(torch.cat([layer.weight_sigma.detach().abs().reshape(-1) for layer in self.noisy]).mean())


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS, hidden=HIDDEN,
          sigma_0=SIGMA_0, resample="step", seed=0, evaluate=None, eval_every=5_000):
    """Same log as `dqn.train` plus `sigma`, the mean |weight sigma| of the noisy layers."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions

    net = NoisyQNetwork(len(obs), n_actions, hidden, sigma_0)
    target = NoisyQNetwork(len(obs), n_actions, hidden, sigma_0)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, len(obs), seed)
    batch_size = min(batch_size, buffer_size)

    log, health = [], dict(q=0.0, td=0.0, grad=0.0)
    net.reset_noise()
    acting_noise = net.get_noise()
    for step in range(1, n_steps + 1):
        # ---- the Noisy change: no epsilon; a weight sample picks the action
        if resample == "step":
            net.reset_noise()
            acting_noise = net.get_noise()
        net.set_noise(acting_noise)
        action = greedy_action(net, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = as_vector(next_obs)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])
            net.reset_noise()
            acting_noise = net.get_noise()

        if step >= min(learning_starts, buffer_size) and buffer.size >= batch_size:
            o, a, r, o2, done = buffer.sample(batch_size)
            net.reset_noise(); target.reset_noise()
            with torch.no_grad():
                y = r + gamma * (1.0 - done) * target(o2).max(1).values
            q = net(o).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            optimiser.zero_grad()
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP))
            optimiser.step()
            health = dict(q=float(q.detach().mean()), td=float((q - y).detach().abs().mean()), grad=grad)

        if step % target_update == 0:
            target.load_state_dict(net.state_dict())
        if evaluate is not None and step % eval_every == 0:
            net.eval()
            score = float(evaluate(net))
            net.train()
            log.append(dict(step=step, score=score, sigma=net.mean_sigma(), **health))
    net.eval()
    return net, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    net, log = train(make(), n_steps=100_000, evaluate=evaluate_gym(make), eval_every=10_000, seed=0)
    print("CartPole-v1, Noisy DQN, 100,000 steps, seed 0 (max return 500)\n")
    print(f"{'step':>7} {'greedy return':>14} {'mean Q':>8} {'|TD|':>7} {'mean sigma':>11}")
    for row in log:
        print(f"{row['step']:>7,} {row['score']:>14.1f} {row['q']:>8.2f} {row['td']:>7.3f} {row['sigma']:>11.4f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nover greedy episodes: network predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Explores no more than a plain greedy DQN: noise is sampled once at
#   construction and never again. `reset_noise` must run every step.
# - Evaluation score jumps around between identical checkpoints: evaluating in
#   training mode, with noise. Call net.eval() first.
# - Mean sigma grows without bound: noise in the target network but none in the
#   online one, or sigma excluded from the optimiser. Log it.
# - Epsilon-greedy added on top: then the noise is not what explores, and a
#   comparison with DQN no longer measures it.
