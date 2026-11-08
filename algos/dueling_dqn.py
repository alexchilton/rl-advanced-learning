"""Dueling DQN (Wang et al. 2016).

    python algos/dueling_dqn.py

Only the network changes. Instead of one head producing Q(s, .), two heads:

    V(s)       how good the state is, whatever you do
    A(s, a)    how much better action a is than average, here

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

In many states the action barely matters (CartPole's pole upright and centred).
A plain Q head must learn the same value separately for every action. The dueling
head learns it once in V and spends A only on the differences.

The `- mean A` is not decoration. Without it V and A are not identifiable: add a
constant to V and subtract it from A and Q is unchanged, so neither head means
anything and training drifts. Subtracting the mean pins A to average zero.

Training is `algos/dqn.py` unchanged; the marked line is where the network is built.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE,
                       LEARNING_STARTS, TARGET_UPDATE, ReplayBuffer, as_vector, epsilon_at,
                       evaluate_gym, greedy_action, value_vs_return)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        features = self.body(obs)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS,
          hidden=HIDDEN, seed=0, evaluate=None, eval_every=5_000):
    """Same signature and log as `dqn.train`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions

    # ---- the Dueling change: the network, nothing else
    net = DuelingQNetwork(len(obs), n_actions, hidden)
    target = DuelingQNetwork(len(obs), n_actions, hidden)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, len(obs), seed)
    batch_size = min(batch_size, buffer_size)

    log, health = [], dict(q=0.0, td=0.0, grad=0.0)
    for step in range(1, n_steps + 1):
        if rng.random() < epsilon_at(step, n_steps):
            action = int(rng.integers(n_actions))
        else:
            action = greedy_action(net, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = as_vector(next_obs)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])

        if step >= min(learning_starts, buffer_size) and buffer.size >= batch_size:
            o, a, r, o2, done = buffer.sample(batch_size)
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
            log.append(dict(step=step, score=float(evaluate(net)), **health))

    return net, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    net, log = train(make(), n_steps=100_000, evaluate=evaluate_gym(make), eval_every=10_000, seed=0)
    print("CartPole-v1, Dueling DQN, 100,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} greedy {row['score']:>6.1f}   mean Q {row['q']:>6.2f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nnetwork predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - V and A drift apart in magnitude while Q looks fine: the mean subtraction is
#   missing, so the split between the heads is arbitrary.
# - Subtracting the max instead of the mean: the paper's first form. It also
#   identifies the heads, but the advantage target jumps whenever the argmax
#   changes; the mean form is the one the paper settled on for stability.
# - No gain on CartPole: two actions leave little to share. The benefit grows with
#   the number of actions that do not matter in a given state.
