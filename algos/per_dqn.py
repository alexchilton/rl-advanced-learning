"""Prioritized Experience Replay (Schaul, Quan, Antonoglou & Silver, 2015).

    python algos/per_dqn.py

Uniform replay spends most samples on transitions the network already predicts
well. PER samples transition i with probability

    P(i) = p_i^alpha / sum_k p_k^alpha,     p_i = |TD error_i| + eps

so surprising transitions are replayed more. That changes the distribution the
loss is averaged over, which biases the fixed point. Importance-sampling weights
undo it:

    w_i = (N * P(i)) ^ -beta,  normalised by max w,   beta annealed from 0.4 to 1

At beta = 1 the correction is exact, which matters at the end of training when the
estimate must converge to the right place. Early on a partial correction is
accepted for the speed-up.

The buffer is a sum tree: sampling and priority updates are O(log N), vectorised
over the batch. A plain `np.random.choice(p=...)` over 50,000 entries every step
works and is about a hundred times slower.

The marked lines are the difference from `algos/dqn.py`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE,
                       LEARNING_STARTS, TARGET_UPDATE, QNetwork, as_vector, epsilon_at,
                       evaluate_gym, greedy_action, value_vs_return)

ALPHA = 0.6
BETA_START = 0.4
PRIORITY_EPS = 1e-3


class PrioritizedReplayBuffer:
    """Ring buffer plus a sum tree over priorities^alpha."""

    def __init__(self, capacity, obs_dim, seed=0, alpha=ALPHA):
        self.capacity = 1
        while self.capacity < capacity:          # the tree wants a power of two
            self.capacity *= 2
        self.limit = int(capacity)
        self.alpha = alpha
        self.tree = np.zeros(2 * self.capacity)  # node i has children 2i, 2i+1; leaves at capacity..
        self.obs = np.zeros((self.limit, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.limit, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.limit, dtype=np.int64)
        self.rewards = np.zeros(self.limit, dtype=np.float32)
        self.terminated = np.zeros(self.limit, dtype=np.float32)
        self.size = self.position = 0
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def _set(self, index, priority):
        node = np.asarray(index) + self.capacity
        self.tree[node] = priority
        node //= 2
        while node[0] >= 1:
            # Duplicates in `node` all receive the same correct sum, so plain
            # assignment is safe here.
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            node //= 2

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        # New transitions get the highest priority seen so far, so each is
        # replayed at least once before its real TD error is known.
        self._set([i], self.max_priority ** self.alpha)
        self.position = (self.position + 1) % self.limit
        self.size = min(self.size + 1, self.limit)

    def sample(self, batch_size, beta):
        total = self.tree[1]
        # Stratified: one draw per equal slice of the total mass.
        targets = (np.arange(batch_size) + self.rng.random(batch_size)) * total / batch_size
        node = np.ones(batch_size, dtype=np.int64)
        while node[0] < self.capacity:
            left = 2 * node
            go_right = targets > self.tree[left]
            targets = np.where(go_right, targets - self.tree[left], targets)
            node = np.where(go_right, left + 1, left)
        index = np.minimum(node - self.capacity, self.size - 1)
        probability = self.tree[index + self.capacity] / total
        weights = (self.size * probability) ** -beta
        weights /= weights.max()
        return index, (torch.from_numpy(self.obs[index]), torch.from_numpy(self.actions[index]),
                       torch.from_numpy(self.rewards[index]), torch.from_numpy(self.next_obs[index]),
                       torch.from_numpy(self.terminated[index]),
                       torch.from_numpy(weights.astype(np.float32)))

    def update(self, index, td_errors):
        priority = np.abs(td_errors) + PRIORITY_EPS
        self.max_priority = max(self.max_priority, float(priority.max()))
        self._set(index, priority ** self.alpha)


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS,
          hidden=HIDDEN, seed=0, evaluate=None, eval_every=5_000, alpha=ALPHA, beta_start=BETA_START):
    """Same signature and log as `dqn.train`, plus `alpha` and `beta_start`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions

    net = QNetwork(len(obs), n_actions, hidden)
    target = QNetwork(len(obs), n_actions, hidden)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = PrioritizedReplayBuffer(buffer_size, len(obs), seed, alpha)   # ---- PER change 1
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
            beta = beta_start + (1.0 - beta_start) * step / n_steps
            index, (o, a, r, o2, done, w) = buffer.sample(batch_size, beta)
            with torch.no_grad():
                y = r + gamma * (1.0 - done) * target(o2).max(1).values
            q = net(o).gather(1, a[:, None]).squeeze(1)
            # ---- PER change 2: weight each sample's loss, then update its priority
            loss = (w * nn.functional.smooth_l1_loss(q, y, reduction="none")).mean()
            optimiser.zero_grad()
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP))
            optimiser.step()
            td = (q - y).detach()
            buffer.update(index, td.numpy())
            health = dict(q=float(q.detach().mean()), td=float(td.abs().mean()), grad=grad)

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
    print("CartPole-v1, DQN + prioritized replay, 100,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} greedy {row['score']:>6.1f}   mean Q {row['q']:>6.2f}   |TD| {row['td']:.3f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nnetwork predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Learning is fast early and then the values settle slightly wrong: beta never
#   reaches 1, so the importance correction is incomplete at convergence.
# - The same few transitions are replayed thousands of times: priorities were
#   never updated after sampling, or eps is 0 and a zero-error transition can
#   never be drawn again. Check `update` is called with the fresh TD errors.
# - New transitions are never seen: they were inserted with priority 0. They must
#   enter at the current max priority.
# - It is slower per step than DQN: it is. Judge it per environment step, not per
#   second, and report both.
