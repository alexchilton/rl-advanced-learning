"""n-step DQN (the n-step return in Rainbow; Sutton 1988; Hessel et al. 2017).

    python algos/n_step_dqn.py

One-step DQN learns from r + gamma * max Q(s'). Reward information therefore
travels back one state per update, through a value estimate that starts wrong.
The n-step target uses n real rewards before bootstrapping:

    y = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n max_a Q_target(s_{t+n}, a)

More real reward, less reliance on the estimate: less bias, more variance, and
reward propagates n states per update instead of one. n = 1 is DQN; n -> episode
length is Monte Carlo. It is the same dial as GAE's lambda.

The returns are computed off-policy without correction: the n rewards came from
the epsilon-greedy policy, not the greedy one. Rainbow does the same and gets
away with it for small n; for large n it is a real bias.

The marked lines are the difference from `algos/dqn.py`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE,
                       LEARNING_STARTS, TARGET_UPDATE, QNetwork, ReplayBuffer, as_vector,
                       epsilon_at, evaluate_gym, greedy_action, value_vs_return)

N = 3


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS,
          hidden=HIDDEN, seed=0, evaluate=None, eval_every=5_000, n=N):
    """Same signature and log as `dqn.train`, plus `n`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions

    net = QNetwork(len(obs), n_actions, hidden)
    target = QNetwork(len(obs), n_actions, hidden)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, len(obs), seed)
    batch_size = min(batch_size, buffer_size)
    pending = deque()      # ---- n-step change 1: the last n transitions of this episode

    def flush(final_obs, terminated, everything):
        """Turn pending transitions into n-step ones. Stored reward is the discounted
        sum; stored 'next obs' is n steps on; the bootstrap discount gamma^k is
        applied at training time via the `k` folded into the terminated column."""
        while pending and (everything or len(pending) == n):
            ret = sum(gamma ** i * t[2] for i, t in enumerate(pending))
            k = len(pending)
            first_obs, first_action = pending[0][0], pending[0][1]
            # Encode "terminated" as 1.0 and otherwise the bootstrap discount gamma^k.
            buffer.add(first_obs, first_action, ret, final_obs, 1.0 if terminated else -(gamma ** k))
            pending.popleft()
            if not everything:
                break

    log, health = [], dict(q=0.0, td=0.0, grad=0.0)
    for step in range(1, n_steps + 1):
        if rng.random() < epsilon_at(step, n_steps):
            action = int(rng.integers(n_actions))
        else:
            action = greedy_action(net, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = as_vector(next_obs)
        pending.append((obs, action, reward))
        flush(next_obs, terminated, everything=terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])

        if step >= min(learning_starts, buffer_size) and buffer.size >= batch_size:
            o, a, r, o2, code = buffer.sample(batch_size)
            with torch.no_grad():
                # ---- n-step change 2: bootstrap with gamma^k, or not at all if terminated
                discount = torch.where(code > 0, torch.zeros_like(code), -code)
                y = r + discount * target(o2).max(1).values
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
    print(f"CartPole-v1, {N}-step DQN, 100,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} greedy {row['score']:>6.1f}   mean Q {row['q']:>6.2f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nnetwork predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Values are systematically too high or low by a factor near gamma^(n-1): the
#   bootstrap is discounted by gamma instead of gamma^k for the k steps actually
#   summed. Short tails at episode end have k < n.
# - The n-step window runs across an episode boundary: rewards from the next
#   episode leak into this one's return. `pending` must be flushed on every
#   termination AND truncation.
# - Truncated windows are treated as terminal: they still bootstrap from the
#   final observation. Only `terminated` cuts the bootstrap.
# - Large n gets worse, not better: the off-policy bias. The n rewards were
#   collected by the exploring policy, not the greedy one being evaluated.
