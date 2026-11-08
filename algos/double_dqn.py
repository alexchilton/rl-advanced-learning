"""Double DQN (van Hasselt, Guez & Silver, 2015).

    python algos/double_dqn.py

DQN's target takes a max over noisy estimates:

    DQN         y = r + gamma * Q_target(s', argmax_a' Q_target(s', a'))
    Double DQN  y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

The same network both picks the best-looking action and scores it, so whichever
action is most over-estimated wins, and the target inherits its error. The max of
noisy estimates is biased upwards even when every estimate is unbiased.

Double DQN lets the online network choose and the target network evaluate. Their
errors are partly independent, so the bias shrinks. It is the tabular Double
Q-learning idea (`algos/double_q.py`) using the two networks DQN already has.

One line differs from `algos/dqn.py`; it is marked below. Shared helpers that are
not part of the algorithm (network, buffer, evaluation) are imported from there.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE,
                       LEARNING_STARTS, TARGET_UPDATE, QNetwork, ReplayBuffer, as_vector,
                       epsilon_at, evaluate_gym, greedy_action, value_vs_return)


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS,
          hidden=HIDDEN, seed=0, evaluate=None, eval_every=5_000):
    """Same signature and log as `dqn.train`."""
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
                # ---- the Double DQN change: online net selects, target net evaluates
                chosen = net(o2).argmax(1, keepdim=True)
                bootstrap = target(o2).gather(1, chosen).squeeze(1)
                y = r + gamma * (1.0 - done) * bootstrap
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
    print("CartPole-v1, Double DQN, 100,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} greedy {row['score']:>6.1f}   mean Q {row['q']:>6.2f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nnetwork predicts {predicted:.2f}, discounted return actually {actual:.2f}")
    print("Compare with python algos/dqn.py. experiments/deep_value_compare.py does it over seeds.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - No difference from DQN: on CartPole the true values are bounded by the time
#   limit and rewards are deterministic, so there is little noise for the max to
#   exploit. Overestimation needs noise; measure predicted vs actual
#   (`value_vs_return`) rather than the score.
# - Both networks select and evaluate with the target net: that is plain DQN with
#   extra steps. The argmax must come from the ONLINE network.
