"""DQN with PopArt target normalisation (van Hasselt, Guez, Hessel, Mnih & Silver, 2016).

    python algos/popart_dqn.py

Multiply every reward by 1,000 and nothing about the task changes: the optimal
policy is the same. But Q values are now ~1,000 times larger, so the network's
last layer must produce outputs of order 1e5 from hidden units of order 1, the
TD errors are huge, and the learning rate tuned for the original scale is wrong
for this one.

PopArt makes the network predict NORMALISED values and keeps running statistics
mu, sigma of the targets:

    Q(s, a) = sigma * f(s, a) + mu            f is the network's raw output
    loss    = huber( f(s, a) - (y - mu) / sigma )

"ART" (adaptive rescaling of targets): update mu and sigma from each batch of
targets. "POP" (preserving outputs precisely): when they change, rescale the
last layer so that sigma * f + mu is exactly what it was before,

    W <- W * sigma_old / sigma_new,      b <- (sigma_old * b + mu_old - mu_new) / sigma_new

so that changing the normalisation never changes what the agent believes.

DQN's own answer (Mnih et al. 2015) was to clip every reward to [-1, 1], which
keeps the scale fixed and changes the task whenever reward magnitudes matter.
`experiments/reward_normalisation.py` compares the three.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE, LEARNING_STARTS,
                       TARGET_UPDATE, QNetwork, ReplayBuffer, as_vector, epsilon_at, evaluate_gym)

STATS_RATE = 3e-4


class PopArtQ(nn.Module):
    """A Q network whose last layer is PopArt-normalised. `forward` returns unnormalised Q."""

    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = QNetwork(obs_dim, n_actions, hidden)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("second", torch.ones(1))   # running E[y^2]

    def normalised(self, obs):
        return self.net(obs)

    def forward(self, obs):
        return self.sigma * self.net(obs) + self.mu

    @torch.no_grad()
    def update_stats(self, targets, rate=STATS_RATE):
        """ART then POP: move the statistics towards this batch, then preserve outputs."""
        mu_old, sigma_old = self.mu.clone(), self.sigma.clone()
        self.mu.mul_(1 - rate).add_(rate * targets.mean())
        self.second.mul_(1 - rate).add_(rate * (targets ** 2).mean())
        self.sigma.copy_((self.second - self.mu ** 2).clamp(min=1e-4).sqrt())
        last = self.net.net[-1]
        last.weight.mul_(sigma_old / self.sigma)
        last.bias.copy_((sigma_old * last.bias + mu_old - self.mu) / self.sigma)


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS, hidden=HIDDEN,
          seed=0, evaluate=None, eval_every=5_000):
    """Same signature and log as `dqn.train`; the log also carries mu and sigma."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions
    net = PopArtQ(len(obs), n_actions, hidden)
    target = copy.deepcopy(net)
    optimiser = torch.optim.Adam(net.net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, len(obs), seed)

    log, health = [], dict(q=0.0, td=0.0, grad=0.0)
    for step in range(1, n_steps + 1):
        if rng.random() < epsilon_at(step, n_steps):
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(net(torch.from_numpy(obs)[None]).argmax(1).item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = as_vector(next_obs)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])

        if step >= learning_starts and buffer.size >= batch_size:
            o, a, r, o2, done = buffer.sample(batch_size)
            with torch.no_grad():
                y = r + gamma * (1.0 - done) * target(o2).max(1).values
            # ---- PopArt: update statistics (preserving outputs), then regress normalised values
            net.update_stats(y)
            f = net.normalised(o).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(f, (y - net.mu) / net.sigma)
            optimiser.zero_grad(); loss.backward()
            grad = float(nn.utils.clip_grad_norm_(net.net.parameters(), GRAD_CLIP)); optimiser.step()
            with torch.no_grad():
                q = net.sigma * f.detach() + net.mu
            health = dict(q=float(q.mean()), td=float((q - y).abs().mean()), grad=grad)
        if step % target_update == 0:
            target.load_state_dict(net.state_dict())
        if evaluate is not None and step % eval_every == 0:
            log.append(dict(step=step, score=float(evaluate(net)), mu=float(net.mu), sigma=float(net.sigma), **health))
    return net, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    _, log = train(make(), n_steps=50_000, evaluate=evaluate_gym(make), eval_every=10_000, seed=0)
    for row in log:
        print(f"{row['step']:>7,} greedy {row['score']:6.1f}  mean Q {row['q']:7.2f}  mu {row['mu']:6.2f}  sigma {row['sigma']:6.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Learning collapses each time the statistics move: the last layer is not
#   rescaled (ART without POP), so every stats update silently changes every Q.
# - The target network disagrees wildly with the online one after a sync: the
#   target must carry its own copy of mu and sigma (here: state_dict includes the
#   buffers), or its outputs are unnormalised with the wrong statistics.
# - sigma collapses to ~0 early: all early targets are equal (e.g. all zero),
#   so the variance is 0. Clamp it from below.
