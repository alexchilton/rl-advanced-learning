"""C51: distributional DQN (Bellemare, Dabney & Munos 2017; one of Rainbow's six parts).

    python algos/c51_dqn.py

DQN learns the expected return. C51 learns the whole distribution of the return,
as probabilities over a fixed grid of N_ATOMS values ("atoms") z_1..z_N between
V_MIN and V_MAX. Q is recovered as the mean, so acting is unchanged:

    Q(s, a) = sum_i p_i(s, a) z_i

The Bellman update moves a distribution, not a number. Each target atom
r + gamma * z_j lands between two grid points, so its probability is split
between them in proportion to distance (`project`). The loss is the cross-entropy
between that projected target and the predicted distribution.

Why it helps is still argued about. The mean is the same quantity DQN learns;
what changes is the training signal: 51 outputs per action trained by
cross-entropy instead of one by regression. Bellemare et al. show the
distribution itself is learned correctly; the notebook checks that against an
exact distribution on a coin-flip probe.

`forward` returns Q, so `greedy_action`, `evaluate_gym` and `value_vs_return`
from `algos/dqn.py` work unchanged. The target uses the target network's greedy
action, as in the paper (no Double DQN selection).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE, LEARNING_STARTS,
                       TARGET_UPDATE, ReplayBuffer, as_vector, epsilon_at, evaluate_gym, greedy_action,
                       value_vs_return)

N_ATOMS = 51
V_MIN, V_MAX = 0.0, 100.0  # CartPole: +1 per step, gamma 0.99, so discounted returns lie in [0, 100]


class CategoricalQ(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions * n_atoms))
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

    def logits(self, obs):
        return self.net(obs).view(-1, self.n_actions, self.n_atoms)

    def distribution(self, obs):
        return self.logits(obs).softmax(-1)

    def forward(self, obs):
        return (self.distribution(obs) * self.support).sum(-1)


def project(next_probs, rewards, done, gamma, support):
    """Distribute each atom of r + gamma * z over the two nearest grid points. Returns (batch, n_atoms)."""
    v_min, v_max, n = float(support[0]), float(support[-1]), len(support)
    dz = (v_max - v_min) / (n - 1)
    tz = (rewards[:, None] + gamma * (1.0 - done[:, None]) * support[None]).clamp(v_min, v_max)
    b = (tz - v_min) / dz
    lower, upper = b.floor().long(), b.ceil().long()
    # An atom that lands exactly on a grid point has lower == upper and both weights
    # below would be zero; move one index so the whole mass goes to that point.
    lower[(upper > 0) & (lower == upper)] -= 1
    upper[(lower < n - 1) & (lower == upper)] += 1
    target = torch.zeros_like(next_probs)
    target.scatter_add_(1, lower, next_probs * (upper.float() - b))
    target.scatter_add_(1, upper, next_probs * (b - lower.float()))
    return target


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS, hidden=HIDDEN,
          n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX, seed=0, evaluate=None, eval_every=5_000):
    """Same log as `dqn.train`; `td` is the cross-entropy loss instead of |TD error|."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions

    net = CategoricalQ(len(obs), n_actions, hidden, n_atoms, v_min, v_max)
    target = CategoricalQ(len(obs), n_actions, hidden, n_atoms, v_min, v_max)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, len(obs), seed)
    batch_size = min(batch_size, buffer_size)
    rows = torch.arange(batch_size)

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
            # ---- the C51 change: project the target distribution, cross-entropy loss
            with torch.no_grad():
                next_probs = target.distribution(o2)
                best = (next_probs * target.support).sum(-1).argmax(1)
                m = project(next_probs[rows, best], r, done, gamma, target.support)
            log_p = net.logits(o)[rows, a].log_softmax(-1)
            loss = -(m * log_p).sum(-1).mean()
            optimiser.zero_grad()
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP))
            optimiser.step()
            q = (log_p.detach().exp() * net.support).sum(-1)
            health = dict(q=float(q.mean()), td=float(loss.detach()), grad=grad)

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
    print("CartPole-v1, C51, 100,000 steps, seed 0 (max return 500)\n")
    print(f"{'step':>7} {'greedy return':>14} {'mean Q':>8} {'CE loss':>8} {'grad norm':>10}")
    for row in log:
        print(f"{row['step']:>7,} {row['score']:>14.1f} {row['q']:>8.2f} {row['td']:>8.3f} {row['grad']:>10.3f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nover greedy episodes: network predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Mean Q pinned at V_MAX or V_MIN: the support does not cover the returns, and
#   every target atom is clamped to the edge. Set V_MIN/V_MAX from the reward
#   scale and gamma (here 0 and 1 / (1 - 0.99) = 100).
# - Probability mass disappears (target rows sum to less than 1): atoms landing
#   exactly on a grid point, lower == upper, get weight zero. `project` moves one
#   index; `test_c51_projection_keeps_all_mass` checks the sums.
# - Loss decreases but the policy does not improve: the loss is taken over the
#   distribution of the wrong action (index with the stored action `a`, not argmax).
# - Terminal transitions still spread mass: the target support must collapse to
#   the single point r when done = 1.
