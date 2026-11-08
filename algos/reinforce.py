"""REINFORCE (Williams, 1992).

    python algos/reinforce.py

The policy-gradient theorem in its plainest form. Play a whole episode with the
current stochastic policy, then push up the log-probability of every action in
proportion to the return that followed it:

    grad J = E[ sum_t  grad log pi(a_t | s_t) * G_t ],     G_t = sum_{k>=t} gamma^(k-t) r_k

No value function, no bootstrapping, no replay. Every sample is used once and
thrown away, because the gradient is only correct for the policy that generated it.

The variance is the whole story. G_t is a sum of up to 500 random rewards, and its
spread dwarfs the differences between actions. `algos/reinforce_baseline.py`
subtracts a learned V(s_t) from G_t, which changes nothing in expectation and cuts
the variance; `experiments/pg_variance.py` measures by how much.

`returns_to_go` is the "reward-to-go" form: action t is credited only with rewards
after it. The original sums the whole episode's return for every action, which is
also unbiased (earlier rewards cannot depend on a later action) and strictly noisier.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

GAMMA = 0.99
LEARNING_RATE = 1e-3
HIDDEN = 64
EPISODES_PER_UPDATE = 8


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_actions))

    def distribution(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))


def returns_to_go(rewards, gamma):
    out, running = np.zeros(len(rewards), dtype=np.float32), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def play(env, policy, rng):
    """One episode. Returns (observations, actions, rewards)."""
    obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
    observations, actions, rewards = [], [], []
    done = False
    while not done:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            action = int(policy.distribution(obs_t[None]).sample().item())
        observations.append(obs_t)
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return torch.stack(observations), torch.tensor(actions), rewards


def train(env, total_steps=100_000, gamma=GAMMA, learning_rate=LEARNING_RATE,
          episodes_per_update=EPISODES_PER_UPDATE, hidden=HIDDEN, seed=0, evaluate=None, eval_every=10_000):
    """Returns (policy, log). Stops after `total_steps` environment steps, so it is
    comparable with A2C and PPO. `log` rows: step, score, mean training return, gradient norm."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    policy = Policy(obs_dim, env.action_space.n, hidden)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    log, batch, returns, grad = [], [], [], 0.0
    step, episode, next_eval = 0, 0, eval_every
    while step < total_steps:
        observations, actions, rewards = play(env, policy, rng)
        step += len(rewards)
        episode += 1
        batch.append((observations, actions, torch.as_tensor(returns_to_go(rewards, gamma))))
        returns.append(sum(rewards))

        if episode % episodes_per_update == 0:
            # ---- the REINFORCE loss: -log pi(a|s) * G_t, averaged over every step
            obs = torch.cat([b[0] for b in batch])
            act = torch.cat([b[1] for b in batch])
            g = torch.cat([b[2] for b in batch])
            loss = -(policy.distribution(obs).log_prob(act) * g).mean()
            optimiser.zero_grad()
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(policy.parameters(), 1e6))
            optimiser.step()
            batch = []

        if evaluate is not None and step >= next_eval:
            next_eval += eval_every
            log.append(dict(step=step, score=float(evaluate(policy)),
                            train_return=float(np.mean(returns[-20:])), grad=grad))
    return policy, log


def evaluate_gym(make_env, episodes=10, seed=10_000):
    """Mean return of the GREEDY (most probable) action over fresh episodes."""
    def score(policy):
        env, total = make_env(), 0.0
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                with torch.no_grad():
                    logits = policy.net(torch.as_tensor(np.asarray(obs, dtype=np.float32))[None])
                obs, reward, terminated, truncated, _ = env.step(int(logits.argmax()))
                total += reward
                done = terminated or truncated
        return total / episodes
    return score


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    _, log = train(make(), total_steps=100_000, evaluate=evaluate_gym(make), seed=0)
    print("CartPole-v1, REINFORCE, 100,000 steps, seed 0\n")
    print(f"{'step':>8} {'greedy':>8} {'train return':>13} {'grad norm':>10}")
    for row in log:
        print(f"{row['step']:>8,} {row['score']:>8.1f} {row['train_return']:>13.1f} {row['grad']:>10.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Learns, then collapses to a terrible deterministic policy: one huge-return
#   batch pushed the logits so far that the policy stopped exploring, and with no
#   exploration there is no gradient to bring it back. Smaller learning rate, more
#   episodes per update, or an entropy bonus.
# - Gradient norm swings by orders of magnitude between updates: that is the
#   variance of G_t, not a bug. A baseline is the fix (reinforce_baseline.py).
# - Learning stalls near CartPole's time limit: G_t for early steps is ~100 at
#   gamma=0.99 whatever the action, so the signal is a small difference on a large
#   number. Same fix.
# - Sign error: maximising the loss instead of minimising -log pi * G. The return
#   goes DOWN reliably, which is the tell; random code is not that consistent.
