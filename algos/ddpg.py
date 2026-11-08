"""DDPG (Lillicrap et al. 2015).

    python algos/ddpg.py

DQN needs max_a Q(s, a). With a continuous action that max is an optimisation
problem at every step. DDPG learns a deterministic actor mu(s) that approximates
the argmax, and trains it by following the critic's gradient with respect to the
action:

    critic   Q(s, a)  ->  r + gamma * Q_target(s', mu_target(s'))
    actor    maximise Q(s, mu(s)),   i.e.  grad_theta = grad_a Q(s, a) * grad_theta mu(s)

Everything else is DQN's: a replay buffer, target networks (updated softly,
theta_target <- tau theta + (1 - tau) theta_target), and exploration by adding
Gaussian noise to the actor's action.

The actor maximises the critic, so any error in the critic that looks like a
peak is exactly what the actor will find. That is the failure TD3
(`algos/td3.py`) is about.

`ReplayBuffer`, `mlp` and the evaluation helpers here are shared with td3.py and
sac.py; each of those files contains its own update.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
HIDDEN = 256
START_STEPS = 1_000
EXPLORATION_NOISE = 0.1


def mlp(inp, out, hidden=HIDDEN, final=None):
    layers = [nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out)]
    return nn.Sequential(*layers, final) if final is not None else nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, seed=0):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros((capacity, act_dim), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i], self.next_obs[i], self.terminated[i] = (
            obs, action, reward, next_obs, terminated)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        i = self.rng.integers(0, self.size, n)
        return tuple(torch.as_tensor(x[i]) for x in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.terminated))


def soft_update(target, source, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(tau * s)


def spaces(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    scale = float(env.action_space.high[0])
    return obs_dim, act_dim, scale


def critic_input(obs, action):
    return torch.cat([obs, action], dim=-1)


def train(env, total_steps=20_000, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
          buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, hidden=HIDDEN, start_steps=START_STEPS,
          exploration_noise=EXPLORATION_NOISE, seed=0, evaluate=None, eval_every=2_000):
    """Returns (actor, log). The actor outputs actions in [-1, 1]; they are scaled to the env's bound.

    Log rows: step, score, mean Q on the batch, critic loss.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim, act_dim, scale = spaces(env)
    actor = mlp(obs_dim, act_dim, hidden, nn.Tanh())
    critic = mlp(obs_dim + act_dim, 1, hidden)
    actor_target, critic_target = copy.deepcopy(actor), copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    buffer = ReplayBuffer(buffer_size, obs_dim, act_dim, seed)

    obs = np.asarray(env.reset(seed=seed)[0], np.float32)
    log, health = [], dict(q=0.0, critic_loss=0.0)
    for step in range(1, total_steps + 1):
        if step <= start_steps:
            action = rng.uniform(-1, 1, act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor(torch.as_tensor(obs)[None])[0].numpy()
            action = np.clip(action + exploration_noise * rng.standard_normal(act_dim), -1, 1).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action * scale)
        next_obs = np.asarray(next_obs, np.float32)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = np.asarray(env.reset(seed=int(rng.integers(2**31 - 1)))[0], np.float32)

        if step > start_steps:
            o, a, r, o2, done = buffer.sample(batch_size)
            with torch.no_grad():
                y = r + gamma * (1 - done) * critic_target(critic_input(o2, actor_target(o2))).squeeze(1)
            q = critic(critic_input(o, a)).squeeze(1)
            critic_loss = ((q - y) ** 2).mean()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            # ---- the DDPG actor: climb the critic's gradient with respect to the action
            actor_loss = -critic(critic_input(o, actor(o))).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            soft_update(critic_target, critic, tau)
            soft_update(actor_target, actor, tau)
            health = dict(q=float(q.detach().mean()), critic_loss=float(critic_loss.detach()))

        if evaluate is not None and step % eval_every == 0:
            log.append(dict(step=step, score=float(evaluate(actor)), **health))
    return actor, log


def evaluate_gym(make_env, episodes=5, seed=10_000):
    """Mean return of the deterministic actor (no exploration noise) over fresh episodes.

    Works for DDPG and TD3 actors (a network ending in tanh) and for SAC's policy,
    whose `deterministic` method returns tanh of the mean.
    """
    def score(actor):
        env, total = make_env(), 0.0
        scale = float(env.action_space.high[0])
        act = getattr(actor, "deterministic", actor)
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                with torch.no_grad():
                    action = act(torch.as_tensor(np.asarray(obs, np.float32))[None])[0].numpy()
                obs, reward, terminated, truncated, _ = env.step(action * scale)
                total += reward
                done = terminated or truncated
        return total / episodes
    return score


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("Pendulum-v1")
    _, log = train(make(), total_steps=20_000, evaluate=evaluate_gym(make), seed=0)
    print("Pendulum-v1, DDPG, 20,000 steps, seed 0 (about -150 is solved; random is about -1,200)\n")
    for row in log:
        print(f"{row['step']:>7,} return {row['score']:>8.1f}   mean Q {row['q']:>8.2f}   critic loss {row['critic_loss']:.3f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Mean Q climbs far above any achievable return while the score stalls: the
#   actor is exploiting critic overestimation. That is TD3's motivating failure.
# - The actor's gradient flows into the critic: `actor_loss.backward()` puts
#   gradients on the critic's parameters too. They are cleared by the next
#   `critic_opt.zero_grad()` here; skip that and the critic is trained to please
#   the actor.
# - Actions pinned at ±1: the tanh saturated. Usually a critic whose gradient with
#   respect to the action never changes sign; check the critic first.
# - Learns on Pendulum, nothing on a sparse task: Gaussian action noise is weak
#   exploration. That is a problem for exploration methods, not for DDPG.
