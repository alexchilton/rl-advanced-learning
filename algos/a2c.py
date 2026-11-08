"""Advantage actor-critic, synchronous (A2C; the synchronous form of Mnih et al. 2016, A3C).

    python algos/a2c.py

REINFORCE with a baseline waits for the episode to end and uses the full return
G_t. A2C bootstraps instead: collect `n_steps` transitions, then estimate the
advantage of each from the critic,

    A_t = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1} + gamma^n V(s_{t+n}) - V(s_t)

computed here with GAE (Schulman et al. 2015), whose lambda moves the estimate
between one-step TD (lambda = 0) and the rollout's Monte Carlo return (lambda = 1).
Updates happen every `n_steps` environment steps, not every episode, so long
episodes no longer starve the learner.

A3C ran many such learners asynchronously on CPU threads. The paper's own later
finding, and everyone's since, is that the synchronous version is as good and
much simpler. One environment here, for readability; `n_envs` parallel copies
would only change how the batch is gathered.

`collect` and `gae` are shared with `algos/ppo.py`. The update below is the part
that differs: one gradient step per batch, no ratio, no clipping.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 7e-4
N_STEPS = 32
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
HIDDEN = 64
MAX_GRAD = 0.5


class ActorCritic(nn.Module):
    """Separate actor and critic bodies. Sharing one body couples their gradient
    scales, which is a tuning problem this file does not need."""

    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        def body(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out))
        self.actor, self.critic = body(n_actions), body(1)

    def distribution(self, obs):
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class Rollout:
    """Stateful collector: keeps the current observation between calls."""

    def __init__(self, env, seed):
        self.env, self.rng = env, np.random.default_rng(seed)
        self.obs = np.asarray(env.reset(seed=seed)[0], dtype=np.float32)
        self.episode_return, self.finished = 0.0, []


def collect(rollout, model, n_steps):
    """n_steps transitions with the current policy. Returns a dict of tensors.

    `terminated` cuts the bootstrap. `truncated` does not: the value of the state
    the clock stopped in is folded into that step's reward, gamma * V(s_final),
    so GAE can treat the step as an ending without pretending the MDP ended.
    """
    obs, actions, rewards, dones, logps, values = [], [], [], [], [], []
    for _ in range(n_steps):
        o = torch.as_tensor(rollout.obs)[None]
        with torch.no_grad():
            dist = model.distribution(o)
            action = dist.sample()
            v = model.value(o)
        next_obs, reward, terminated, truncated, _ = rollout.env.step(int(action.item()))
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rollout.episode_return += reward
        if truncated and not terminated:
            with torch.no_grad():
                reward = reward + GAMMA * float(model.value(torch.as_tensor(next_obs)[None]))
        obs.append(o[0]); actions.append(action[0]); rewards.append(reward)
        dones.append(float(terminated or truncated)); logps.append(dist.log_prob(action)[0]); values.append(v[0])
        rollout.obs = next_obs
        if terminated or truncated:
            rollout.finished.append(rollout.episode_return)
            rollout.episode_return = 0.0
            rollout.obs = np.asarray(rollout.env.reset(seed=int(rollout.rng.integers(2**31 - 1)))[0],
                                     dtype=np.float32)
    with torch.no_grad():
        last_value = model.value(torch.as_tensor(rollout.obs)[None])[0]
    return dict(obs=torch.stack(obs), actions=torch.stack(actions), rewards=torch.tensor(rewards),
                dones=torch.tensor(dones), logp=torch.stack(logps), values=torch.stack(values),
                last_value=last_value)


def gae(batch, gamma=GAMMA, lam=LAMBDA):
    """Generalised advantage estimation, backwards over the batch. Returns (advantages, returns)."""
    rewards, values, dones = batch["rewards"], batch["values"], batch["dones"]
    advantages = torch.zeros_like(rewards)
    next_value, next_advantage = batch["last_value"], 0.0
    for t in reversed(range(len(rewards))):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        next_advantage = delta + gamma * lam * alive * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


def train(env, total_steps=100_000, gamma=GAMMA, lam=LAMBDA, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
          value_coef=VALUE_COEF, entropy_coef=ENTROPY_COEF, hidden=HIDDEN, seed=0, evaluate=None,
          eval_every=10_000):
    """Returns (model, log). Log rows: step, score, entropy, explained variance, grad norm."""
    torch.manual_seed(seed)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rollout = Rollout(env, seed)
    log, step, next_eval = [], 0, eval_every
    while step < total_steps:
        batch = collect(rollout, model, n_steps)
        step += n_steps
        advantages, returns = gae(batch, gamma, lam)
        # ---- the A2C update: one step on policy loss + value loss - entropy bonus
        dist = model.distribution(batch["obs"])
        values = model.value(batch["obs"])
        policy_loss = -(dist.log_prob(batch["actions"]) * advantages).mean()
        value_loss = ((values - returns) ** 2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimiser.zero_grad()
        loss.backward()
        grad = float(nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD))
        optimiser.step()

        if evaluate is not None and step >= next_eval:
            next_eval += eval_every
            explained = float(1 - (returns - values.detach()).var() / (returns.var() + 1e-8))
            log.append(dict(step=step, score=float(evaluate(model)), entropy=float(entropy.detach()),
                            explained_variance=explained, grad=grad))
    return model, log


def evaluate_gym(make_env, episodes=10, seed=10_000):
    """Mean return of the most probable action over fresh episodes."""
    def score(model):
        env, total = make_env(), 0.0
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                with torch.no_grad():
                    logits = model.actor(torch.as_tensor(np.asarray(obs, dtype=np.float32))[None])
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
    print("CartPole-v1, A2C, 100,000 steps, seed 0\n")
    print(f"{'step':>8} {'greedy':>8} {'entropy':>8} {'critic EV':>10} {'grad':>6}")
    for row in log:
        print(f"{row['step']:>8,} {row['score']:>8.1f} {row['entropy']:>8.3f} "
              f"{row['explained_variance']:>10.2f} {row['grad']:>6.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Entropy collapses to ~0 early and the return stops improving: the policy
#   became deterministic before it found anything good. Raise entropy_coef.
# - Critic explained variance stays negative: V is worse than predicting the mean,
#   so the advantages are noise. Check the value loss scale against the policy loss.
# - Returns cap below the time limit: truncation treated as termination. `collect`
#   folds gamma * V(s_final) into the last reward for exactly this.
# - Advantages leak across episodes: the `alive` mask in `gae` must cut both the
#   bootstrap and the recursion at every episode end inside a batch.
