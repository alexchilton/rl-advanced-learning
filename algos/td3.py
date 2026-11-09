"""TD3: Twin Delayed DDPG (Fujimoto, van Hoof & Meger, 2018).

    python algos/td3.py

DDPG's actor maximises the critic, so it seeks out the critic's errors. Any
overestimation becomes the policy's target, and the target network carries it
into the next update. TD3 is three fixes, each a number here so they can be
removed one at a time (`experiments/continuous_control_sweep.py ablation`):

    n_critics=2       clipped double Q: two critics, the target uses the SMALLER.
                      Biases the target down rather than up. n_critics=1 removes it.
    policy_delay=2    update the actor (and the targets) every 2nd critic update,
                      so the actor chases a critic that has had time to settle.
                      policy_delay=1 removes it.
    target_noise=0.2  target policy smoothing: add clipped noise to the target
                      action, so the target averages Q over a small neighbourhood
                      and a narrow spike in the critic cannot be exploited.
                      target_noise=0 removes it.

With all three removed this is `algos/ddpg.py` with DDPG's own learning rates.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from algos.ddpg import (BATCH_SIZE, BUFFER_SIZE, GAMMA, HIDDEN, START_STEPS, TAU, ReplayBuffer,
                        critic_input, evaluate_gym, mlp, soft_update, spaces)

LEARNING_RATE = 3e-4
EXPLORATION_NOISE = 0.1
N_CRITICS = 2
POLICY_DELAY = 2
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5


def train(env, total_steps=20_000, gamma=GAMMA, tau=TAU, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, hidden=HIDDEN, start_steps=START_STEPS, exploration_noise=EXPLORATION_NOISE,
          n_critics=N_CRITICS, policy_delay=POLICY_DELAY, target_noise=TARGET_NOISE, noise_clip=NOISE_CLIP,
          seed=0, evaluate=None, eval_every=2_000, obs_transform=None):
    """Returns (actor, log). Log rows: step, score, mean Q (first critic), critic loss.

    `obs_transform` maps a tensor of stored observations to what the networks see,
    applied when acting and to every sampled batch, so the replay buffer holds raw
    observations (`experiments/obs_normalisation.py`, sample-time normalisation).
    """
    view = obs_transform or (lambda x: x)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim, act_dim, scale = spaces(env)
    actor = mlp(obs_dim, act_dim, hidden, nn.Tanh())
    critics = nn.ModuleList(mlp(obs_dim + act_dim, 1, hidden) for _ in range(n_critics))
    actor_target, critics_target = copy.deepcopy(actor), copy.deepcopy(critics)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, obs_dim, act_dim, seed)

    obs = np.asarray(env.reset(seed=seed)[0], np.float32)
    log, health, updates = [], dict(q=0.0, critic_loss=0.0), 0
    for step in range(1, total_steps + 1):
        if step <= start_steps:
            action = rng.uniform(-1, 1, act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor(view(torch.as_tensor(obs)[None]))[0].numpy()
            action = np.clip(action + exploration_noise * rng.standard_normal(act_dim), -1, 1).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action * scale)
        next_obs = np.asarray(next_obs, np.float32)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = np.asarray(env.reset(seed=int(rng.integers(2**31 - 1)))[0], np.float32)

        if step > start_steps:
            updates += 1
            o, a, r, o2, done = buffer.sample(batch_size)
            o, o2 = view(o), view(o2)
            with torch.no_grad():
                # ---- fix 3: target policy smoothing
                noise = (torch.randn_like(a) * target_noise).clamp(-noise_clip, noise_clip)
                a2 = (actor_target(o2) + noise).clamp(-1, 1)
                # ---- fix 1: the smaller of the target critics
                target_q = torch.stack([c(critic_input(o2, a2)).squeeze(1) for c in critics_target]).min(0).values
                y = r + gamma * (1 - done) * target_q
            qs = [c(critic_input(o, a)).squeeze(1) for c in critics]
            critic_loss = sum(((q - y) ** 2).mean() for q in qs)
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            # ---- fix 2: delayed actor and target updates
            if updates % policy_delay == 0:
                actor_loss = -critics[0](critic_input(o, actor(o))).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                soft_update(critics_target, critics, tau)
                soft_update(actor_target, actor, tau)
            health = dict(q=float(qs[0].detach().mean()), critic_loss=float(critic_loss.detach()))

        if evaluate is not None and step % eval_every == 0:
            log.append(dict(step=step, score=float(evaluate(actor)), **health))
    return actor, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("Pendulum-v1")
    _, log = train(make(), total_steps=20_000, evaluate=evaluate_gym(make), seed=0)
    print("Pendulum-v1, TD3, 20,000 steps, seed 0\n")
    for row in log:
        print(f"{row['step']:>7,} return {row['score']:>8.1f}   mean Q {row['q']:>8.2f}   critic loss {row['critic_loss']:.3f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Target uses the max or the mean of the two critics: no bias correction. It
#   must be the minimum.
# - Both critics initialised identically: torch.manual_seed then building them in
#   a loop gives different weights; copying one into the other does not. With
#   identical critics the minimum is a no-op.
# - Actor updated through critic 2 or through min(Q1, Q2): the paper uses Q1;
#   min works too but is a different method. Be explicit about which.
# - Learns slower than DDPG early on: expected; the delayed actor takes half as
#   many steps. Compare at the end, not at step 5,000.
