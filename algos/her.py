"""Hindsight Experience Replay (Andrychowicz et al. 2017), on goal-conditioned DQN.

    python algos/her.py

On a sparse goal-reaching task, a failed episode teaches a standard agent nothing:
every reward is -1. But the episode did reach *somewhere*. HER stores each
transition a second time with the goal replaced by a state the agent actually
reached later in the same episode, and the reward recomputed for that goal.

    original     (s, a, r=-1, s', goal g)               failure
    relabelled   (s, a, r=compute_reward(s', g'), s', g')   g' = a state reached later

Some relabelled transitions are successes, so the value function gets real
signal from the first episode on, and a network that learns to reach the goals
it happened to reach generalises to the goal it was given.

This is the "future" strategy with k = 4: four extra copies per transition,
goals drawn from the states reached after it in the episode. `k=0` is plain
goal-conditioned DQN.

Uses `envs/bit_flip.py`. The network reads [state, goal]; a target network and
replay are DQN's.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import GRAD_CLIP, QNetwork, ReplayBuffer, greedy_action
from envs.bit_flip import BitFlip

GAMMA = 0.98
LEARNING_RATE = 1e-3
BUFFER_SIZE = 200_000
BATCH_SIZE = 128
K = 4
EPSILON = 0.2
TARGET_UPDATE = 1_000
HIDDEN = 256


def train(env, n_episodes=4_000, k=K, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, epsilon=EPSILON, target_update=TARGET_UPDATE, hidden=HIDDEN, updates_per_episode=None,
          seed=0, evaluate=None, eval_every=200):
    """Returns (net, log). Log rows: episode, success rate (greedy, fresh goals), mean Q."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = env.n
    net, target = QNetwork(2 * n, n, hidden), QNetwork(2 * n, n, hidden)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, 2 * n, seed)
    updates_per_episode = updates_per_episode or n
    updates, log = 0, []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        goal = obs[n:].copy()
        episode_steps = []                        # (state, action, next_state)
        done = False
        while not done:
            action = int(rng.integers(n)) if rng.random() < epsilon else greedy_action(net, obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_steps.append((obs[:n].copy(), action, next_obs[:n].copy()))
            buffer.add(obs, action, reward, next_obs, float(terminated))
            obs, done = next_obs, terminated or truncated

        # ---- the HER change: replay the episode against goals it actually reached
        for t, (state, action, next_state) in enumerate(episode_steps):
            for _ in range(k):
                future = int(rng.integers(t, len(episode_steps)))
                new_goal = episode_steps[future][2]
                reward = BitFlip.compute_reward(next_state, new_goal)
                buffer.add(np.concatenate([state, new_goal]), action, reward,
                           np.concatenate([next_state, new_goal]), float(reward == 0.0))

        for _ in range(updates_per_episode):
            o, a, r, o2, d = buffer.sample(batch_size)
            with torch.no_grad():
                y = r + gamma * (1.0 - d) * target(o2).max(1).values
                # Returns are bounded: -1 per step, at most 1/(1-gamma) steps.
                y = y.clamp(-1.0 / (1.0 - gamma), 0.0)
            q = net(o).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            optimiser.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP); optimiser.step()
            updates += 1
            if updates % target_update == 0:
                target.load_state_dict(net.state_dict())

        if evaluate is not None and episode % eval_every == 0:
            log.append(dict(episode=episode, success=float(evaluate(net)), q=float(q.detach().mean())))
    return net, log


def success_rate(make_env, episodes=50, seed=50_000):
    """Fraction of fresh episodes the greedy goal-conditioned policy solves."""
    def score(net):
        env, wins = make_env(), 0
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(greedy_action(net, obs))
                done = terminated or truncated
            wins += terminated
        return wins / episodes
    return score


def main():
    torch.set_num_threads(1)
    print("Bit flipping: greedy success rate on fresh goals, after 4,000 episodes\n")
    import json
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "results" / "exploration" / "her_bits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'bits':>5} {'DQN (k=0)':>10} {'DQN + HER (k=4)':>16}")
    for n in (5, 10, 15, 20):
        make = lambda n=n: BitFlip(n)
        row = []
        for k in (0, K):
            _, log = train(make(), n_episodes=4_000, k=k, evaluate=success_rate(make), eval_every=500, seed=0)
            row.append(log[-1]["success"])
            results[f"{n} bits, k={k}"] = log
            out.write_text(json.dumps(results))
        print(f"{n:>5} {row[0]:>10.0%} {row[1]:>16.0%}", flush=True)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - HER helps nothing: the relabelled reward is not recomputed, so the copies
#   carry the original -1. Every relabelled transition whose next state equals
#   its new goal must get reward 0 and be terminal.
# - The relabelled goal is drawn from BEFORE the transition: a goal already
#   passed is not one this action led to. "future" means index >= t.
# - The network cannot read the goal: the observation must contain it, or HER
#   only teaches the agent to reach wherever it usually goes.
# - Success rises then collapses at large n: Q targets drift below the true
#   bound. Clipping targets to [-1/(1-gamma), 0] is the paper's own fix.
