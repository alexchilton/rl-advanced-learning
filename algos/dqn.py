"""DQN (Mnih et al. 2013, 2015).

    python algos/dqn.py

Q-learning with a neural network in place of the table:

    loss = huber( Q(s,a; theta)  -  [ r + gamma * max_a' Q(s',a'; theta_target) ] )

Written that way it diverges. Two additions make it work, and the 2015 paper is
mostly about them:

    replay buffer    train on random old transitions, not the last one. Consecutive
                     transitions are almost identical, so fitting them in order is
                     fitting one point over and over, and the network forgets the rest.
    target network   compute the target with a frozen copy of the network, refreshed
                     every `target_update` steps. Otherwise every update moves the
                     target it is chasing.

Both are plain numbers here, not switches. `buffer_size=1` is no replay: every
update uses only the newest transition. `target_update=1` is no target network:
the copy is refreshed every step, so it is the online network.
`experiments/dqn_ablation.py` turns them off one at a time.

Everything else is deliberately plain: an MLP, Huber loss, Adam, a linear epsilon
decay. `double_dqn.py`, `dueling_dqn.py`, `per_dqn.py` and `n_step_dqn.py` each change
one thing about this file, and the duplication is so the change is on screen.

Runs on Gymnasium environments and on this repo's tabular ones with
`obs_mode="onehot"`, where the learned Q can be compared with the exact one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

GAMMA = 0.99
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
TARGET_UPDATE = 500
LEARNING_STARTS = 1_000
EPSILON_START, EPSILON_END, EPSILON_FRACTION = 1.0, 0.05, 0.1
HIDDEN = 128
GRAD_CLIP = 10.0


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ReplayBuffer:
    """A ring buffer of transitions. Sampling is uniform."""

    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # With capacity 1 this returns the newest transition every time, which is
        # exactly online Q-learning with a network.
        index = self.rng.integers(0, self.size, batch_size)
        return (torch.from_numpy(self.obs[index]), torch.from_numpy(self.actions[index]),
                torch.from_numpy(self.rewards[index]), torch.from_numpy(self.next_obs[index]),
                torch.from_numpy(self.terminated[index]))


def as_vector(obs):
    return np.atleast_1d(np.asarray(obs, dtype=np.float32))


def epsilon_at(step, n_steps, start=EPSILON_START, end=EPSILON_END, fraction=EPSILON_FRACTION):
    progress = min(step / max(fraction * n_steps, 1), 1.0)
    return start + progress * (end - start)


def greedy_action(net, obs):
    with torch.no_grad():
        return int(net(torch.from_numpy(as_vector(obs))[None]).argmax(1).item())


def train(
    env,
    n_steps=50_000,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    target_update=TARGET_UPDATE,
    learning_starts=LEARNING_STARTS,
    hidden=HIDDEN,
    seed=0,
    evaluate=None,
    eval_every=5_000,
):
    """Returns (net, log). `evaluate(net) -> float` is called every `eval_every` steps.

    `log` has one dict per evaluation: step, score, and the health numbers worth
    watching (mean Q on the batch, mean |TD error|, gradient norm before clipping).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    obs = as_vector(obs)
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
        # Store `terminated`, not `terminated or truncated`. A time limit is not the
        # end of the MDP, so the target must still bootstrap from next_obs.
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])

        if step >= min(learning_starts, buffer_size) and buffer.size >= batch_size:
            o, a, r, o2, done = buffer.sample(batch_size)
            with torch.no_grad():
                bootstrap = target(o2).max(1).values
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


def evaluate_gym(make_env, episodes=10, seed=10_000):
    """Mean greedy return over fresh episodes. Returns a function of the network."""
    def score(net):
        env, total = make_env(), 0.0
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(greedy_action(net, obs))
                total += reward
                done = terminated or truncated
        return total / episodes
    return score


def value_vs_return(net, make_env, gamma=GAMMA, episodes=10, seed=20_000):
    """(mean predicted max Q, mean actual discounted return-to-go) over greedy episodes.

    The predicted number is what the network believes each visited state is worth.
    The actual number is what following its greedy policy from that state really
    returned, discounted the same way. Their gap is overestimation, in return units.
    This is the measurement van Hasselt et al. 2015 is built on.
    """
    predicted, actual = [], []
    env = make_env()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        values, rewards, done, truncated = [], [], False, False
        while not done:
            with torch.no_grad():
                q = net(torch.from_numpy(as_vector(obs))[None])[0]
            values.append(float(q.max()))
            obs, reward, terminated, truncated, _ = env.step(int(q.argmax()))
            rewards.append(reward)
            done = terminated or truncated
        # A truncated episode would have continued, so its tail is bootstrapped
        # from the network's own estimate rather than counted as zero.
        running = values[-1] if truncated else 0.0
        if truncated:
            values, rewards = values[:-1], rewards[:-1]
        for value, reward in zip(reversed(values), reversed(rewards)):
            running = reward + gamma * running
            predicted.append(value)
            actual.append(running)
    return float(np.mean(predicted)), float(np.mean(actual))


def q_table(net, env):
    """The network's Q for every state of a tabular env, for comparison with env.true_q."""
    obs = np.stack([as_vector(env.observation(s)) for s in range(env.n_states)])
    with torch.no_grad():
        return net(torch.from_numpy(obs)).numpy()


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    net, log = train(make(), n_steps=100_000, evaluate=evaluate_gym(make), eval_every=10_000, seed=0)
    print("CartPole-v1, DQN, 100,000 steps, seed 0 (max return 500)\n")
    print(f"{'step':>7} {'greedy return':>14} {'mean Q':>8} {'|TD|':>7} {'grad norm':>10}")
    for row in log:
        print(f"{row['step']:>7,} {row['score']:>14.1f} {row['q']:>8.2f} {row['td']:>7.3f} {row['grad']:>10.3f}")
    predicted, actual = value_vs_return(net, make)
    print(f"\nover greedy episodes: network predicts {predicted:.2f}, discounted return actually {actual:.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Mean Q climbs without bound: no target network (target_update=1), or gamma near
#   1 with no terminal mask. Log mean Q; it should level off near the true return.
# - Return rises then collapses and does not recover: the classic DQN instability.
#   Smaller learning rate, longer target period, or Double DQN.
# - CartPole hovers near 200 and you expected 500: you are treating the 500-step
#   time limit as termination, so the agent learns that late states are worth 0.
#   Store `terminated`, not `terminated or truncated`.
# - Nothing is learned and the loss is tiny: the buffer is full of the first
#   random episodes and epsilon decayed before any good transition arrived.
#   Check the epsilon schedule against n_steps.
# - Works on CartPole, fails on a one-state probe: see diagnostics/probe_envs.py.
#   A probe failure points at one component; a CartPole failure points at all of them.
