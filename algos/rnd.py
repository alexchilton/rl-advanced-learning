"""Random Network Distillation (Burda, Edwards, Storkey & Klimov, 2018), on DQN.

    python algos/rnd.py

Counts need states that repeat. In a large or continuous space they never do, so
RND estimates novelty with a prediction error instead:

    target network f       random, fixed forever
    predictor network g    trained to output f(s) on the states visited so far
    intrinsic reward       || g(s') - f(s') ||^2

States visited often have been fitted, so their error is small; new states have
not, so their error is large. The bonus generalises between similar states,
which a table of counts cannot.

The intrinsic reward is divided by a running estimate of its standard deviation,
so its scale stays fixed as the predictor improves. As in the paper, it gets its
own value head: Q_E learns the task's reward, Q_I the bonus (non-episodic: it does
not stop at episode ends), and the agent acts on Q_E + beta * Q_I. The final
policy is read from Q_E alone.

The single-head version, with the bonus added to the reward, was tried first and
is worth knowing about: on Chain(20) it reached the goal 2-3x sooner than DQN and
then ended with a greedy return of 0 on 4 of 5 seeds, even with the bonus faded
to zero and recomputed per batch. Its Q was flat at about 1.44 in every state,
above the largest achievable return of 1.0, so the greedy action was noise.

Training loop is `algos/dqn.py` with the marked additions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import (BATCH_SIZE, BUFFER_SIZE, GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE, LEARNING_STARTS,
                       TARGET_UPDATE, QNetwork, ReplayBuffer, as_vector, epsilon_at, greedy_action)

BETA = 1.0
RND_LEARNING_RATE = 1e-3
EMBED = 32


class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self):
        return float(np.sqrt(self.m2 / max(self.n - 1, 1))) + 1e-8


def train(env, n_steps=50_000, gamma=GAMMA, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
          batch_size=BATCH_SIZE, target_update=TARGET_UPDATE, learning_starts=LEARNING_STARTS, hidden=HIDDEN,
          beta=BETA, gamma_intrinsic=0.99, seed=0, evaluate=None, eval_every=5_000):
    """DQN with an RND bonus and two value heads; `beta=0` is plain DQN.

    Returns (extrinsic net, log, first_goal_step). The returned network values the
    TASK only: the bonus lives in a second network, so it can drive exploration
    without ever entering the values the final policy is read from.
    Log rows: step, score, mean intrinsic reward over the last eval period.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = as_vector(env.reset(seed=seed)[0])
    n_actions = env.action_space.n if hasattr(env, "action_space") else env.n_actions
    dim = len(obs)
    # ---- RND change 1: two value heads, extrinsic and intrinsic, each with a target copy
    q_e, q_i = QNetwork(dim, n_actions, hidden), QNetwork(dim, n_actions, hidden)
    t_e, t_i = QNetwork(dim, n_actions, hidden), QNetwork(dim, n_actions, hidden)
    t_e.load_state_dict(q_e.state_dict()); t_i.load_state_dict(q_i.state_dict())
    optimiser = torch.optim.Adam([*q_e.parameters(), *q_i.parameters()], lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, dim, seed)

    # ---- RND change 2: a fixed random network and a predictor that chases it
    def embed():
        return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, EMBED))
    rnd_target, predictor = embed(), embed()
    for p in rnd_target.parameters():
        p.requires_grad_(False)
    rnd_optimiser = torch.optim.Adam(predictor.parameters(), lr=RND_LEARNING_RATE)
    scale, intrinsic_log = RunningStd(), []

    def act(o):
        with torch.no_grad():
            x = torch.from_numpy(o)[None]
            return int((q_e(x) + beta * q_i(x)).argmax(1).item())

    log, first_goal = [], None
    for step in range(1, n_steps + 1):
        action = int(rng.integers(n_actions)) if rng.random() < epsilon_at(step, n_steps) else act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = as_vector(next_obs)
        if terminated and first_goal is None:
            first_goal = step
        if beta > 0:
            x = torch.from_numpy(next_obs)[None]
            error = ((predictor(x) - rnd_target(x)) ** 2).mean()
            rnd_optimiser.zero_grad(); error.backward(); rnd_optimiser.step()
            scale.update(float(error.detach()))
            intrinsic_log.append(float(error.detach()) / scale.std)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs = as_vector(env.reset(seed=int(rng.integers(2**31 - 1)))[0])

        if step >= learning_starts and buffer.size >= batch_size:
            o, a, r, o2, done = buffer.sample(batch_size)
            with torch.no_grad():
                y_e = r + gamma * (1.0 - done) * t_e(o2).max(1).values
                # ---- RND change 3: the intrinsic head learns the CURRENT novelty, and does
                # not stop at episode ends (the paper's non-episodic intrinsic return)
                novelty = ((predictor(o2) - rnd_target(o2)) ** 2).mean(1) / scale.std if beta > 0 else torch.zeros_like(r)
                y_i = novelty + gamma_intrinsic * t_i(o2).max(1).values
            loss = (nn.functional.smooth_l1_loss(q_e(o).gather(1, a[:, None]).squeeze(1), y_e)
                    + nn.functional.smooth_l1_loss(q_i(o).gather(1, a[:, None]).squeeze(1), y_i))
            optimiser.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([*q_e.parameters(), *q_i.parameters()], GRAD_CLIP); optimiser.step()
        if step % target_update == 0:
            t_e.load_state_dict(q_e.state_dict()); t_i.load_state_dict(q_i.state_dict())
        if evaluate is not None and step % eval_every == 0:
            log.append(dict(step=step, score=float(evaluate(q_e)),
                            intrinsic=float(np.mean(intrinsic_log[-eval_every:])) if intrinsic_log else 0.0))
    return q_e, log, first_goal


def main():
    from algos.dqn import q_table
    from envs.chain import Chain

    torch.set_num_threads(1)
    n = 20
    env = Chain(n=n, max_steps=4 * n, obs_mode="onehot")
    evaluate = lambda net: env.policy_return(q_table(net, env).argmax(1), GAMMA)
    print(f"Chain(n={n}), one-hot observations, 50,000 steps. Exact optimum {env.optimal_return(GAMMA):.4f}\n")
    for name, beta in (("DQN", 0.0), ("DQN + RND", BETA)):
        for seed in range(3):
            _, log, first = train(env, n_steps=50_000, beta=beta, evaluate=evaluate, eval_every=10_000, seed=seed)
            print(f"{name:>10} seed {seed}: goal first reached at step {first}; "
                  f"exact greedy return {[round(r['score'], 3) for r in log]}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The bonus is the same everywhere from the start: the predictor is too large
#   or its learning rate too high, so it fits every state immediately.
# - The bonus never decays: the predictor is not being trained, or is trained on
#   different inputs from the ones the bonus is computed on.
# - The agent hovers at a "noisy TV": a state whose observation is random cannot
#   be predicted however often it is seen. RND is fooled less than forward-model
#   curiosity, because the target is a deterministic function of the state, but a
#   stochastic observation still defeats it.
# - Works with one-hot observations, nothing with raw ones: normalise the
#   observation before it reaches either network; the paper whitens it.
