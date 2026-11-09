"""Curriculum failure: catastrophic forgetting when stages are trained one after another.

    python algos/curriculum_failure.py

A curriculum trains on stage 1, then stage 2, and so on. With a Q table that is
safe: each state has its own entry, and training stage 3 writes only stage 3's
entries. With a network every update moves shared weights, so training stage 3
can change what the network says about stage 1, and nothing in the curriculum
checks. The measurement this file exists for is the one the plan says nobody
plots: every earlier stage re-evaluated after every stage of training.

The task is goal-conditioned, so ONE network can solve every stage: a 7x7 grid,
start in the centre, observation [row, col, goal_row, goal_col] scaled to [0, 1],
+1 on reaching the goal. Stage k is goal k, the four corners in turn, each
trained for the same number of steps. Two numbers vary:

    rehearsal    fraction of episodes whose goal is drawn from the EARLIER stages.
                 0 is the plain schedule.
    buffer_size  replay capacity. A buffer holding every stage's transitions is
                 rehearsal by another name; a small one holds only the current stage.

The learner is `algos/dqn.py`'s network, buffer and Huber target, with epsilon
decayed from 1 to 0.05 over the first 20% of each stage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.dqn import GAMMA, GRAD_CLIP, HIDDEN, LEARNING_RATE, QNetwork, ReplayBuffer
from envs.gridworld import GridWorld

SIZE = 7
START = (3, 3)
GOALS = ((0, 0), (0, 6), (6, 6), (6, 0))
MAX_STEPS = 40
STAGE_STEPS = 30_000  # one stage alone: final greedy success 11/12 (4 goals x 3 seeds) at 30,000, 8/12 at 15,000
BUFFER_SIZE = 2_000
BATCH_SIZE = 64
TARGET_UPDATE = 500
LEARNING_STARTS = 500
EXPLORE_FRACTION = 0.2


class GoalGrid:
    """GridWorld with the goal appended to the observation. `set_goal` switches task."""

    def __init__(self, goal=GOALS[0]):
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = tuple(goal)
        self.env = GridWorld(size=SIZE, start=START, goal=self.goal, obs_mode="xy", max_steps=MAX_STEPS)

    def _obs(self, xy):
        return np.concatenate([xy, np.array(self.goal, dtype=np.float32) / (SIZE - 1)]).astype(np.float32)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(obs), reward, terminated, truncated, info


def goal_success(net, goals=GOALS):
    """Greedy rollout from the start for each goal: 1 if it arrives within MAX_STEPS, else 0.
    The grid is deterministic, so one rollout per goal is the exact answer."""
    env, result = GoalGrid(), []
    for goal in goals:
        env.set_goal(goal)
        obs, _ = env.reset(seed=0)
        while True:
            with torch.no_grad():
                action = int(net(torch.as_tensor(obs)[None]).argmax(1))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                result.append(int(terminated))
                break
    return result


def train(rehearsal=0.0, buffer_size=BUFFER_SIZE, stage_steps=STAGE_STEPS, goals=GOALS, gamma=GAMMA,
          learning_rate=LEARNING_RATE, hidden=HIDDEN, batch_size=BATCH_SIZE, target_update=TARGET_UPDATE,
          learning_starts=LEARNING_STARTS, eval_every=500, seed=0):
    """Returns (net, log). Log rows: step, stage, success (one 0/1 per goal)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = GoalGrid(goals[0])
    net, target = QNetwork(4, 4, hidden), QNetwork(4, 4, hidden)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, 4, seed)

    def new_episode(stage):
        # ---- the knob: with probability `rehearsal`, an earlier stage's goal
        if stage > 0 and rng.random() < rehearsal:
            env.set_goal(goals[int(rng.integers(stage))])
        else:
            env.set_goal(goals[stage])
        return env.reset(seed=int(rng.integers(2**31 - 1)))[0]

    log, step = [], 0
    for stage in range(len(goals)):
        obs = new_episode(stage)
        for t in range(stage_steps):
            step += 1
            epsilon = max(0.05, 1.0 - t / (EXPLORE_FRACTION * stage_steps) * 0.95)
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                with torch.no_grad():
                    action = int(net(torch.as_tensor(obs)[None]).argmax(1))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(terminated))
            obs = next_obs if not (terminated or truncated) else new_episode(stage)

            if step >= learning_starts and buffer.size >= batch_size:
                o, a, r, o2, done = buffer.sample(batch_size)
                with torch.no_grad():
                    y = r + gamma * (1.0 - done) * target(o2).max(1).values
                q = net(o).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                optimiser.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                optimiser.step()
            if step % target_update == 0:
                target.load_state_dict(net.state_dict())
            if step % eval_every == 0:
                log.append(dict(step=step, stage=stage, success=goal_success(net, goals)))
    return net, log


def main():
    torch.set_num_threads(1)
    for rehearsal, buffer_size in ((0.0, BUFFER_SIZE), (0.25, BUFFER_SIZE), (0.0, 4 * STAGE_STEPS)):
        _, log = train(rehearsal=rehearsal, buffer_size=buffer_size, seed=0)
        print(f"rehearsal {rehearsal}, buffer {buffer_size:,}: success per goal at the end of each stage")
        for stage in range(len(GOALS)):
            print(f"  after stage {stage + 1}: {[r for r in log if r['stage'] == stage][-1]['success']}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Every earlier goal fails after the last stage and the curriculum reports
#   success: it only ever evaluated the current stage. Evaluate all of them.
# - Forgetting with a table: impossible; each entry is written only by its own
#   states. If a tabular curriculum loses an early stage, the stages share states
#   and have conflicting optima, which is a different problem.
# - Rehearsal "does nothing": check the buffer. A buffer large enough to hold the
#   early stages is already rehearsing, and adding more changes little.
# - The goal is missing from the observation: then the stages genuinely conflict
#   and no network can solve them all; forgetting is the correct answer.
