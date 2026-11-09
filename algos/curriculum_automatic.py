"""Automatic curriculum: a teacher picks the next task by learning progress
(Matiisen, Oliver, Cohen & Schulman 2017, "Teacher-Student Curriculum Learning").

    python algos/curriculum_automatic.py

The reverse curriculum (`algos/curriculum_reverse.py`) moves the start outward
on a hand-written rule: grow the radius when greedy rollouts succeed 80% of the
time. Here the same tasks, starts grouped into buckets by distance to the goal,
are chosen by a teacher instead:

    random     every episode starts in a bucket chosen uniformly
    progress   the teacher keeps the last WINDOW outcomes (goal reached or not)
               of the student's own training episodes in each bucket, and picks
               the bucket whose success rate is changing fastest:
                   LP_k = | mean(newer half) - mean(older half) |
               greedily, with probability TEACHER_EPSILON of a uniform bucket.
               A bucket with fewer than WINDOW outcomes counts as LP = 1.
               This is a simplified version of the paper's Window algorithm, which
               fits a regression slope against timesteps and smooths it with a
               non-stationary bandit update; here the half-window difference is
               used directly. The absolute value is the paper's: a falling score
               means forgetting, and that task should be practised too.

A solved bucket has LP 0 (always succeeds), a hopeless one has LP 0 (always
fails); only the frontier moves. That is the whole argument for the teacher,
and random is the baseline it has to beat: random also puts some episodes on
the frontier, at no cost in machinery.

The teacher sees only the outcomes of training episodes, which cost environment
steps. The fixed reverse schedule decides with extra greedy rollouts that are
not counted, so it has information this teacher does not.

The student is tabular Q-learning with `algos/count_based_exploration.py`'s
settings (alpha 0.5, epsilon 0.1, random tie-breaking).
"""

from __future__ import annotations

import numpy as np

from algos.curriculum_reverse import distance_to_goal
from envs.chain import Chain

GAMMA = 0.99
ALPHA = 0.5
EPSILON = 0.1
N_BUCKETS = 6
WINDOW = 10
TEACHER_EPSILON = 0.2


def buckets(env, n_buckets=N_BUCKETS):
    """Reachable non-goal states grouped into n_buckets bands of distance to the goal."""
    distance = distance_to_goal(env)
    real_start = int(np.argmax(env.start_distribution))
    edges = np.linspace(0, distance[real_start], n_buckets + 1)
    states = np.flatnonzero(distance < 10**6)
    groups = [states[(distance[states] > lo) & (distance[states] <= hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    return [g for g in groups if len(g)]


class ProgressTeacher:
    def __init__(self, n_tasks, window=WINDOW, epsilon=TEACHER_EPSILON):
        self.history = [[] for _ in range(n_tasks)]
        self.window, self.epsilon = window, epsilon

    def progress(self):
        lp = []
        for h in self.history:
            if len(h) < self.window:
                lp.append(1.0)
            else:
                recent = h[-self.window:]
                half = self.window // 2
                lp.append(abs(np.mean(recent[half:]) - np.mean(recent[:half])))
        return np.array(lp)

    def choose(self, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self.history)))
        lp = self.progress()
        return int(rng.choice(np.flatnonzero(lp == lp.max())))

    def update(self, task, outcome):
        self.history[task].append(float(outcome))


class RandomTeacher:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks

    def choose(self, rng):
        return int(rng.integers(self.n_tasks))

    def update(self, task, outcome):
        pass


def train(env, selector="progress", n_steps=200_000, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
          n_buckets=N_BUCKETS, window=WINDOW, teacher_epsilon=TEACHER_EPSILON, eval_every=1_000, seed=0):
    """Returns (Q, curve, choices). `curve` is (step, exact greedy return from the real
    start); `choices` is (step, bucket) for every episode, for the plot of where the
    teacher sent the student."""
    rng = np.random.default_rng(seed)
    tasks = buckets(env, n_buckets)
    teacher = ProgressTeacher(len(tasks), window, teacher_epsilon) if selector == "progress" else RandomTeacher(len(tasks))
    Q = np.zeros((env.n_states, env.n_actions))
    greedy_return = lambda: env.policy_return(Q.argmax(axis=1), gamma)

    def start():
        task = teacher.choose(rng)
        env.reset(seed=int(rng.integers(2**31 - 1)))
        env.state = int(rng.choice(tasks[task]))
        return task

    curve, choices = [(0, greedy_return())], []
    task = start()
    choices.append((0, task))
    for step in range(1, n_steps + 1):
        s = env.state
        if rng.random() < epsilon:
            a = int(rng.integers(env.n_actions))
        else:
            a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        _, r, terminated, truncated, _ = env.step(a)
        target = r + (0.0 if terminated else gamma * Q[env.state].max())
        Q[s, a] += alpha * (target - Q[s, a])
        if terminated or truncated:
            teacher.update(task, terminated)
            task = start()
            choices.append((step, task))
        if step % eval_every == 0:
            curve.append((step, greedy_return()))
    return Q, curve, choices


def main():
    env = Chain(n=30, max_steps=120)
    optimal = env.optimal_return(GAMMA)
    print(f"Chain(30), exact optimum {optimal:.4f}; steps until the greedy policy reaches 95% of it\n")
    for selector in ("random", "progress"):
        hits = []
        for seed in range(5):
            _, curve, _ = train(Chain(n=30, max_steps=120), selector, n_steps=200_000, eval_every=100, seed=seed)
            hits.append(next((s for s, v in curve if v >= 0.95 * optimal), None))
        print(f"{selector:>9}: {hits}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - The teacher keeps choosing the easiest bucket: LP measured as the success
#   rate itself, not its change. A solved task has high success and zero progress.
# - The teacher never leaves the hardest bucket: absolute value missing, or
#   untried buckets given infinite priority forever (they must lose it after
#   WINDOW outcomes).
# - Random wins: not a bug. Matiisen et al. found uniform sampling competitive on
#   one of their tasks too; report it.
# - The buckets leave out the real start: then the curve never reaches the
#   optimum from the real start, however good the teacher is. Print the buckets.
