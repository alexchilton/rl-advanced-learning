"""Probe environments (Andy Jones, "Debugging RL, Without the Agonizing Pain").

Five environments of one or two steps. Each has an exactly known Q function, and
each isolates one part of a value-based agent. Run them in order; the first one
that fails tells you which part is broken.

    1  constant      one action, obs 0, reward +1, ends.          Q = 1
                     Fails: the value head or the loss cannot fit a constant.
    2  observation   one action, obs ±1, reward = obs, ends.        Q(obs) = obs
                     Fails: the network ignores its input.
    3  discount      one action, two steps, reward +1 only at the end. Q(first) = gamma, Q(second) = 1
                     Fails: bootstrapping or the terminal mask.
    4  action        two actions, obs 0, reward +1 for action 0, -1 for action 1, ends.
                     Q = [+1, -1]. Fails: the update ignores which action was taken.
    5  action x obs  two actions, obs ±1, reward +1 if the action matches the obs sign.
                     Q(+1) = [-1, +1], Q(-1) = [+1, -1]. Fails: actions and observations
                     are not combined.

Each probe exposes `action_space.n`, `reset`, `step` and `expected_q(gamma)`, so
any trainer in `algos/` that runs on Gymnasium runs here unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _Probe:
    n_actions = 1
    obs_values = (0.0,)

    def __init__(self):
        self.action_space = SimpleNamespace(n=self.n_actions)
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.obs = float(self.rng.choice(self.obs_values))
        return np.array([self.obs], dtype=np.float32), {}


class Constant(_Probe):
    def step(self, action):
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}

    def expected_q(self, gamma):
        return {0.0: [1.0]}


class Observation(_Probe):
    obs_values = (-1.0, 1.0)

    def step(self, action):
        return np.array([0.0], dtype=np.float32), self.obs, True, False, {}

    def expected_q(self, gamma):
        return {-1.0: [-1.0], 1.0: [1.0]}


class Discount(_Probe):
    def step(self, action):
        self.t += 1
        if self.t == 1:
            self.obs = 1.0
            return np.array([1.0], dtype=np.float32), 0.0, False, False, {}
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}

    def expected_q(self, gamma):
        return {0.0: [gamma], 1.0: [1.0]}


class Action(_Probe):
    n_actions = 2

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 1.0 if action == 0 else -1.0, True, False, {}

    def expected_q(self, gamma):
        return {0.0: [1.0, -1.0]}


class ActionObservation(_Probe):
    n_actions = 2
    obs_values = (-1.0, 1.0)

    def step(self, action):
        wanted = 1 if self.obs > 0 else 0
        return np.array([0.0], dtype=np.float32), 1.0 if action == wanted else -1.0, True, False, {}

    def expected_q(self, gamma):
        return {-1.0: [1.0, -1.0], 1.0: [-1.0, 1.0]}


class CoinFlip(_Probe):
    """One action, obs 0, reward +1 with probability 1/4 and -1 otherwise, ends.

    Not one of the five probes: its expected Q is -0.5, which says nothing new. Its
    return DISTRIBUTION is the point: two spikes, 1/4 at +1 and 3/4 at -1, which a
    distributional agent (`algos/c51_dqn.py`) must recover exactly.
    """

    p_win = 0.25

    def step(self, action):
        reward = 1.0 if self.rng.random() < self.p_win else -1.0
        return np.array([0.0], dtype=np.float32), reward, True, False, {}

    def expected_q(self, gamma):
        return {0.0: [2 * self.p_win - 1]}

    def return_distribution(self):
        return {1.0: self.p_win, -1.0: 1 - self.p_win}


PROBES = (("1 constant", Constant), ("2 observation", Observation), ("3 discount", Discount),
          ("4 action", Action), ("5 action x obs", ActionObservation))


def run_probes(train, gamma=0.99, n_steps=3_000, tolerance=0.1, wrap=None, **kwargs):
    """Train on every probe and compare the learned Q with the exact one.

    `train(env, n_steps=..., gamma=..., **kwargs) -> (net, log)`, as in algos/dqn.py.
    `wrap(env) -> env` optionally wraps each probe, which is how a bug in the
    environment interface (for example termination reported as truncation) is
    injected without editing the algorithm. Returns a list of result dicts.
    """
    import torch

    results = []
    for name, make in PROBES:
        env = make()
        net, _ = train(wrap(env) if wrap else env, n_steps=n_steps, gamma=gamma, **kwargs)
        worst, rows = 0.0, []
        for obs, expected in env.expected_q(gamma).items():
            with torch.no_grad():
                learned = net(torch.tensor([[obs]], dtype=torch.float32))[0].numpy()
            worst = max(worst, float(np.abs(learned - expected).max()))
            rows.append((obs, [round(float(x), 3) for x in learned], expected))
        results.append(dict(probe=name, passed=worst <= tolerance, worst_error=worst, rows=rows))
    return results


def report(results):
    lines = [f"{'probe':<16} {'result':<6} {'worst |error|':>13}   obs: learned vs expected"]
    for r in results:
        detail = "; ".join(f"{o:+.0f}: {l} vs {e}" for o, l, e in r["rows"])
        lines.append(f"{r['probe']:<16} {'pass' if r['passed'] else 'FAIL':<6} {r['worst_error']:>13.3f}   {detail}")
    return "\n".join(lines)
