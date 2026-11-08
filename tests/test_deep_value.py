"""The DQN family, checked where the answer is known exactly: the probe environments,
the PER sum tree, and the maximisation-bias MDP."""

import numpy as np
import pytest
import torch

from algos import double_dqn, double_q, dqn, dueling_dqn, n_step_dqn, per_dqn
from diagnostics.probe_envs import run_probes
from envs.maximization_bias import MaximizationBias

torch.set_num_threads(1)
PROBE_SETTINGS = dict(learning_starts=200, buffer_size=5_000, target_update=100)


@pytest.mark.parametrize("module", [dqn, double_dqn, dueling_dqn, per_dqn, n_step_dqn],
                         ids=lambda m: m.__name__.split(".")[-1])
def test_every_variant_passes_every_probe(module):
    results = run_probes(module.train, **PROBE_SETTINGS)
    failed = [(r["probe"], round(r["worst_error"], 3)) for r in results if not r["passed"]]
    assert not failed, f"{module.__name__} failed probes: {failed}"


class TerminationAsTruncation:
    """Reports every episode end as a time limit, so the target never stops bootstrapping."""

    def __init__(self, env):
        self.env, self.action_space = env, env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, terminated or truncated, info


def test_probes_catch_a_missing_terminal_mask():
    results = {r["probe"]: r for r in run_probes(dqn.train, wrap=TerminationAsTruncation, **PROBE_SETTINGS)}
    assert not results["1 constant"]["passed"]
    assert results["1 constant"]["rows"][0][1][0] > 2.0, "Q should run far above 1 with no terminal mask"


def test_sum_tree_samples_in_proportion_to_priority():
    buffer = per_dqn.PrioritizedReplayBuffer(8, 1, seed=0, alpha=1.0)
    for i in range(4):
        buffer.add(np.zeros(1), 0, 0.0, np.zeros(1), 0.0)
    buffer.update(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]) - per_dqn.PRIORITY_EPS)
    counts = np.zeros(4)
    for _ in range(2_000):
        index, _ = buffer.sample(10, beta=1.0)
        np.add.at(counts, index, 1)
    np.testing.assert_allclose(counts / counts.sum(), [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_double_q_removes_most_of_the_maximisation_bias():
    env = MaximizationBias()
    runs = 200
    q_left = np.mean([double_q.run(env, double=False, episodes=50, seed=s)[10:50] for s in range(runs)])
    dq_left = np.mean([double_q.run(env, double=True, episodes=50, seed=s)[10:50] for s in range(runs)])
    assert q_left > 0.6, f"Q-learning should prefer the losing branch early, went left {q_left:.0%}"
    assert dq_left < q_left - 0.3, f"Double Q went left {dq_left:.0%} vs Q-learning {q_left:.0%}"
