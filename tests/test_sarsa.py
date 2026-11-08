"""SARSA against Q-learning on the cliff: the on-policy/off-policy split, asserted.

Settings match `experiments/cliff_sarsa_vs_qlearning.py`. Measured there on 5 of 5
seeds: Q-learning walks the 13-step edge path, SARSA the 17-step safe path, and
each wins exactly one of the two exact-return columns.
"""

import pytest

from algos import q_learning, sarsa
from envs import solvers
from envs.cliff import CliffWalking

GAMMA = 0.99
SETTINGS = dict(gamma=GAMMA, alpha=0.2, epsilon=0.1, n_steps=30_000, eval_every=30_000)


@pytest.fixture(scope="module")
def learned():
    env = CliffWalking()
    return env, {
        name: [method.train(env, seed=seed, **SETTINGS)[0] for seed in range(3)]
        for name, method in (("q", q_learning), ("sarsa", sarsa))
    }


def test_q_learning_walks_the_edge(learned):
    env, tables = learned
    for Q in tables["q"]:
        assert len(sarsa.greedy_path(env, Q)) - 1 == 13


def test_sarsa_takes_the_safe_path(learned):
    env, tables = learned
    for Q in tables["sarsa"]:
        path = sarsa.greedy_path(env, Q)
        assert path[-1] == env.goal, f"SARSA's greedy policy never reached the goal: {path}"
        assert len(path) - 1 > 13, "SARSA learned the edge path -- is it bootstrapping from max Q?"


def test_each_method_wins_one_column(learned):
    """Q-learning's greedy policy is better; the policy SARSA actually follows is better."""
    env, tables = learned
    for q_table, s_table in zip(tables["q"], tables["sarsa"]):
        greedy_q = env.policy_return(q_table.argmax(axis=1), GAMMA)
        greedy_s = env.policy_return(s_table.argmax(axis=1), GAMMA)
        followed_q = env.policy_return(solvers.epsilon_greedy_policy(q_table, 0.1), GAMMA)
        followed_s = env.policy_return(solvers.epsilon_greedy_policy(s_table, 0.1), GAMMA)
        assert greedy_q > greedy_s
        assert followed_s > followed_q
