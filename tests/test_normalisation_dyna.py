"""PopArt preserves outputs exactly; Dyna-Q's planning reduces the real experience needed."""

import torch

from algos import dyna_q
from algos.popart_dqn import PopArtQ
from envs.gridworld import GridWorld

torch.set_num_threads(1)


def test_popart_preserves_outputs_when_statistics_move():
    torch.manual_seed(0)
    net = PopArtQ(4, 3)
    x = torch.randn(32, 4)
    before = net(x).detach().clone()
    net.update_stats(torch.randn(64) * 50 + 200, rate=0.5)
    assert float(net.mu) > 50, "statistics should have moved"
    assert float((net(x) - before).abs().max()) < 1e-3


def test_dyna_planning_needs_fewer_real_steps():
    env = GridWorld(size=8, reward_mode="sparse", max_steps=200)
    optimal = env.optimal_return(dyna_q.GAMMA)

    def steps(n):
        _, curve = dyna_q.train(env, n_steps=15_000, planning_steps=n, seed=0)
        return next((s for s, v in curve if v >= optimal - 1e-6), None)

    plain, planned = steps(0), steps(20)
    assert planned is not None and (plain is None or planned < plain), (plain, planned)
