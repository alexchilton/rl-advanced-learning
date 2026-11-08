"""Policy-gradient pieces with exact answers: returns-to-go, GAE's two limits, the
tanh log-probability correction, and one short learning check per family."""

import numpy as np
import pytest
import torch

from algos import a2c, ppo, reinforce, sac, td3

torch.set_num_threads(1)


def test_returns_to_go():
    np.testing.assert_allclose(reinforce.returns_to_go([1.0, 1.0, 1.0], 0.5), [1.75, 1.5, 1.0])


def _batch(rewards, values, dones, last_value):
    return dict(rewards=torch.tensor(rewards), values=torch.tensor(values),
                dones=torch.tensor(dones), last_value=torch.tensor(last_value))


def test_gae_lambda_zero_is_the_td_error():
    b = _batch([1.0, 2.0, 3.0], [0.5, 0.4, 0.3], [0.0, 0.0, 0.0], 0.2)
    adv, _ = a2c.gae(b, gamma=0.9, lam=0.0)
    td = [1.0 + 0.9 * 0.4 - 0.5, 2.0 + 0.9 * 0.3 - 0.4, 3.0 + 0.9 * 0.2 - 0.3]
    np.testing.assert_allclose(adv.numpy(), td, rtol=1e-6)


def test_gae_lambda_one_is_monte_carlo_minus_value():
    b = _batch([1.0, 2.0, 3.0], [0.5, 0.4, 0.3], [0.0, 0.0, 1.0], 99.0)   # episode ends at t=2
    adv, ret = a2c.gae(b, gamma=0.9, lam=1.0)
    mc = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3.0]
    np.testing.assert_allclose(ret.numpy(), mc, rtol=1e-6)
    np.testing.assert_allclose(adv.numpy(), np.array(mc) - [0.5, 0.4, 0.3], rtol=1e-6)


def test_gae_does_not_leak_across_an_episode_end():
    b = _batch([0.0, 5.0], [0.0, 0.0], [1.0, 0.0], 0.0)
    adv, _ = a2c.gae(b, gamma=0.99, lam=0.95)
    assert adv[0].item() == pytest.approx(0.0), "reward after the episode ended leaked into it"


def test_tanh_gaussian_log_prob_matches_change_of_variables():
    torch.manual_seed(0)
    policy = sac.TanhGaussian(3, 2, hidden=16)
    obs = torch.randn(64, 3)
    torch.manual_seed(1)
    action, log_prob = policy.sample(obs)
    # Recompute independently: same noise, log N(u) - sum log(1 - tanh(u)^2).
    torch.manual_seed(1)
    mean, log_std = policy.net(obs).chunk(2, dim=-1)
    std = log_std.clamp(sac.LOG_STD_MIN, sac.LOG_STD_MAX).exp()
    u = mean + std * torch.randn_like(mean)
    direct = (torch.distributions.Normal(mean, std).log_prob(u) - torch.log(1 - torch.tanh(u) ** 2 + 1e-12)).sum(-1)
    np.testing.assert_allclose(action.detach().numpy(), torch.tanh(u).detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(log_prob.detach().numpy(), direct.detach().numpy(), atol=1e-3)


def test_ppo_learns_cartpole():
    import gymnasium as gym
    make = lambda: gym.make("CartPole-v1")
    _, log = ppo.train(make(), total_steps=30_720, evaluate=a2c.evaluate_gym(make, episodes=5), eval_every=30_720, seed=0)
    assert log[-1]["score"] >= 200, f"PPO greedy return {log[-1]['score']:.1f} after 30k steps"


def test_td3_learns_pendulum():
    import gymnasium as gym
    from algos.ddpg import evaluate_gym
    make = lambda: gym.make("Pendulum-v1")
    _, log = td3.train(make(), total_steps=10_000, evaluate=evaluate_gym(make, episodes=3), eval_every=10_000, seed=0)
    assert log[-1]["score"] > -400, f"TD3 return {log[-1]['score']:.1f} after 10k steps (random is about -1,200)"
