"""C51, Noisy DQN and TRPO, checked where the answer is known exactly."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos import c51_dqn, noisy_dqn, trpo
from algos.a2c import ActorCritic, evaluate_gym
from diagnostics.probe_envs import CoinFlip, run_probes

torch.set_num_threads(1)
PROBE_SETTINGS = dict(learning_starts=200, buffer_size=5_000, target_update=100)


@pytest.mark.parametrize("module, kwargs", [(c51_dqn, dict(v_min=-2.0, v_max=2.0)), (noisy_dqn, {})],
                         ids=["c51_dqn", "noisy_dqn"])
def test_rainbow_parts_pass_every_probe(module, kwargs):
    results = run_probes(module.train, **PROBE_SETTINGS, **kwargs)
    failed = [(r["probe"], round(r["worst_error"], 3)) for r in results if not r["passed"]]
    assert not failed, f"{module.__name__} failed probes: {failed}"


def test_c51_projection_keeps_all_mass_and_collapses_on_terminal():
    """Atoms landing exactly on a grid point must not lose their mass (lower == upper)."""
    support = torch.linspace(-2, 2, 41)
    probs = torch.softmax(torch.randn(5, 41), -1)
    rewards = torch.tensor([0.0, 0.1, -0.3, 1.0, 0.5])
    done = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    target = c51_dqn.project(probs, rewards, done, 1.0, support)
    assert torch.allclose(target.sum(1), torch.ones(5), atol=1e-5)
    # terminal row: all mass on the single atom at r = 1.0
    assert float(target[3, int(torch.argmin((support - 1.0).abs()))]) == pytest.approx(1.0, abs=1e-5)
    # gamma = 1, r = 0: the distribution is unchanged
    assert torch.allclose(target[0], probs[0], atol=1e-5)


def test_c51_recovers_the_coin_flip_return_distribution():
    """Mean -0.5 is what DQN learns; C51 must put 1/4 at +1 and 3/4 at -1."""
    env = CoinFlip()
    net, _ = c51_dqn.train(env, n_steps=3_000, v_min=-2.0, v_max=2.0, n_atoms=41, **PROBE_SETTINGS)
    with torch.no_grad():
        p = net.distribution(torch.zeros(1, 1))[0, 0]
    support = net.support
    at = lambda v: float(p[int(torch.argmin((support - v).abs()))])
    assert at(1.0) == pytest.approx(0.25, abs=0.06)
    assert at(-1.0) == pytest.approx(0.75, abs=0.06)
    assert at(1.0) + at(-1.0) > 0.95


def test_noisy_network_is_random_in_training_and_fixed_in_evaluation():
    net = noisy_dqn.NoisyQNetwork(4, 2)
    x = torch.randn(8, 4)
    a = net(x).detach(); net.reset_noise(); b = net(x).detach()
    assert not torch.allclose(a, b), "reset_noise must change the training-mode output"
    net.eval()
    c = net(x).detach(); net.reset_noise()
    assert torch.allclose(c, net(x).detach()), "evaluation must use the mean weights only"


def test_conjugate_gradient_solves_a_positive_definite_system():
    torch.manual_seed(0)
    A = torch.randn(20, 20)
    A = A @ A.T + 20 * torch.eye(20)
    b = torch.randn(20)
    x = trpo.conjugate_gradient(lambda v: A @ v, b, iters=20)
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-4)


def test_fisher_vector_product_matches_the_explicit_kl_hessian():
    """Without create_graph=True the second derivative is zero and TRPO divides by ~0."""
    torch.manual_seed(0)
    model = ActorCritic(4, 2, hidden=4)
    params = list(model.actor.parameters())
    obs = torch.randn(32, 4)
    old = torch.distributions.Categorical(logits=model.actor(obs).detach())
    kl = lambda: torch.distributions.kl_divergence(old, model.distribution(obs)).mean()
    n = sum(p.numel() for p in params)
    grad = trpo.flat_grad(kl(), params, create_graph=True)
    hessian = torch.stack([trpo.flat_grad(grad[i], params, retain_graph=True) for i in range(n)])
    v = torch.randn(n)
    fvp = trpo.flat_grad(trpo.flat_grad(kl(), params, create_graph=True) @ v, params)
    assert torch.allclose(fvp, hessian @ v, atol=1e-5)
    assert float((hessian @ v).abs().max()) > 1e-3


def test_trpo_learns_cartpole_inside_its_trust_region():
    make = lambda: gym.make("CartPole-v1")
    _, log = trpo.train(make(), total_steps=24_576, evaluate=evaluate_gym(make), eval_every=8_192, seed=0)
    assert all(row["kl"] <= trpo.MAX_KL for row in log)
    assert log[-1]["score"] >= 400
