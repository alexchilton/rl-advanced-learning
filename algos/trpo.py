"""Trust Region Policy Optimization (Schulman et al. 2015). PyTorch, discrete actions.

    python algos/trpo.py

A policy-gradient step of fixed length in parameter space can be a tiny change
to the policy or a catastrophic one: it depends on the parametrisation. TRPO
fixes the step's length in the space that matters, the policy's own
distributions, and solves

    maximise   L(theta) = E[ pi_theta(a|s) / pi_old(a|s) * A ]
    subject to E_s[ KL(pi_old(.|s) || pi_theta(.|s)) ] <= delta

to second order: L is linearised (gradient g), the KL is quadratic with the
Fisher matrix F as its Hessian. The solution is the natural gradient direction
x = F^-1 g scaled so that 0.5 x^T F x = delta:

    step = sqrt(2 delta / x^T F x) * x

F is never formed. `fisher_vector_product` computes F v by differentiating the
KL twice (Pearlmutter's trick), and `conjugate_gradient` solves F x = g from
those products alone. A backtracking line search then halves the step until
the measured KL is within delta AND the surrogate actually improved: the
quadratic model is only a model.

The rollout, GAE and network are `algos/a2c.py`'s; the critic is fitted with
Adam, as in the paper's experiments. PPO (`algos/ppo.py`) replaces all of this
with one clamp.
"""

from __future__ import annotations

import numpy as np
import torch

from algos.a2c import ActorCritic, Rollout, collect, evaluate_gym, gae

GAMMA = 0.99
LAMBDA = 0.95
N_STEPS = 1_024
MAX_KL = 0.01
CG_ITERS = 10
DAMPING = 0.1
BACKTRACK_STEPS = 10
VALUE_LEARNING_RATE = 1e-3
VALUE_EPOCHS = 10
MINIBATCH = 64
HIDDEN = 64


def flat_grad(y, params, **kwargs):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(y, params, **kwargs)])


def set_flat(params, flat):
    i = 0
    for p in params:
        p.data.copy_(flat[i:i + p.numel()].view_as(p))
        i += p.numel()


def conjugate_gradient(matvec, b, iters=CG_ITERS, tol=1e-10):
    """Solve A x = b for symmetric positive definite A, given only x -> A x."""
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def train(env, total_steps=100_000, gamma=GAMMA, lam=LAMBDA, n_steps=N_STEPS, max_kl=MAX_KL, cg_iters=CG_ITERS,
          damping=DAMPING, backtrack_steps=BACKTRACK_STEPS, value_learning_rate=VALUE_LEARNING_RATE,
          value_epochs=VALUE_EPOCHS, minibatch=MINIBATCH, hidden=HIDDEN, seed=0, evaluate=None, eval_every=10_000):
    """Returns (model, log). Log rows: step, score, KL of the accepted step, backtracks,
    entropy, explained variance. backtracks = backtrack_steps means no step was accepted."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden)
    params = list(model.actor.parameters())
    value_opt = torch.optim.Adam(model.critic.parameters(), lr=value_learning_rate)
    rollout = Rollout(env, seed)
    log, step, next_eval = [], 0, eval_every
    while step < total_steps:
        batch = collect(rollout, model, n_steps)
        step += n_steps
        advantages, returns = gae(batch, gamma, lam)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs, actions, old_logp = batch["obs"], batch["actions"], batch["logp"]
        with torch.no_grad():
            old_logits = model.actor(obs)
        old_dist = torch.distributions.Categorical(logits=old_logits)

        def surrogate():
            return (torch.exp(model.distribution(obs).log_prob(actions) - old_logp) * adv).mean()

        def mean_kl():
            return torch.distributions.kl_divergence(old_dist, model.distribution(obs)).mean()

        # ---- natural gradient direction: solve F x = g with conjugate gradient
        g = flat_grad(surrogate(), params)

        def fisher_vector_product(v):
            kl_grad = flat_grad(mean_kl(), params, create_graph=True)
            return flat_grad(kl_grad @ v, params) + damping * v

        x = conjugate_gradient(fisher_vector_product, g, cg_iters)
        step_size = torch.sqrt(2 * max_kl / (x @ fisher_vector_product(x) + 1e-12))
        full_step = step_size * x

        # ---- backtracking line search: KL within delta and the surrogate improved
        old_params = torch.cat([p.data.reshape(-1) for p in params])
        with torch.no_grad():
            old_surrogate = float(surrogate())
        accepted_kl, backtracks = 0.0, backtrack_steps
        for k in range(backtrack_steps):
            set_flat(params, old_params + 0.5 ** k * full_step)
            with torch.no_grad():
                kl, improvement = float(mean_kl()), float(surrogate()) - old_surrogate
            if kl <= max_kl and improvement > 0:
                accepted_kl, backtracks = kl, k
                break
        else:
            set_flat(params, old_params)

        # ---- critic: plain regression on the GAE returns
        for _ in range(value_epochs):
            for index in np.array_split(rng.permutation(n_steps), n_steps // minibatch):
                index = torch.as_tensor(index)
                value_loss = ((model.value(obs[index]) - returns[index]) ** 2).mean()
                value_opt.zero_grad(); value_loss.backward(); value_opt.step()

        if evaluate is not None and step >= next_eval:
            next_eval += eval_every
            with torch.no_grad():
                values = model.value(obs)
                entropy = float(model.distribution(obs).entropy().mean())
            explained = float(1 - (returns - values).var() / (returns.var() + 1e-8))
            log.append(dict(step=step, score=float(evaluate(model)), kl=accepted_kl, backtracks=backtracks,
                            entropy=entropy, explained_variance=explained))
    return model, log


def main():
    import gymnasium as gym

    torch.set_num_threads(1)
    make = lambda: gym.make("CartPole-v1")
    _, log = train(make(), total_steps=100_000, evaluate=evaluate_gym(make), seed=0)
    print(f"CartPole-v1, TRPO, 100,000 steps, seed 0, max KL {MAX_KL}\n")
    print(f"{'step':>8} {'greedy':>8} {'KL':>8} {'backtracks':>10} {'entropy':>8} {'critic EV':>10}")
    for row in log:
        print(f"{row['step']:>8,} {row['score']:>8.1f} {row['kl']:>8.4f} {row['backtracks']:>10} "
              f"{row['entropy']:>8.3f} {row['explained_variance']:>10.2f}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - KL of the accepted step far above max_kl: the line search is checking the
#   wrong distribution (old_dist must be detached logits from BEFORE the step).
# - backtracks = BACKTRACK_STEPS every update: the policy never moves. The sign
#   of g, or a Fisher product with the wrong curvature (no damping, or the KL
#   taken as KL(new || old) with both sides differentiated) are the usual causes.
# - Conjugate gradient divides by ~0: x^T F x vanishes when the KL gradient is
#   taken without create_graph=True, so the second derivative is zero.
# - Critic and actor share parameters: the trust region then constrains the
#   critic's update too, and the Fisher includes it. `ActorCritic` keeps them apart.
