"""Exact solvers for finite MDPs.

These compute the answer. Not an estimate of the answer -- the answer.

Everything here reads the same transition table `P` that the environment samples
from when it steps, so the ground truth cannot drift away from the environment the
agent actually interacts with. When a learned Q disagrees with `value_iteration`,
the agent is wrong.

Transition table format, used everywhere in this repo:

    P[s][a] -> [(probability, next_state, reward, terminated), ...]

`terminated` is a property of the transition, not of the next state. That is the
Gymnasium convention, and it is also exactly the thing that half of all deep RL
bugs get wrong: a terminal transition must NOT bootstrap through `V[next_state]`.
Every backup below masks it, in one place -- `q_backup`. `broken/` shows what
happens when it does not.

The backups are vectorised. `P` is flattened once into parallel arrays and each
sweep is a handful of numpy operations, which matters because a discount of 0.999
needs tens of thousands of sweeps to converge to machine precision, and the same
loop written in Python takes tens of seconds per call.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

TOL = 1e-12
MAX_ITERS = 200_000


class FlatModel(NamedTuple):
    """`P` as parallel arrays, one entry per transition.

    `sa` indexes the flattened (state, action) pair, so a backup is one
    `np.bincount` over the transition list.
    """

    n_states: int
    n_actions: int
    sa: np.ndarray  # int64, flattened s * n_actions + a
    next_state: np.ndarray  # int64
    prob: np.ndarray  # float64
    reward: np.ndarray  # float64
    continues: np.ndarray  # float64, 0.0 on terminal transitions and 1.0 otherwise


def flatten(P, n_states: int, n_actions: int) -> FlatModel:
    """Flatten a transition table into arrays. Cheap; called once per solve."""
    sa, next_state, prob, reward, continues = [], [], [], [], []
    for s in range(n_states):
        for a in range(n_actions):
            for p, s2, r, terminated in P[s][a]:
                sa.append(s * n_actions + a)
                next_state.append(s2)
                prob.append(p)
                reward.append(r)
                continues.append(0.0 if terminated else 1.0)
    return FlatModel(
        n_states=n_states,
        n_actions=n_actions,
        sa=np.asarray(sa, dtype=np.int64),
        next_state=np.asarray(next_state, dtype=np.int64),
        prob=np.asarray(prob, dtype=np.float64),
        reward=np.asarray(reward, dtype=np.float64),
        continues=np.asarray(continues, dtype=np.float64),
    )


def _as_model(P, n_states, n_actions) -> FlatModel:
    return P if isinstance(P, FlatModel) else flatten(P, n_states, n_actions)


def q_backup(P, n_states, n_actions, V, gamma):
    """One Bellman backup: Q(s,a) = E[r + gamma * V(s') * (1 - terminated)].

    The `* continues` is the terminal masking. It is one multiplication, it is
    invisible when it is missing, and leaving it out is the most common bug in
    deep RL. See `tests/test_solvers.py::test_terminal_transitions_do_not_bootstrap`
    for exactly how much it changes the answer.
    """
    model = _as_model(P, n_states, n_actions)
    V = np.asarray(V, dtype=np.float64)
    backup = model.reward + gamma * model.continues * V[model.next_state]
    flat = np.bincount(
        model.sa, weights=model.prob * backup, minlength=model.n_states * model.n_actions
    )
    return flat.reshape(model.n_states, model.n_actions)


def value_iteration(P, n_states, n_actions, gamma, tol=TOL, max_iters=MAX_ITERS):
    """Optimal V, Q and greedy policy.

    Returns (V, Q, pi) where pi is a deterministic policy as an int array.
    Raises if it fails to converge, rather than returning a half-converged answer
    that would silently become a wrong "ground truth".
    """
    model = _as_model(P, n_states, n_actions)
    V = np.zeros(n_states)
    delta = np.inf
    for _ in range(max_iters):
        Q = q_backup(model, n_states, n_actions, V, gamma)
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta <= tol:
            Q = q_backup(model, n_states, n_actions, V, gamma)
            return V, Q, greedy_from_q(Q)
    raise RuntimeError(
        f"value iteration did not converge in {max_iters} sweeps "
        f"(last delta {delta:.3e}, gamma={gamma}). "
        "gamma=1 on a task that need not terminate will do this."
    )


def policy_evaluation(P, n_states, n_actions, policy, gamma, tol=TOL, max_iters=MAX_ITERS):
    """Exact V for a given policy.

    `policy` is either an int array of shape (n_states,) for a deterministic
    policy, or a float array of shape (n_states, n_actions) of action
    probabilities. `epsilon_greedy_policy` produces the latter, which is what you
    want when asking what SARSA converges to.
    """
    model = _as_model(P, n_states, n_actions)
    pi = as_stochastic(policy, n_states, n_actions)
    V = np.zeros(n_states)
    delta = np.inf
    for _ in range(max_iters):
        Q = q_backup(model, n_states, n_actions, V, gamma)
        V_new = (pi * Q).sum(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta <= tol:
            return V
    raise RuntimeError(
        f"policy evaluation did not converge in {max_iters} sweeps "
        f"(last delta {delta:.3e}, gamma={gamma})"
    )


def policy_iteration(P, n_states, n_actions, gamma, max_iters=1_000, **kwargs):
    """Optimal policy by alternating exact evaluation and greedy improvement.

    Agrees with `value_iteration` on every environment here. Kept because seeing
    two independent methods agree is a real check on both, and because policy
    iteration is the skeleton that every actor-critic method approximates.
    """
    model = _as_model(P, n_states, n_actions)
    pi = np.zeros(n_states, dtype=np.int64)
    for _ in range(max_iters):
        V = policy_evaluation(model, n_states, n_actions, pi, gamma, **kwargs)
        Q = q_backup(model, n_states, n_actions, V, gamma)
        pi_new = greedy_from_q(Q)
        if np.array_equal(pi_new, pi):
            return V, Q, pi
        pi = pi_new
    raise RuntimeError(f"policy iteration did not converge in {max_iters} improvements")


def policy_return(P, n_states, n_actions, policy, gamma, start_distribution, **kwargs):
    """Exact expected discounted return of a policy from the start distribution.

    Use this to score an agent instead of averaging sampled episodes. It has no
    variance, so a difference between two policies is a real difference and not
    seed noise. Sampled evaluation is still worth running -- it catches bugs in
    the agent's own action selection that an exact evaluation of its policy
    cannot -- but when the question is "is A better than B", ask this.
    """
    V = policy_evaluation(P, n_states, n_actions, policy, gamma, **kwargs)
    return float(np.dot(np.asarray(start_distribution, dtype=np.float64), V))


def as_stochastic(policy, n_states, n_actions):
    """Coerce a deterministic or stochastic policy to shape (n_states, n_actions)."""
    policy = np.asarray(policy)
    if policy.ndim == 1:
        if policy.shape != (n_states,):
            raise ValueError(
                f"deterministic policy must have shape ({n_states},), got {policy.shape}"
            )
        pi = np.zeros((n_states, n_actions))
        pi[np.arange(n_states), policy.astype(np.int64)] = 1.0
        return pi
    if policy.shape != (n_states, n_actions):
        raise ValueError(
            f"stochastic policy must have shape ({n_states}, {n_actions}), got {policy.shape}"
        )
    sums = policy.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-8):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(f"policy rows must sum to 1; row {bad} sums to {sums[bad]:.6f}")
    return policy.astype(np.float64)


def epsilon_greedy_policy(Q, epsilon):
    """The stochastic policy an epsilon-greedy agent actually follows.

    SARSA converges to the value of THIS policy, not to the value of the greedy
    policy. That difference is the entire on-policy/off-policy story, and
    `experiments/cliff_sarsa_vs_qlearning.py` evaluates against exactly this.
    """
    Q = np.asarray(Q)
    n_states, n_actions = Q.shape
    pi = np.full((n_states, n_actions), epsilon / n_actions)
    pi[np.arange(n_states), Q.argmax(axis=1)] += 1.0 - epsilon
    return pi


def greedy_from_q(Q):
    """Deterministic greedy policy. Ties break to the lowest action index."""
    return np.asarray(Q).argmax(axis=1).astype(np.int64)
