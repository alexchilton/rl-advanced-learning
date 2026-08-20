"""Base class for the finite MDPs in this repo.

A subclass builds the transition table `P` and hands it over. It does not
implement `step`. That is the point: the dynamics exist in exactly one place, so
`envs.solvers` and the agent's experience cannot disagree.

The interface is Gymnasium's, so every algorithm in `algos/` runs unchanged on
both these environments and on `gymnasium.make(...)`:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

`terminated` and `truncated` are separate on purpose. `terminated` means the
episode ended because the MDP says so, and the return really is over -- do not
bootstrap. `truncated` means we ran out of patience at `max_steps`, the MDP would
have continued, and you SHOULD bootstrap from the final observation. Collapsing
the two into one `done` flag is one of the most common and least visible bugs in
deep RL; `broken/truncation_as_termination.py` measures what it costs.
"""

from __future__ import annotations

import numpy as np

from envs import solvers

# A transition is (probability, next_state, reward, terminated).
Transition = tuple[float, int, float, bool]


class TabularEnv:
    """A finite MDP defined by a transition table."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        P: dict,
        start_distribution,
        max_steps: int = 100,
        obs_mode: str = "index",
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.P = P
        self.start_distribution = np.asarray(start_distribution, dtype=np.float64)
        self.max_steps = int(max_steps)
        self.obs_mode = obs_mode
        self.state: int | None = None
        self.t = 0
        self._rng = np.random.default_rng()
        self._validate()
        # Flattened once here so repeated exact solves do not re-walk the table.
        self._model = solvers.flatten(P, self.n_states, self.n_actions)

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = int(self._rng.choice(self.n_states, p=self.start_distribution))
        self.t = 0
        return self.observation(self.state), {"state": self.state, "steps": 0}

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("step() before reset()")
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside [0, {self.n_actions})")

        outcomes = self.P[self.state][action]
        probs = [t[0] for t in outcomes]
        choice = int(self._rng.choice(len(outcomes), p=probs))
        _, next_state, reward, terminated = outcomes[choice]

        self.state = int(next_state)
        self.t += 1
        # An episode is never both. Termination wins: the MDP ended before the
        # step budget mattered.
        truncated = bool(not terminated and self.t >= self.max_steps)
        info = {"state": self.state, "steps": self.t}
        return self.observation(self.state), float(reward), bool(terminated), truncated, info

    def observation(self, state: int):
        """Map a state index to what the agent sees.

        `index` for tabular methods, `onehot` for linear function approximation
        and small networks. Subclasses add their own modes.
        """
        if self.obs_mode == "index":
            return int(state)
        if self.obs_mode == "onehot":
            obs = np.zeros(self.n_states, dtype=np.float32)
            obs[state] = 1.0
            return obs
        raise ValueError(f"unknown obs_mode {self.obs_mode!r}")

    @property
    def obs_dim(self) -> int:
        """Length of the observation vector, for sizing a network's input layer."""
        obs = np.asarray(self.observation(0))
        return 1 if obs.ndim == 0 else int(obs.shape[0])

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    def true_q(self, gamma: float):
        """Exact optimal Q. Compare a learned Q against this, not against a curve."""
        _, Q, _ = solvers.value_iteration(self._model, self.n_states, self.n_actions, gamma)
        return Q

    def true_v(self, gamma: float):
        V, _, _ = solvers.value_iteration(self._model, self.n_states, self.n_actions, gamma)
        return V

    def optimal_policy(self, gamma: float):
        _, _, pi = solvers.value_iteration(self._model, self.n_states, self.n_actions, gamma)
        return pi

    def policy_return(self, policy, gamma: float) -> float:
        """Exact expected discounted return of a policy. No sampling, no noise."""
        return solvers.policy_return(
            self._model, self.n_states, self.n_actions, policy, gamma, self.start_distribution
        )

    def optimal_return(self, gamma: float) -> float:
        return self.policy_return(self.optimal_policy(gamma), gamma)

    # ------------------------------------------------------------------
    def _validate(self):
        if self.start_distribution.shape != (self.n_states,):
            raise ValueError(
                f"start_distribution must have shape ({self.n_states},), "
                f"got {self.start_distribution.shape}"
            )
        total = self.start_distribution.sum()
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"start_distribution sums to {total:.6f}, not 1")
        if (self.start_distribution < 0).any():
            raise ValueError("start_distribution has a negative entry")

        for s in range(self.n_states):
            if s not in self.P:
                raise ValueError(f"P is missing state {s}")
            for a in range(self.n_actions):
                if a not in self.P[s]:
                    raise ValueError(f"P[{s}] is missing action {a}")
                outcomes = self.P[s][a]
                if not outcomes:
                    raise ValueError(f"P[{s}][{a}] is empty")
                total = sum(t[0] for t in outcomes)
                if not np.isclose(total, 1.0, atol=1e-8):
                    raise ValueError(f"P[{s}][{a}] probabilities sum to {total:.6f}, not 1")
                for prob, s2, reward, terminated in outcomes:
                    if prob < 0:
                        raise ValueError(f"P[{s}][{a}] has a negative probability")
                    if not 0 <= s2 < self.n_states:
                        raise ValueError(f"P[{s}][{a}] goes to state {s2}, outside [0, {self.n_states})")
                    if not np.isfinite(reward):
                        raise ValueError(f"P[{s}][{a}] has non-finite reward {reward}")
                    if not isinstance(terminated, (bool, np.bool_)):
                        raise ValueError(f"P[{s}][{a}] terminated flag is {type(terminated)}, not bool")
