"""Residual RL on the Smart Buildings simulator: the organisers' PID runs the building, PPO learns a bounded correction.

The wrapper turns the starter kit's hybrid environment into a plain gymnasium env with one continuous action vector:

    continuous = clip(pid_continuous + SCALE * a, -1, 1)      a in [-1, 1]^n, from PPO
    discrete   = pid_discrete                                  on/off switching stays with the PID

`warmup_steps` lets the PID drive alone for that many steps after reset, so an episode can start later in the
30-day weather window (used to evaluate on days the policy never trained on).
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

import os
# The official starter kit (github.com/smart-buildings-2026/starter-kit), installed with `pip install -e`.
KIT = Path(os.environ.get("SB_KIT", Path(__file__).resolve().parent / "starter-kit"))
SCENARIOS = KIT / "smart_buildings" / "training_scenarios"


def _kit():
    import os, sys
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    os.chdir(KIT)                         # the kit resolves weather and building files relative to its root
    from smart_buildings import evaluate as E
    return E


def make_base(building="building_train_8", weather="atlanta_1", policy="pid_water"):
    """The official environment and the official baseline policy for one scenario."""
    E = _kit()
    d = SCENARIOS / building / weather
    env = E.load_hybrid_environment(d / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0        # as evaluate.py sets it
    meta = E.create_building_meta_data(env, d)
    spec = env.action_spec()
    pol = E.initialize_policy(E.POLICY_MAP[policy], meta, spec)
    return E, env, meta, spec, pol


class ResidualPID(gym.Env):
    metadata = {}

    def __init__(self, building="building_train_8", weather="atlanta_1", episode_steps=576, warmup_steps=0,
                 scale=0.3, reward_scale=10.0):
        self.E, self.env, self.meta, self.spec, self.pid = make_base(building, weather)
        self.cfg = (building, weather)
        n = int(self.spec["continuous_action"].shape[0])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.episode_steps, self.warmup_steps, self.scale, self.reward_scale = episode_steps, warmup_steps, scale, reward_scale

    def _pid_action(self):
        return self.E.validate_action(self.E.get_policy_action(self.pid, self.obs), self.spec)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pid = self.E.initialize_policy(self.E.POLICY_MAP["pid_water"], self.meta, self.spec)   # fresh PID integrators
        self.obs, _ = self.env.reset()
        for _ in range(self.warmup_steps):
            self.obs, *_ = self.env.step(self._pid_action())
        self.t, self.raw = 0, []
        return self.obs.astype(np.float32), {}

    def step(self, a):
        base = self._pid_action()
        cont = np.clip(base["continuous_action"] + self.scale * np.asarray(a, np.float32), -1, 1).astype(np.float32)
        self.obs, r, term, trunc, info = self.env.step({"continuous_action": cont, "discrete_action": base["discrete_action"]})
        self.t += 1; self.raw.append(float(r))
        trunc = trunc or self.t >= self.episode_steps
        return self.obs.astype(np.float32), float(r) * self.reward_scale, bool(term), bool(trunc), {"raw_reward": float(r)}


def load_ppo(path, vec_env, net_arch=(128, 128)):
    """PPO.load fails under torch 2.14.1: it hands torch.load a zip member stream, which torch can no longer read
    ("PytorchStreamReader failed reading file .data/serialization_id"). The same bytes load fine from memory, so
    rebuild the model and load the policy weights from a BytesIO."""
    import io, zipfile, torch
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", vec_env, policy_kwargs=dict(net_arch=list(net_arch)), device="cpu")
    model.policy.load_state_dict(torch.load(io.BytesIO(zipfile.ZipFile(path).read("policy.pth")), map_location="cpu"))
    return model
