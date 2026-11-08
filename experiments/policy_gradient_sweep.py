"""Tier 3 sweeps: discrete policy gradient on CartPole, continuous control on Pendulum.

    python experiments/policy_gradient_sweep.py GROUP [--resume]
    python experiments/policy_gradient_sweep.py report

GROUP is one of:

    discrete      REINFORCE, REINFORCE + baseline, A2C, PPO; CartPole-v1, 100,000 steps
    gae           PPO with lambda in {0, 0.5, 0.9, 0.95, 1}: the bias-variance dial
    continuous    DDPG, TD3, SAC (learned alpha); Pendulum-v1, 20,000 steps
    td3_ablation  TD3 with each of its three fixes removed, and with all three removed
    sac_alpha     SAC with alpha fixed at 0.01, 0.1, 1 against the learned alpha

5 seeds per configuration (3 for gae). One JSON per run in
experiments/results/policy_gradient/, written when the run ends, so `--resume`
skips finished runs and a crash costs one run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import time

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "policy_gradient"

GROUPS = {
    "discrete": dict(env="CartPole-v1", steps=100_000, seeds=range(5), configs={
        "REINFORCE": ("reinforce", {}), "REINFORCE + baseline": ("reinforce_baseline", {}),
        "A2C": ("a2c", {}), "PPO": ("ppo", {})}),
    "gae": dict(env="CartPole-v1", steps=100_000, seeds=range(3), configs={
        f"lambda={lam}": ("ppo", dict(lam=lam)) for lam in (0.0, 0.5, 0.9, 0.95, 1.0)}),
    "continuous": dict(env="Pendulum-v1", steps=20_000, seeds=range(5), configs={
        "DDPG": ("ddpg", {}), "TD3": ("td3", {}), "SAC": ("sac", {})}),
    "td3_ablation": dict(env="Pendulum-v1", steps=20_000, seeds=range(5), configs={
        "TD3 (all three)": ("td3", {}), "no clipped double Q": ("td3", dict(n_critics=1)),
        "no delay": ("td3", dict(policy_delay=1)), "no target smoothing": ("td3", dict(target_noise=0.0)),
        "none of them": ("td3", dict(n_critics=1, policy_delay=1, target_noise=0.0))}),
    "sac_alpha": dict(env="Pendulum-v1", steps=20_000, seeds=range(5), configs={
        "learned alpha": ("sac", {}), "alpha=0.01": ("sac", dict(alpha=0.01)),
        "alpha=0.1": ("sac", dict(alpha=0.1)), "alpha=1": ("sac", dict(alpha=1.0))}),
}


def run_one(env_id, steps, module_name, kwargs, seed):
    import gymnasium as gym
    import torch

    torch.set_num_threads(1)
    module = importlib.import_module(f"algos.{module_name}")
    make = lambda: gym.make(env_id)
    if env_id == "Pendulum-v1":
        from algos.ddpg import evaluate_gym
        evaluate, eval_every = evaluate_gym(make), 2_000
    else:
        evaluate, eval_every = module.evaluate_gym(make), 10_000
    start = time.time()
    _, log = module.train(make(), total_steps=steps, evaluate=evaluate, eval_every=eval_every, seed=seed, **kwargs)
    return dict(module=module_name, kwargs=kwargs, seed=seed, env=env_id, log=log, seconds=time.time() - start)


def sweep(group, resume):
    spec = GROUPS[group]
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, (module, kwargs) in spec["configs"].items():
        for seed in spec["seeds"]:
            slug = name.replace(" ", "_").replace("+", "and").replace("=", "").replace("(", "").replace(")", "")
            path = RESULTS / f"{group}__{slug}__seed{seed}.json"
            if resume and path.exists():
                print(f"skip  {path.name}", flush=True)
                continue
            result = run_one(spec["env"], spec["steps"], module, kwargs, seed)
            result["name"] = name
            path.write_text(json.dumps(result))
            print(f"done  {path.name}  final {result['log'][-1]['score']:.1f}  {result['seconds']:.0f}s", flush=True)


def load(group):
    rows = {}
    for path in sorted(RESULTS.glob(f"{group}__*.json")):
        result = json.loads(path.read_text())
        rows.setdefault(result["name"], []).append(result)
    order = list(GROUPS[group]["configs"])
    return {name: rows[name] for name in order if name in rows}


def report():
    for group in GROUPS:
        rows = load(group)
        if not rows:
            continue
        spec = GROUPS[group]
        print(f"\n{group}: {spec['env']}, {spec['steps']:,} steps, median [min, max] over seeds\n")
        print(f"{'':>22} {'seeds':>5} {'final':>24} {'mean over curve':>16}")
        for name, results in rows.items():
            final = np.array([r["log"][-1]["score"] for r in results])
            area = np.median([np.mean([e["score"] for e in r["log"]]) for r in results])
            print(f"{name:>22} {len(results):>5} {f'{np.median(final):7.1f} [{final.min():7.1f}, {final.max():7.1f}]':>24} "
                  f"{area:>16.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=(*GROUPS, "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.what == "report":
        report()
    else:
        sweep(args.what, args.resume)
