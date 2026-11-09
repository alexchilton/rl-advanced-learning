"""Preference-tier sweeps on the toy task with a known true reward.

    python experiments/preference_sweep.py GROUP [--resume]
    python experiments/preference_sweep.py all [--resume]

GROUP is one of:

    rm_accuracy   reward-model accuracy against the true reward, 100 to 10,000 pairs
    estimators    RLHF with mean / RLOO / GRPO advantages, reward model from 3,000 pairs
    overopt       RLHF with beta from 0 to 0.3 against a weaker reward model (1,000 pairs):
                  proxy reward, true reward and KL over 600 steps (Gao et al. 2022)
    labelers      reward models from a length-biased labeler and from a rule-following
                  AI labeler, then RLHF on each (Singhal et al. 2023; Bai et al. 2022)
    dpo           DPO with beta 0.03 / 0.1 / 0.3 on the same 3,000 human pairs

One JSON per run in experiments/results/preference/; reward models and the SFT
policy are cached in data/runs/preference/ so a resumed sweep does not retrain them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "preference"
CACHE = ROOT / "data" / "runs" / "preference"
GROUPS = ("rm_accuracy", "estimators", "overopt", "labelers", "dpo")


def sft_policy():
    import torch

    from algos import sequence_policy as sp

    path = CACHE / "sft.pt"
    policy = sp.SequencePolicy()
    if path.exists():
        policy.load_state_dict(torch.load(path))
    else:
        policy = sp.sft()
        torch.save(policy.state_dict(), path)
    return policy


def reward_model(labeler, n_pairs, seed=0):
    import torch

    from algos import reward_model as rm
    from envs.preference_toy import preference_data

    path = CACHE / f"rm_{labeler}_{n_pairs}_seed{seed}.pt"
    model = rm.RewardModel()
    if path.exists():
        model.load_state_dict(torch.load(path))
    else:
        model = rm.train(preference_data(np.random.default_rng(seed), n_pairs, labeler=labeler), seed=seed)
        torch.save(model.state_dict(), path)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save(name, payload, resume):
    path = RESULTS / f"{name}.json"
    if resume and path.exists():
        print(f"skip {path.name}", flush=True)
        return False
    start = time.time()
    payload = payload()
    payload["seconds"] = time.time() - start
    path.write_text(json.dumps(payload))
    print(f"done {path.name} ({payload['seconds']:.0f}s)", flush=True)
    return True


def run(group, resume):
    import torch

    from algos import dpo, reward_model as rm, rlhf
    from algos.sequence_policy import evaluate
    from envs.preference_toy import preference_data, score

    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    def fresh():
        return sft_policy()

    if group == "rm_accuracy":
        for n in (100, 300, 1_000, 3_000, 10_000):
            save(f"rm_accuracy__{n}", lambda n=n: dict(n_pairs=n, accuracy=rm.accuracy(reward_model("human", n))), resume)

    if group == "estimators":
        model = reward_model("human", 3_000)
        for estimator in ("mean", "rloo", "grpo"):
            for seed in range(3):
                save(f"estimators__{estimator}__seed{seed}",
                     lambda e=estimator, s=seed: dict(estimator=e, seed=s, sft=evaluate(fresh()),
                                                      log=rlhf.train(fresh(), model, estimator=e, n_steps=300, seed=s)),
                     resume)

    if group == "overopt":
        model = reward_model("human", 1_000)
        for beta in (0.0, 0.01, 0.03, 0.1, 0.3):
            for seed in range(2):
                save(f"overopt__beta{beta}__seed{seed}",
                     lambda b=beta, s=seed: dict(beta=b, seed=s,
                                                 log=rlhf.train(fresh(), model, beta=b, n_steps=600, seed=s, eval_every=25)),
                     resume)

    if group == "labelers":
        for labeler in ("human", "length_bias", "constitution"):
            def payload(labeler=labeler):
                rng = np.random.default_rng(5)
                agree = []
                for pair in preference_data(rng, 2_000, labeler="human", temperature=1e-6):
                    p, a, b = pair
                    agree.append(score(labeler, p, a) > score(labeler, p, b) if score(labeler, p, a) != score(labeler, p, b) else None)
                agree = [x for x in agree if x is not None]
                model = reward_model(labeler, 3_000)
                return dict(labeler=labeler, agreement_with_true=float(np.mean(agree)),
                            rm_accuracy_vs_true=rm.accuracy(model),
                            log=rlhf.train(fresh(), model, n_steps=300, seed=0))
            save(f"labelers__{labeler}", payload, resume)

    if group == "dpo":
        pairs = preference_data(np.random.default_rng(0), 3_000)
        for beta in (0.03, 0.1, 0.3):
            for seed in range(2):
                save(f"dpo__beta{beta}__seed{seed}",
                     lambda b=beta, s=seed: dict(beta=b, seed=s, log=dpo.train(fresh(), pairs, beta=b, n_epochs=4, seed=s,
                                                                                 eval_every=25)),
                     resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=(*GROUPS, "all"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for group in (GROUPS if args.group == "all" else (args.group,)):
        run(group, args.resume)
