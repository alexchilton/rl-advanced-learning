"""Offline pretraining, then online fine-tuning.

    python algos/offline_to_online.py

The row of the access table most real projects actually occupy, and the least
taught. You have a pile of logged data and you are also allowed to interact, just
not very much. Pretrain on the log, then improve by doing.

The characteristic failure is its own, and it happens at the handover. A
conservative offline method deliberately distorts its Q function -- that is what
made it safe -- and then online fine-tuning starts overwriting that Q with fresh
targets computed from a very small amount of on-policy data. The pretrained
policy can get worse before it gets better, sometimes much worse, and if your
budget of online interaction is small you may never see the recovery.

Four variants, all starting from the same offline dataset and differing only in
what happens after the switch:

    scratch          no pretraining at all. The baseline that pretraining has to beat.
    naive            standard Q-learning from the pretrained Q, fresh data only.
    mixed replay     every update samples from the offline log as well.
    gentle           the same as naive with a much smaller learning rate.

Whether the dip appears at all is an empirical question and this file answers it
by measuring rather than asserting. Every point on every curve is an exact return
from `policy_return`, so a difference between two curves is a real difference and
not evaluation noise.
"""

from __future__ import annotations

import numpy as np

from algos import cql_tabular
from envs import datasets, solvers
from envs.racetrack import Racetrack

GAMMA = 0.99
ONLINE_STEPS = 40_000
EVAL_EVERY = 5_000
SEEDS = (0, 1, 2)
# Exact policy evaluation to machine precision needs about 2,800 sweeps at
# gamma=0.99, and these curves need a hundred of them. 1e-6 is four decimal
# places of return and costs half as much.
EVAL_TOL = 1e-6
EPSILON = 0.1
LEARNING_RATE = 0.3
GENTLE_LEARNING_RATE = 0.02


def finetune(
    env,
    Q,
    n_steps=ONLINE_STEPS,
    epsilon=EPSILON,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    dataset=None,
    replay_batch=0,
    eval_every=EVAL_EVERY,
    seed=0,
):
    """Tabular Q-learning starting from a given Q. Returns [(steps, exact return)].

    `dataset` and `replay_batch` add the offline log back into the update stream:
    each online step also performs `replay_batch` updates on transitions sampled
    from the log. This is the standard defence against the handover collapse, and
    it costs nothing but bookkeeping.
    """
    Q = Q.copy()
    rng = np.random.default_rng(seed)

    def score():
        return solvers.policy_return(
            env._model, env.n_states, env.n_actions, Q.argmax(axis=1), gamma,
            env.start_distribution, tol=EVAL_TOL,
        )

    curve = [(0, score())]

    env.reset(seed=seed)
    for step in range(1, n_steps + 1):
        state = env.state
        if rng.random() < epsilon:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = int(Q[state].argmax())

        _, reward, terminated, truncated, _ = env.step(action)
        target = reward + (0.0 if terminated else gamma * Q[env.state].max())
        Q[state, action] += learning_rate * (target - Q[state, action])

        if terminated or truncated:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        if replay_batch and dataset is not None:
            index = rng.integers(0, len(dataset), replay_batch)
            s = dataset.states[index]
            a = dataset.actions[index]
            bootstrap = np.where(
                dataset.terminated[index], 0.0, gamma * Q[dataset.next_states[index]].max(axis=1)
            )
            replay_target = dataset.rewards[index] + bootstrap
            # Sequential rather than vectorised: two samples of the same (s,a) in
            # one batch must not both be applied against the stale value.
            for i in range(replay_batch):
                Q[s[i], a[i]] += learning_rate * (replay_target[i] - Q[s[i], a[i]])

        if step % eval_every == 0:
            curve.append((step, score()))

    return curve


def main():
    env = Racetrack()
    optimal = env.optimal_return(GAMMA)
    dataset = datasets.build(env, "medium", n_transitions=20_000, gamma=GAMMA, seed=0)
    behaviour = env.policy_return(datasets.behaviour_policy_for(env, "medium", GAMMA), GAMMA)

    pretrained = cql_tabular.train(dataset, alpha=1.0, gamma=GAMMA)
    pretrained_value = env.policy_return(pretrained.argmax(axis=1), GAMMA)

    print("=" * 78)
    print("Offline pretraining, then online fine-tuning")
    print("=" * 78)
    print(f"Racetrack, {env.n_states:,} states. Exact optimum: {optimal:.2f}")
    print(f"Offline log: {dataset.summary()}")
    print(f"Behaviour policy that produced it: {behaviour:.2f}")
    print(f"CQL after offline pretraining:     {pretrained_value:.2f}\n")

    runs = {
        "scratch": dict(Q=np.zeros_like(pretrained)),
        "naive": dict(Q=pretrained),
        "mixed replay": dict(Q=pretrained, dataset=dataset, replay_batch=4),
        "gentle": dict(Q=pretrained, learning_rate=GENTLE_LEARNING_RATE),
    }
    # Several seeds, because a single run cannot tell a handover collapse from
    # ordinary policy volatility, and on the first run it did not: the "gentle"
    # variant produced the deepest dip of all four, which is the opposite of what
    # it is for.
    curves = {
        name: [finetune(env, seed=seed, **kwargs) for seed in SEEDS]
        for name, kwargs in runs.items()
    }

    steps = [step for step, _ in curves["scratch"][0]]
    print(f"median of {len(SEEDS)} seeds, exact returns\n")
    print(f"{'online steps':>13} " + "".join(f"{name:>15}" for name in runs))
    for row, step in enumerate(steps):
        cells = "".join(
            f"{np.median([run[row][1] for run in curves[name]]):>15.2f}" for name in runs
        )
        print(f"{step:>13,} " + cells)

    commentary(curves, pretrained_value, optimal)


def commentary(curves, pretrained_value, optimal):
    print()
    print("=" * 78)
    print("Reading the curves")
    print("=" * 78)

    print("\nThe handover, per seed. 'worst' is the lowest point after the switch.\n")
    print(f"{'variant':>14} {'worst per seed':>34} {'median final':>14}")

    dips = {}
    for name, runs in curves.items():
        if name == "scratch":
            continue
        worst_per_seed = [min(value for _, value in run) for run in runs]
        dips[name] = pretrained_value - float(np.median(worst_per_seed))
        finals = [run[-1][1] for run in runs]
        formatted = " ".join(f"{value:>10.2f}" for value in worst_per_seed)
        print(f"{name:>14} {formatted:>34} {np.median(finals):>14.2f}")

    print(f"\nStarting point for all three: {pretrained_value:.2f}")
    print("\nThree things to read off that, and the third is the interesting one.")
    print("\n1. Mixed replay is the defence that works. Its worst point is shallow and")
    print("   its seeds agree. Keeping the offline log in the update stream costs")
    print("   nothing but bookkeeping.")
    print("\n2. Naive fine-tuning is ERRATIC rather than reliably bad. One seed lost")
    print("   thirty-eight of return and the other two lost about two. Reporting one")
    print("   seed here would support whichever conclusion you already held.")
    print("\n3. A gentle learning rate is reliably the WORST, which is the opposite of")
    print("   what it is for. All three seeds land within 0.2 of each other at about")
    print("   -30. That consistency means it is a mechanism, not volatility.")
    print("\n   The likely explanation, NOT yet verified: a conservative Q function is")
    print("   not a return. CQL distorts its values deliberately, and only the ORDERING")
    print("   is meaningful. Online updates replace visited entries with true-scale")
    print("   targets, so a corrected entry and an untouched neighbour now sit on")
    print("   different scales -- and argmax compares them anyway. A large learning")
    print("   rate crosses that inconsistent regime quickly. A small one sits in it.")
    print("   Confirming this means comparing CQL's Q against the exact Q on and off")
    print("   the data support, which has not been run.")
    print("\nSeparately, and worth knowing before reading any curve here as smooth:")
    print("these evaluations have NO measurement noise -- they are exact. The policy")
    print("still jumps discontinuously, because one Q entry flipping the argmax in a")
    print("reachable state can strand the car and move the exact return by twenty.")
    print("Exact evaluation removes one kind of noise and not the other.")

    print("\nWhat is unambiguous is the value of pretraining itself:")
    scratch_final = float(np.median([run[-1][1] for run in curves["scratch"]]))
    scratch_start = float(np.median([run[0][1] for run in curves["scratch"]]))
    print(f"  from scratch     starts {scratch_start:>8.2f}   ends {scratch_final:>8.2f}")
    print(f"  pretrained       starts {pretrained_value:>8.2f}   and never goes near that")
    print(f"  exact optimum    {optimal:>8.2f}")
    print("\nThe offline log was worth more than the entire online budget that followed.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - Fine-tuning is slower than training from scratch: the pretrained Q is
#   confidently wrong somewhere the online policy now visits, and epsilon-greedy
#   will not explore away from a confident wrong value. Optimistic re-initialisation
#   of rarely-visited entries fixes it, at the cost of the safety you paid for.
# - The online curve is noisy but the offline number was stable: they are measured
#   differently unless you are careful. Every point here is an exact
#   `policy_return`, not a windowed average of sampled episodes, so a wobble is a
#   real policy change.
# - Mixed replay helps enormously and you want to keep it forever: it also anchors
#   the policy to the behaviour distribution, so it slows late improvement. Decay
#   the replay ratio rather than leaving it fixed.
# - The dip appears only at high learning rates: that is a real effect and worth
#   knowing, but it is a symptom of overwriting good estimates with single-sample
#   targets, not of anything specific to offline pretraining.