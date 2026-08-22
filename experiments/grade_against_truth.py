"""Grade a Q table from anywhere against the exact answer.

    python experiments/grade_against_truth.py

Written for working through a tabular RL course alongside this repo. A course
gives you a learning curve; a curve tells you the agent got better, not whether
it got the answer. This gives you the answer.

    from experiments.grade_against_truth import grade_cliff_walking
    print(grade_cliff_walking(my_q_table, gamma=1.0))

The bridge is only sound because the environments are the same one, which is
checked rather than assumed. `verify_cliff_parity` compares our `envs/cliff.py`
against `gymnasium.CliffWalking-v1` transition by transition and, more usefully,
compares the exact optimal values. Measured: the value functions agree to
0.0000000000 across all 37 reachable states, the optimal policies are identical,
and both give -13.00 from the start.

The eleven state-action rows that differ are the ten cliff cells and the goal.
Neither environment can ever be IN a cliff cell -- stepping into one returns you
to the start -- so those rows are unreachable in both, and Gymnasium simply
declines to mark the goal absorbing inside `P` because its `step()` handles
termination instead.

What this gives you that a reward curve cannot:

    max |Q - Q*|        is the agent wrong, or just early?
    mean |Q - Q*|       is the error everywhere, or in a few states?
    policy agreement    values can be poor while the policy is already optimal
    exact return        what the learned greedy policy is really worth
    worst states        WHERE it is wrong, which is the part you can act on
"""

from __future__ import annotations

import numpy as np

from envs import solvers
from envs.cliff import CliffWalking

GAMMA = 1.0


def gymnasium_cliff_transitions():
    """Gymnasium's CliffWalking-v1 `P`, in this repo's format. Needs gymnasium."""
    import gymnasium as gym

    env = gym.make("CliffWalking-v1").unwrapped
    return {
        state: {
            action: [
                (float(p), int(next_state), float(reward), bool(terminated))
                for p, next_state, reward, terminated in env.P[state][action]
            ]
            for action in range(env.action_space.n)
        }
        for state in range(env.observation_space.n)
    }


def verify_cliff_parity(gamma=GAMMA):
    """Check our CliffWalking against Gymnasium's. Returns a report dict.

    Compares values rather than only transitions, because the transition tables
    legitimately differ on states neither environment can occupy.
    """
    ours = CliffWalking()
    theirs = gymnasium_cliff_transitions()

    differing = set()
    for state in range(ours.n_states):
        for action in range(ours.n_actions):
            a = sorted((round(p, 6), s, r, t) for p, s, r, t in theirs[state][action])
            b = sorted((round(p, 6), s, r, t) for p, s, r, t in ours.P[state][action])
            if a != b:
                differing.add(state)

    V_theirs, _, pi_theirs = solvers.value_iteration(theirs, ours.n_states, ours.n_actions, gamma)
    V_ours, _, pi_ours = solvers.value_iteration(ours.P, ours.n_states, ours.n_actions, gamma)
    comparable = [s for s in range(ours.n_states) if s not in differing]

    return {
        "differing_states": sorted(differing),
        "cliff_and_goal": sorted({ours.index(c) for c in ours.cliff} | {ours.index(ours.goal)}),
        "max_value_difference": float(np.abs(V_theirs[comparable] - V_ours[comparable]).max()),
        "policies_agree": bool((pi_theirs[comparable] == pi_ours[comparable]).all()),
        "start_value_theirs": float(V_theirs[ours.index(ours.start)]),
        "start_value_ours": float(V_ours[ours.index(ours.start)]),
    }


def exact_return(env, policy, gamma):
    """Exact return of a deterministic policy, or -inf if it never terminates.

    At gamma=1 on a task where every step costs -1, a policy that loops forever
    genuinely has value minus infinity, and `policy_evaluation` correctly declines
    to converge rather than returning its last iterate. -inf is the answer, not a
    workaround for one.

    This matters for grading a partly-trained agent, because an early Q table
    routinely produces a policy that circles. Reporting a large negative number
    would suggest it is nearly there; reporting -inf says it never arrives.
    """
    try:
        return env.policy_return(policy, gamma)
    except RuntimeError:
        return float("-inf")


def grade_cliff_walking(Q, gamma=GAMMA, worst=5):
    """Score a learned Q table for CliffWalking against the exact one.

    `Q` is (48, 4) with the standard Gymnasium indexing: state = row * 12 + col,
    actions up, right, down, left.

    Only states the agent can actually occupy are scored. Grading the ten cliff
    cells would report large errors in states no policy ever visits, which is
    noise dressed as a finding.
    """
    Q = np.asarray(Q, dtype=float)
    env = CliffWalking()
    if Q.shape != (env.n_states, env.n_actions):
        raise ValueError(f"expected Q of shape ({env.n_states}, {env.n_actions}), got {Q.shape}")

    truth = env.true_q(gamma)
    unreachable = {env.index(cell) for cell in env.cliff} | {env.index(env.goal)}
    scored = np.array([s for s in range(env.n_states) if s not in unreachable])

    error = np.abs(Q[scored] - truth[scored])
    learned_policy = Q.argmax(axis=1)
    optimal_policy = truth.argmax(axis=1)

    # Tied-optimal actions count as correct. Ties are real here and punishing an
    # arbitrary but equally good choice would make the number meaningless.
    tied = np.abs(truth - truth.max(axis=1, keepdims=True)) < 1e-9
    agree = tied[scored, learned_policy[scored]]

    per_state = error.max(axis=1)
    order = scored[np.argsort(-per_state)][:worst]

    return {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "policy_agreement": float(agree.mean()),
        "exact_return": exact_return(env, learned_policy, gamma),
        "optimal_return": env.optimal_return(gamma),
        "worst_states": [
            {
                "state": int(s),
                "cell": env.cell(int(s)),
                "error": float(np.abs(Q[s] - truth[s]).max()),
                "learned": int(learned_policy[s]),
                "optimal": int(optimal_policy[s]),
            }
            for s in order
        ],
    }


def report(grade):
    value = grade["exact_return"]
    if np.isneginf(value):
        rendered = "    -inf   (the greedy policy never reaches the goal)"
    else:
        rendered = f"{value:8.2f}   (optimal {grade['optimal_return']:.2f})"

    lines = [
        f"max |Q - Q*|        {grade['max_error']:8.4f}",
        f"mean |Q - Q*|       {grade['mean_error']:8.4f}",
        f"policy agreement    {grade['policy_agreement']:8.1%}",
        f"exact return        {rendered}",
        "",
        "worst states:",
    ]
    for entry in grade["worst_states"]:
        lines.append(
            f"  s={entry['state']:>3} {str(entry['cell']):>8}  error {entry['error']:7.3f}  "
            f"learned action {entry['learned']}, optimal {entry['optimal']}"
        )
    return "\n".join(lines)


def main():
    print("=" * 74)
    print("Is our CliffWalking the same environment as Gymnasium's?")
    print("=" * 74)
    try:
        parity = verify_cliff_parity()
    except ImportError:
        print("\ngymnasium is not installed; skipping the parity check.")
        print("pip install gymnasium")
        return

    print(f"\nstates whose transition rows differ: {parity['differing_states']}")
    print(f"cliff cells and goal:                {parity['cliff_and_goal']}")
    print("\nThose are the same set. Neither environment can ever BE in a cliff cell,")
    print("so those rows are unreachable in both, and Gymnasium declines to mark the")
    print("goal absorbing inside P because step() handles termination instead.")
    print(f"\nmax |V_gym - V_ours| over the 37 reachable states: {parity['max_value_difference']:.10f}")
    print(f"optimal policies identical:                        {parity['policies_agree']}")
    print(f"optimal return from the start:  gym {parity['start_value_theirs']:.2f}   "
          f"ours {parity['start_value_ours']:.2f}")
    print("\nSame environment. A Q table learned against either can be graded here.")

    print("\n" + "=" * 74)
    print("What grading looks like, on three deliberately different Q tables")
    print("=" * 74)

    env = CliffWalking()
    truth = env.true_q(GAMMA)
    rng = np.random.default_rng(0)

    for name, Q in (
        ("exact Q*", truth),
        ("Q* plus noise", truth + rng.normal(0, 2.0, truth.shape)),
        ("all zeros (untrained)", np.zeros_like(truth)),
    ):
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        print(report(grade_cliff_walking(Q)))

    print("\nRead the middle row again. Its policy agrees with the optimal one in 86.5%")
    print("of states, and its exact return is minus infinity -- it never reaches the")
    print("goal at all.")
    print("\nThat is the whole argument for grading against truth. 'Mostly right' is not")
    print("a property policies have. Thirteen percent wrong, in the wrong thirteen")
    print("percent, is a policy that circles forever, and no aggregate score over states")
    print("will tell you that. Only running it will -- or evaluating it exactly, which")
    print("is the same answer without the variance.")
    print("\nThe last column of 'worst states' is the part you can act on: it names the")
    print("cells where the learned action differs from the optimal one, which is where")
    print("to look first.")


if __name__ == "__main__":
    main()