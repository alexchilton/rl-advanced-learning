"""A tour of the environments and their exact solutions.

    python experiments/show_envs.py

Prints, for each environment, what the optimal policy is and what it is worth --
computed exactly, not learned. Run this first. It is also the fastest check that
the environments and the solvers still agree after a change.

The GridWorld section is the one to read. The same grid appears four times under
four reward functions. Three of them produce the same optimal policy. The fourth
produces a policy that never reaches the goal, because it was asked the wrong
question.
"""

from __future__ import annotations

import numpy as np

from envs.bandit import Bandit
from envs.chain import Chain
from envs.cliff import CliffWalking
from envs.gridworld import REWARD_MODES, GridWorld
from envs.pointmass import PointMass

GAMMA = 0.95


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_gridworld() -> None:
    rule("GridWorld -- one grid, four reward functions")
    size = 6
    print(f"{size}x{size}, start top-left, goal bottom-right, gamma={GAMMA}\n")

    for mode in REWARD_MODES:
        env = GridWorld(size=size, reward_mode=mode, gamma=GAMMA)
        pi = env.optimal_policy(GAMMA)
        reaches = rollout_reaches_goal(env, pi)

        print(f"--- reward_mode = {mode} " + "-" * (48 - len(mode)))
        print(env.render_policy(pi))
        print(f"    optimal return from start: {env.optimal_return(GAMMA):+.4f}")
        print(f"    optimal policy reaches the goal: {reaches}")
        if mode == "misspecified":
            honest = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
            honest_value = env.policy_return(honest.optimal_policy(GAMMA), GAMMA)
            print(f"    value of actually solving the task: {honest_value:+.4f}")
            print("    the agent is not broken. it is doing what it was asked.")
        print()

    rule("GridWorld -- shaping changes the values, not the policy")
    sparse = GridWorld(size=size, reward_mode="sparse", gamma=GAMMA)
    shaped = GridWorld(size=size, reward_mode="shaped", gamma=GAMMA)
    print("sparse V*:")
    print(sparse.render_values(sparse.true_v(GAMMA)))
    print("\nshaped V*:")
    print(shaped.render_values(shaped.true_v(GAMMA)))
    same = np.array_equal(sparse.optimal_policy(GAMMA), shaped.optimal_policy(GAMMA))
    print(f"\nsame optimal policy: {same}   (Ng et al. 1999, and tests assert it)")

    rule("GridWorld -- slip makes the optimal policy cautious")
    for slip in (0.0, 0.2, 0.4):
        env = GridWorld(size=size, slip=slip, reward_mode="sparse", gamma=GAMMA)
        print(f"slip={slip:.1f}  optimal return {env.optimal_return(GAMMA):+.4f}")


def show_cliff() -> None:
    rule("CliffWalking -- why SARSA and Q-learning disagree")
    env = CliffWalking()
    print(env.render())
    print("\noptimal (greedy) policy, gamma=1:")
    pi = env.optimal_policy(1.0)
    print(env.render_policy(pi))
    print(f"\noptimal return, no exploration:            {env.optimal_return(1.0):+.2f}")

    from envs import solvers

    gamma, epsilon = 0.999, 0.1
    risky = solvers.epsilon_greedy_policy(env.true_q(gamma), epsilon)
    safe = np.zeros(env.n_states, dtype=np.int64)
    for r in range(env.rows):
        for c in range(env.cols):
            s = env.index((r, c))
            safe[s] = 2 if c == env.cols - 1 else (0 if r > 1 else 1)
    safe_stochastic = np.full((env.n_states, env.n_actions), epsilon / env.n_actions)
    safe_stochastic[np.arange(env.n_states), safe] += 1.0 - epsilon

    print(f"greedy-optimal path under epsilon={epsilon}:      {env.policy_return(risky, gamma):+.2f}")
    print(f"safe path two rows up, under epsilon={epsilon}:   {env.policy_return(safe_stochastic, gamma):+.2f}")
    print("\nQ-learning finds the first policy. SARSA finds something like the second.")
    print("Neither is wrong -- they are evaluating different policies.")


def show_chain() -> None:
    rule("Chain -- exploration is not a hyperparameter")
    print(f"{'n':>4}  {'random-walk success':>20}  {'episodes to see it once':>24}")
    for n in (5, 10, 15, 20, 25):
        env = Chain(n=n)
        p = env.random_walk_success_probability()
        print(f"{n:>4}  {p:>20.2e}  {1 / p:>24,.0f}")
    print("\nEpsilon-greedy IS the random walk until something is learned.")
    print("At n=20 no learning rate helps, because there is nothing to learn from.")

    rule("Chain -- the discount is part of the problem statement")
    print("n=12. 'left' pays a small reward EVERY step, forever. 'right' pays a")
    print("large one ONCE, eleven steps away, and ends the episode.\n")
    print(f"{'gamma':>6}  {'prize=0.005':>28}  {'prize=0.05':>28}")
    for gamma in (0.5, 0.8, 0.9, 0.95, 0.99):
        row = [f"{gamma:>6.2f}"]
        for prize in (0.005, 0.05):
            env = Chain(n=12, small_reward=prize, goal_reward=1.0)
            first = int(env.optimal_policy(gamma)[0])
            row.append(f"{'left (take the prize)' if first == 0 else 'right (go the distance)':>28}")
        print("  ".join(row))
    print("\nAt prize=0.05 no discount saves you: 0.05/(1-gamma) beats gamma^10 for")
    print("every gamma. A reward that pays forever dominates one that pays once.")


def show_bandit() -> None:
    rule("Bandit -- regret, not reward")
    bandit = Bandit(k=10, seed=0)
    print(bandit.summary())
    print("\nRegret is zero for the best arm and positive for every other one, so it")
    print("measures the decisions rather than how generous the arms happen to be.")


def show_pointmass() -> None:
    rule("PointMass -- the observation scales do not match, on purpose")
    env = PointMass(seed=0)
    rng = np.random.default_rng(0)

    print("peak |observation| over 20 episodes, [x, y, vx, vy]:\n")
    for name, policy in (
        ("random policy (what a fresh agent sees)", lambda e: rng.uniform(-1, 1, 2)),
        ("full thrust (the worst case)", lambda e: np.array([1.0, 0.0])),
        ("the PD expert (smooth, so it hides the problem)", lambda e: e.expert_action()),
    ):
        peak = np.zeros(4)
        for episode in range(20):
            env.reset(seed=episode)
            for _ in range(env.max_steps):
                obs, _, terminated, truncated, _ = env.step(policy(env))
                peak = np.maximum(peak, np.abs(obs))
                if terminated or truncated:
                    break
        ratio = peak[2:].max() / max(peak[:2].max(), 1e-9)
        print(f"  {np.round(peak, 2)}   ratio {ratio:5.1f}x   {name}")

    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(env.expert_action())
        if terminated or truncated:
            break
    print(f"\nexpert finished in {env.t} steps: {env.render()}")
    print("\nA fresh agent explores randomly, so it sees the top row: velocities")
    print("roughly ten times larger than positions. Its first layer needs weights")
    print("ten times smaller on two of four inputs before it can learn anything,")
    print("and one learning rate for all four mostly will not get there.")
    print("\nNote the third row. The expert's own trajectories look well scaled, so")
    print("measuring the problem on expert data hides it completely.")


def rollout_reaches_goal(env, policy, max_steps: int = 500) -> bool:
    env.reset(seed=0)
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(int(policy[env.state]))
        if terminated:
            return True
        if truncated:
            return False
    return False


def main() -> None:
    show_gridworld()
    show_cliff()
    show_chain()
    show_bandit()
    show_pointmass()
    print()


if __name__ == "__main__":
    main()
