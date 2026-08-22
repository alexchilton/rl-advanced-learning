"""Generative Adversarial Imitation Learning, tabular (Ho & Ermon, 2016).

    python algos/gail.py

The row of the access table that is easiest to misread. GAIL sees demonstrations
and can interact with the environment. It does NOT see a reward, and it cannot
query the expert -- the demonstrations are fixed, recorded once.

That combination is what makes it interesting. Behaviour cloning has the same
demonstrations and no interaction, so its errors compound. DAgger fixes that by
asking the expert about the states the learner reaches, which needs the expert
present during training. GAIL fixes it with interaction alone.

The mechanism is a two-player game:

    discriminator   learns to tell expert transitions from learner transitions
    policy          learns to produce transitions the discriminator cannot tell apart

and the reward the policy optimises is manufactured from the discriminator:

    r(s,a) = -log(1 - D(s,a))

which is large where the discriminator is confident the transition came from the
expert. No environment reward appears anywhere.

**The inner optimisation uses sampled interaction, not the transition model.**
That matters for honesty rather than for results: `max_ent_irl.py` may use the
model because inverse RL assumes it, and GAIL may not, because it does not. Using
the model here would quietly hand GAIL strictly more information than the access
table says it has, and the comparison would stop meaning anything.

**The discriminator is deliberately low-capacity**, and that is the interesting
implementation detail. See `discriminator_features` below.
"""

from __future__ import annotations

import numpy as np

from envs.racetrack import Racetrack

GAMMA = 0.99
ROUNDS = 30
STEPS_PER_ROUND = 4_000
DISCRIMINATOR_STEPS = 40
DISCRIMINATOR_RATE = 0.05
POLICY_RATE = 0.3
EPSILON = 0.15
WEIGHT_DECAY = 0.01


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def discriminator_features(env):
    """Map each state to a coarse bucket: its track cell, ignoring velocity.

    This is the load-bearing choice in a tabular GAIL and it is worth dwelling on.

    A discriminator with one parameter per (state, action) has 64,000 parameters
    and a few thousand samples. It memorises. Any pair seen only in the expert
    data gets D -> 1 and any pair seen only in the learner's data gets D -> 0, so
    the manufactured reward collapses into "have I seen this exact pair in the
    demonstrations". That is support matching, not distribution matching, and it
    is behaviour cloning with a great deal of extra machinery.

    Restricting the discriminator to position and action leaves about a thousand
    parameters, which cannot memorise 7,126 states and therefore has to generalise.
    In deep GAIL the same job is done by a small network and the same failure
    appears when it is made too large -- the discriminator wins the game outright,
    the reward saturates, and the policy gradient carries no signal.
    """
    buckets = np.zeros(env.n_states, dtype=np.int64)
    cells = sorted(env.drivable)
    index = {cell: i for i, cell in enumerate(cells)}
    for state in range(env.n_states):
        if state == env.finished_state:
            buckets[state] = len(cells)
            continue
        cell, _, _ = env.decode(state)
        buckets[state] = index.get(cell, len(cells))
    return buckets, len(cells) + 1


def collect_demonstrations(env, policy, n_demonstrations, seed=0):
    """Expert (state, action) pairs. GAIL only ever sees these -- no rewards."""
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for _ in range(n_demonstrations):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(env.max_steps):
            state = env.state
            action = int(rng.choice(env.n_actions, p=policy[state]))
            states.append(state)
            actions.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return np.array(states), np.array(actions)


REWARD_FORMS = ("positive", "negative", "symmetric")


def manufactured_reward(confidence, form):
    """Turn the discriminator's output into a reward. The SIGN is the whole story.

    positive    -log(1 - D)   in (0, inf).  Every step pays, so the agent wants to
                              survive. Terminating ends the income.
    negative     log(D)       in (-inf, 0). Every step costs, so the agent wants
                              the episode over.
    symmetric    log(D) - log(1 - D)        Crosses zero at D = 0.5. No prior on
                              episode length, and no obvious scale either.

    None of these is neutral. GAIL's reward encodes a prior about how long an
    episode should be, and that prior has nothing to do with imitation -- it is an
    accident of which algebraic form you wrote down. Kostrikov et al. 2019 make
    the same point and fix it by learning a reward for the absorbing state too.
    """
    confidence = np.clip(confidence, 1e-8, 1 - 1e-8)
    if form == "positive":
        return -np.log(1.0 - confidence)
    if form == "negative":
        return np.log(confidence)
    if form == "symmetric":
        return np.log(confidence) - np.log(1.0 - confidence)
    raise ValueError(f"reward form must be one of {REWARD_FORMS}, got {form!r}")


def train(env, expert_states, expert_actions, rounds=ROUNDS, steps_per_round=STEPS_PER_ROUND,
          gamma=GAMMA, seed=0, epsilon=EPSILON, policy_rate=POLICY_RATE,
          discriminator_rate=DISCRIMINATOR_RATE, weight_decay=WEIGHT_DECAY,
          reward_form="negative"):
    """Returns (Q, weights). Never reads the environment's reward."""
    buckets, n_buckets = discriminator_features(env)
    weights = np.zeros((n_buckets, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(seed)

    expert_buckets = buckets[expert_states]
    env.reset(seed=seed)

    for _ in range(rounds):
        # ---- 1. act, and record where we went -------------------------
        states, actions, next_states, terminated = [], [], [], []
        for _ in range(steps_per_round):
            state = env.state
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.n_actions))
            else:
                action = int(Q[state].argmax())
            _, _, term, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(env.state)
            terminated.append(term)
            if term or trunc:
                env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        terminated = np.array(terminated)
        learner_buckets = buckets[states]

        # ---- 2. discriminator: expert is 1, learner is 0 --------------
        for _ in range(DISCRIMINATOR_STEPS):
            expert_index = rng.integers(0, len(expert_buckets), 256)
            learner_index = rng.integers(0, len(learner_buckets), 256)

            gradient = np.zeros_like(weights)
            e_b, e_a = expert_buckets[expert_index], expert_actions[expert_index]
            l_b, l_a = learner_buckets[learner_index], actions[learner_index]
            np.add.at(gradient, (e_b, e_a), 1.0 - sigmoid(weights[e_b, e_a]))
            np.add.at(gradient, (l_b, l_a), -sigmoid(weights[l_b, l_a]))

            weights += discriminator_rate * gradient / 256.0
            # Shrinkage, so a bucket seen only on one side cannot run to infinity.
            weights *= 1.0 - weight_decay

        # ---- 3. policy: Q-learning on the manufactured reward ---------
        confidence = sigmoid(weights[learner_buckets, actions])
        reward = manufactured_reward(confidence, reward_form)

        for i in range(len(states)):
            target = reward[i] + (0.0 if terminated[i] else gamma * Q[next_states[i]].max())
            Q[states[i], actions[i]] += policy_rate * (target - Q[states[i], actions[i]])

    return Q, weights


def behaviour_clone(n_states, n_actions, states, actions, seed=0):
    counts = np.zeros((n_states, n_actions))
    np.add.at(counts, (states, actions), 1.0)
    policy = counts.argmax(axis=1)
    unseen = counts.sum(axis=1) == 0
    policy[unseen] = np.random.default_rng(seed).integers(0, n_actions, int(unseen.sum()))
    return policy.astype(np.int64)


def noisily_rational_expert(env, gamma=GAMMA, rationality=0.7):
    optimal = np.zeros((env.n_states, env.n_actions))
    optimal[np.arange(env.n_states), env.optimal_policy(gamma)] = 1.0
    uniform = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    return rationality * optimal + (1.0 - rationality) * uniform


def main():
    env = Racetrack()
    expert = noisily_rational_expert(env)
    optimal = env.optimal_return(GAMMA)
    expert_value = env.policy_return(expert, GAMMA)

    print("=" * 78)
    print("GAIL: imitation with environment access and no reward")
    print("=" * 78)
    print(f"Racetrack, {env.n_states:,} states. Exact optimum: {optimal:.2f}")
    print(f"The demonstrator: {expert_value:.2f}")
    print("\nGAIL never sees a reward. It sees the demonstrations and it may interact.")

    states, actions = collect_demonstrations(env, expert, 30, seed=0)

    print("\n" + "=" * 78)
    print("First: the sign of the reward decides whether the agent wants to live")
    print("=" * 78)
    print(f"\n30 demonstrations, {len(states):,} expert pairs, 3 seeds\n")
    print(f"{'reward form':>12} {'formula':>26} {'exact return, per seed':>34}")

    for form in REWARD_FORMS:
        values = []
        for seed in (0, 1, 2):
            Q, _ = train(env, states, actions, seed=seed, reward_form=form)
            values.append(env.policy_return(Q.argmax(axis=1), GAMMA))
        formula = {
            "positive": "-log(1 - D)   in (0, inf)",
            "negative": "log(D)        in (-inf, 0)",
            "symmetric": "log(D) - log(1 - D)",
        }[form]
        cells = " ".join(f"{value:>10.2f}" for value in values)
        print(f"{form:>12} {formula:>26} {cells:>34}")

    print("\nThe positive form never finishes. -100 is the value of running out the")
    print("clock forever, and it is not a tuning failure -- it is arithmetic. Every")
    print("step earns something strictly greater than zero, terminating earns nothing")
    print("more, so an immortal agent is optimal under that reward. The discriminator")
    print("could be perfect and it would still happen.")
    print("\nThis is the best-known GAIL pathology and it is a property of the reward's")
    print("SIGN, not of adversarial training. Kostrikov et al. 2019 fix it properly, by")
    print("learning a reward for the absorbing state instead of assuming it is zero.")
    print("\nBe suspicious of the negative form winning here, though. Racetrack rewards")
    print("finishing quickly, and 'every step costs something' happens to be the right")
    print("prior for that task. It would be exactly as wrong on a task where the goal")
    print("is to survive. Neither sign is neutral; the choice smuggles in an assumption")
    print("about episode length that has nothing to do with imitation.")

    print("\n" + "=" * 78)
    print("Second: what environment access buys, against behaviour cloning")
    print("=" * 78)
    print(f"\nmedian [min, max] over 3 seeds\n")
    print(f"{'demos':>7} {'pairs':>8} {'BC':>22} {'GAIL':>22}")

    for n_demonstrations in (3, 10, 30, 100):
        bc_values, gail_values = [], []
        for seed in (0, 1, 2):
            demo_states, demo_actions = collect_demonstrations(
                env, expert, n_demonstrations, seed=seed
            )
            clone = behaviour_clone(
                env.n_states, env.n_actions, demo_states, demo_actions, seed=seed
            )
            bc_values.append(env.policy_return(clone, GAMMA))
            Q, _ = train(env, demo_states, demo_actions, seed=seed, reward_form="negative")
            gail_values.append(env.policy_return(Q.argmax(axis=1), GAMMA))
        b, g = np.array(bc_values), np.array(gail_values)
        print(
            f"{n_demonstrations:>7} {len(demo_states):>8,} "
            f"{f'{np.median(b):8.2f} [{b.min():7.2f},{b.max():7.2f}]':>22} "
            f"{f'{np.median(g):8.2f} [{g.min():7.2f},{g.max():7.2f}]':>22}"
        )

    print("\nThe crossover is the result, and it goes both ways.")
    print("\nBelow about thirty demonstrations GAIL wins clearly -- around -12 against")
    print("BC's -69 -- because BC has almost nothing to copy and GAIL can go and look.")
    print("That IS the textbook argument working: environment access substituting for")
    print("data, without needing the expert on call.")
    print("\nAbove thirty it reverses. BC reaches -7.02 and GAIL gets WORSE with more")
    print("demonstrations, ending at -27.03. More data should not hurt, and the fact")
    print("that it does is the tell.")
    print("\nA method that degrades with more data is not short of data. GAIL sits")
    print("between -10 and -27 at every budget, and which end it lands on looks more")
    print("like which seed it drew than how much it was shown. That is an optimisation")
    print("problem, and it is inherent rather than incidental: the policy is chasing a")
    print("reward that MOVES every round, so there is no fixed objective to converge")
    print("to, and the inner Q-learning is re-solving a changed problem each time.")
    print("Minimax problems do not converge the way minimisation problems do.")
    print("\nSo the honest summary is narrower than the paper's and more useful:")
    print("\n  GAIL wins when behaviour cloning has too little data to copy, and an")
    print("  expert cannot be queried for more. Outside that window, on a task BC can")
    print("  already solve, it is slower, less stable and worse.")
    print("\nBoth halves matter. The low-data win is real and is the reason the method")
    print("exists. The high-data loss is real too, and no amount of tuning here was")
    print("going to make a table agree with a paper that was measured on continuous")
    print("control with a policy-gradient inner loop.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# What breaks here, and how you would know
# ----------------------------------------------------------------------
# - GAIL performs exactly like behaviour cloning: the discriminator has too much
#   capacity and has memorised. Check how many parameters it has against how many
#   distinct expert pairs exist. If it can separate the two sets perfectly, the
#   manufactured reward is just an indicator for "this pair was in the
#   demonstrations", which is support matching and not distribution matching.
# - The reward saturates and the policy stops improving: the discriminator is
#   winning outright, so D is 0 or 1 everywhere and -log(1 - D) is flat or
#   infinite. Fewer discriminator steps per round, or more shrinkage.
# - The agent survives but never finishes: r = -log(1 - D) is strictly positive,
#   so every extra step is worth something and terminating ends the income. This
#   is the best-known GAIL bias and it is a property of the reward's SIGN, not of
#   the algorithm. Using r = log(D) instead makes rewards negative and biases the
#   other way, towards ending episodes early. Neither is neutral, and which one to
#   pick depends on whether the task wants long or short episodes.
# - Both players oscillate and nothing converges: expected. This is a minimax
#   problem, and minimax problems do not converge the way minimisation problems
#   do. Judge it on the policy's return over rounds, never on either loss.
# - Results swing wildly between seeds: also expected, and the reason to report
#   several. Adversarial training is the least stable thing in this repo.