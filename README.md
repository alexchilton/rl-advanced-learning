# RL: everything — tools, tips and cookbook

The smallest honest implementation of each reinforcement learning method, with its
failure modes attached.

Most RL tutorials show you an algorithm that works. This repo is more interested in
the other case: the run that does not learn, produces no error, and looks fine.
Every algorithm here ships with what breaks it, how you would notice, and how to
tell a bug from a hard problem.

**Status:** tabular, deep value, policy gradient, imitation and offline, exploration, preference (on a toy
task, no real LLM) and normalisation tiers are built, each with a notebook. See [`docs/rl-curriculum-plan.md`](docs/rl-curriculum-plan.md) for the
full plan.

## The idea

Three jobs, in increasing order of how rare they are elsewhere.

1. **Show how each algorithm works.** One self-contained file each, readable in one
   screen. No shared base class, no framework. The duplication is deliberate — you
   should never have to open a second file to see what an update rule does.
2. **Show how to see it going wrong.** Probe environments, a health panel, seed
   discipline, and a `broken/` directory of working algorithms with one realistic
   bug each.
3. **Be a cookbook** (planned, not built yet). `cookbook/symptoms.md` will be the thing you open when a run is not
   learning at 11pm and you want a lookup table, not a lecture.

## Ground truth

Every finite environment defines its dynamics once, as a transition table `P`, and
`step()` samples from that table. `envs/solvers.py` reads the same table.

So the "true" Q function cannot drift away from the environment the agent is
actually interacting with. When a learned Q disagrees with `value_iteration`, the
agent is wrong — never the comparison. Almost no standard benchmark can give you
that, and it changes what debugging feels like.

```python
from envs import GridWorld

env = GridWorld(size=6, reward_mode="sparse")
print(env.render_policy(env.optimal_policy(gamma=0.95)))   # exact, not learned
print(env.optimal_return(gamma=0.95))                      # exact, no sampling noise
```

Everything also runs on Gymnasium. The custom environments tell you whether an
implementation matches the truth; Gymnasium tells you whether it matches the field.
An algorithm that solves the gridworld but not CartPole has a bug that ground truth
did not catch.

## Environments

| Env | What it is for |
| --- | --- |
| `GridWorld` | The workhorse. Four reward modes: sparse, dense, potential-shaped, and misspecified. |
| `CliffWalking` | On-policy vs off-policy, in one picture. |
| `Chain` | Exploration, the discount, long-horizon credit assignment. `ScrambledActions` permutes action meanings per state, so no constant action solves it. |
| `Baird` | Baird's counterexample. The deadly triad, minimal form. |
| `Bandit` | Exploration with nothing else attached. |
| `PointMass` | Continuous control. Deliberately badly scaled, so normalisation has something to fix. |

### The reward-mode demo

The same grid, four reward functions. Three give the same optimal policy. The fourth
gives a policy that never reaches the goal:

```
sparse        +1 for entering the goal. Nothing else.
dense         a per-step progress signal. Easy, and it changes the problem.
shaped        potential-based (Ng et al. 1999). Provably same optimal policy.
misspecified  a one-sided progress bonus. The optimal policy is to oscillate
              next to the goal forever, collecting the bonus, and never enter it.
```

The difference between `shaped` and `misspecified` is one `max(0, ...)`. That is the
whole lesson, and `tests/test_gridworld.py` asserts both halves of it.

## Notebooks

Each one runs top to bottom on a laptop CPU and is committed with its outputs, so it reads on GitHub without running anything.
Run them from the repo root, or after `pip install -e .`.

| notebook | what it covers | runtime |
| --- | --- | --- |
| [`00_ground_truth`](notebooks/00_ground_truth.ipynb) | exact solutions, exact policy scoring, the terminal-mask bug, reward hacking vs the discount | seconds |
| [`tabular_01_q_learning`](notebooks/tabular_01_q_learning.ipynb) | Q-learning against the exact optimum; `np.argmax` ties score 0.0 where random ties score 0.5133; four reward modes; SARSA vs Q-learning on the cliff (17-step safe path vs 13-step edge); Dyna-Q planning | ~2 min |
| [`imitation_01_bc_dagger_irl_gail`](notebooks/imitation_01_bc_dagger_irl_gail.ipynb) | BC, DAgger, MaxEnt IRL and GAIL on one 7,126-state racetrack, with what each is allowed to see | ~3 min |
| [`offline_01_q_cql_bcq_iql`](notebooks/offline_01_q_cql_bcq_iql.ipynb) | naive offline Q (believes −1.84, gets −98.73), CQL, BCQ, IQL, four dataset qualities, offline-to-online | ~2 min |
| [`deep_value_01_dqn_family`](notebooks/deep_value_01_dqn_family.ipynb) | probe environments catch a terminal-mask bug DQN hides on CartPole; replay and target network removed one at a time (no target: Q diverges to ~3e10); maximisation bias; DQN, Double, Dueling, PER, 3-step, C51 and Noisy over 5 seeds; C51 recovers a coin flip's exact return distribution | ~2 min live + stored sweep |
| [`policy_gradient_01_reinforce_to_sac`](notebooks/policy_gradient_01_reinforce_to_sac.ipynb) | gradient variance under four weightings (baseline: 12x lower); REINFORCE, A2C, TRPO, PPO (TRPO best on CartPole, curve average 497.6); TRPO's quadratic KL model against the real KL, δ swept; GAE λ; PPO epochs vs clip; DDPG, TD3 ablation, SAC temperature; a pendulum swing-up | ~4 min live + stored sweep |
| [`exploration_01_bonus_curriculum_her`](notebooks/exploration_01_bonus_curriculum_her.ipynb) | Chain(30): ε-greedy never, optimistic init 1,300 steps, size curriculum 850; a learning-progress teacher against random task choice; a count bonus inflates Q past any reachable return; RND with two value heads (5/5 on scrambled chains, DQN 0/5 on Chain(30)); noisy nets never reach the goal there; HER 0% → 100% on 10-20 bits; a network curriculum forgets every earlier goal unless it rehearses | ~3 min live + stored sweeps |
| [`preference_01_rlhf_dpo_grpo`](notebooks/preference_01_rlhf_dpo_grpo.ipynb) | toy task with a known true reward: reward-model accuracy, RLHF with REINFORCE / RLOO / GRPO, over-optimisation measured, biased and AI labelers, DPO | ~1 min live + stored sweeps |
| [`normalisation_01_popart_obs`](notebooks/normalisation_01_popart_obs.ipynb) | rewards ×1000: DQN diverges, PopArt recovers; observation normalisation on PointMass: worst when applied as transitions are stored, best when applied as batches are sampled | seconds + stored sweeps |
| [`kaggle_01_behaviour_cloning_fails`](notebooks/kaggle_01_behaviour_cloning_fails.ipynb) | self-contained BC vs DAgger, no repo imports | ~2 min |
| [`jax_01_lunar_lander`](notebooks/jax_01_lunar_lander.ipynb) | PPO from scratch in Flax/Optax, SB3 vs SBX, gymnax | long; see the notebook |
| [`case_smart_buildings_residual_ppo`](notebooks/case_smart_buildings_residual_ppo.ipynb) | a run that does not learn, kept | see the notebook |

The paper notes in [`papers/`](papers/README.md) map each equation to the line that implements it.

## Try it

```bash
python -m pip install -r requirements.txt
python experiments/show_envs.py    # a tour, with exact solutions
pytest                             # the ground-truth guarantees, checked
```

## Layout

```
envs/          tiny environments, pure numpy, exact ground truth
algos/         one algorithm per file
diagnostics/   probe environments, terminal and matplotlib renderers of the environments
experiments/   comparison runs and resumable seed sweeps; results/ holds their JSON
notebooks/     runnable walkthroughs, committed with their outputs
papers/        the PDFs, notes mapping equations to file:line, and an index
tests/         the guarantees, asserted
```

Planned and not built yet: `broken/` (working algorithms with one realistic bug each),
`cookbook/` (symptom tables, defaults, checklists), and a reusable health panel in
`diagnostics/` (the notebooks plot their own).

## Case studies

Real environments, run end to end, kept whether or not they learned.

| notebook | environment | what happened |
|---|---|---|
| [`case_smart_buildings_residual_ppo`](notebooks/case_smart_buildings_residual_ppo.ipynb) | NeurIPS 2026 Smart Buildings HVAC simulator (official starter kit) | residual PPO on top of the organisers' PID, 200k steps: no improvement on held-out days (0.03020 vs 0.03006), exploration noise never moved from its initial value; and `env.reset()` does not restore the start state |

## Credit

The environments come from Sutton & Barto unless noted. The probe-environment idea
is Andy Jones'. The insistence on seeds and on reporting spread rather than a single
curve comes from Henderson et al., *Deep Reinforcement Learning That Matters*.

## JAX notebooks

| notebook | environment | what happened |
|---|---|---|
| [`jax_01_lunar_lander`](notebooks/jax_01_lunar_lander.ipynb) | LunarLander-v3 (Box2D), CartPole-v1 (gymnax) | PPO from scratch in Flax + Optax: 32 gradient steps per 4,096 samples never learns (greedy 23.8, KL ~1e-3), 256 solves it (271.9) in 92 s; SBX and SB3 PPO tie at ~1,650 steps/s (255.9 / 255.7); the fused-update version runs 13x faster per env step; with the env in JAX, 8 seeds train at 242k steps/s; three silent JAX failure modes measured. Built by `notebooks/build_jax_01.py` |
