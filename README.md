# RL: everything — tools, tips and cookbook

The smallest honest implementation of each reinforcement learning method, with its
failure modes attached.

Most RL tutorials show you an algorithm that works. This repo is more interested in
the other case: the run that does not learn, produces no error, and looks fine.
Every algorithm here ships with what breaks it, how you would notice, and how to
tell a bug from a hard problem.

**Status: early.** The environments and the exact solvers are in. Algorithms follow,
tier by tier. See [`docs/rl-curriculum-plan.md`](docs/rl-curriculum-plan.md) for the
full plan.

## The idea

Three jobs, in increasing order of how rare they are elsewhere.

1. **Show how each algorithm works.** One self-contained file each, readable in one
   screen. No shared base class, no framework. The duplication is deliberate — you
   should never have to open a second file to see what an update rule does.
2. **Show how to see it going wrong.** Probe environments, a health panel, seed
   discipline, and a `broken/` directory of working algorithms with one realistic
   bug each.
3. **Be a cookbook.** `cookbook/symptoms.md` is the thing you open when a run is not
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
| `Chain` | Exploration, the discount, long-horizon credit assignment. |
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
| [`tabular_01_q_learning`](notebooks/tabular_01_q_learning.ipynb) | Q-learning against the exact optimum; `np.argmax` ties score 0.0 where random ties score 0.5133; four reward modes | ~1 min |
| [`imitation_01_bc_dagger_irl_gail`](notebooks/imitation_01_bc_dagger_irl_gail.ipynb) | BC, DAgger, MaxEnt IRL and GAIL on one 7,126-state racetrack, with what each is allowed to see | ~3 min |
| [`offline_01_q_cql_bcq_iql`](notebooks/offline_01_q_cql_bcq_iql.ipynb) | naive offline Q (believes −1.84, gets −98.73), CQL, BCQ, IQL, four dataset qualities, offline-to-online | ~2 min |
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
diagnostics/   text renderers: value heatmaps, policy arrows, learned-vs-true grids
experiments/   comparison runs that print a table
notebooks/     runnable walkthroughs, committed with their outputs
papers/        the PDFs, notes mapping equations to file:line, and an index
tests/         the guarantees, asserted
```

Planned and not built yet: `broken/` (working algorithms with one realistic bug each),
`cookbook/` (symptom tables, defaults, checklists), and the probe environments, health
panel and seed sweeps in `diagnostics/`.

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
