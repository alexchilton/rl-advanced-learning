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
broken/        working algorithms with one realistic bug each, plus FIX.md
diagnostics/   probe environments, health panel, seed sweeps
experiments/   comparison runs that emit a table and a plot
cookbook/      symptom tables, defaults, checklists
papers/        notes and an index. PDFs are fetched, not vendored.
tests/         the guarantees, asserted
```

## Credit

The environments come from Sutton & Barto unless noted. The probe-environment idea
is Andy Jones'. The insistence on seeds and on reporting spread rather than a single
curve comes from Henderson et al., *Deep Reinforcement Learning That Matters*.
