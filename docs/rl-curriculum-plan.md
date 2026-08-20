# RL: Everything — Tools, Tips and Cookbook

Date: 2026-08-20

## Goal

One repo covering the full range of RL, as the smallest working example of each thing.

It has three jobs, in increasing order of how rare they are elsewhere:

1. **Show how each algorithm works** — the smallest honest implementation, readable in one screen.
2. **Show how to see it going wrong** — the diagnostics, the probe environments, the health panel, the seed discipline.
3. **Be a cookbook** — the thing you open when a run is not learning at 11pm and you need the symptom-to-cause table, not a lecture.

Job 3 is the one that makes this worth building.
Every algorithm file therefore ends with a short "what breaks here and how you would know" block, and `cookbook/` collects those into one index.

## Decisions already made

| Decision | Choice |
| --- | --- |
| Implementation style | From scratch, one self-contained file per algorithm. No shared base class. Duplication is deliberate. |
| Environments | Two tracks. Custom dependency-free envs are primary, because they have exact ground truth. Gymnasium classic control is the secondary track, for comparability with published numbers. MuJoCo optional, gated on install working. |
| Format | `.py` scripts for algorithms and experiments. Notebooks only for guided debugging walkthroughs. |
| Papers | PDFs in `papers/`, plus a `notes.md` per paper mapping equations to our line numbers. Link to author code, do not vendor it. |
| Python | `.venv` Python 3.14.7. torch 2.13.0 has a `cp314` arm64 wheel — verified by dry-run install on 2026-08-20. |

### Why two environment tracks

The custom envs are small enough to solve exactly by value iteration.
That exact solution is the most valuable debugging asset in the repo: you can always compare what the agent learned to what is true.
No standard benchmark gives you that.

Gymnasium earns its place for the opposite reason.
It is what every paper, tutorial and Udemy course uses, so it is the only way to check that an implementation is right rather than merely convergent.
An algorithm that solves our gridworld but not CartPole has a bug that ground truth did not catch.

Rule for the repo: **every algorithm runs on both.**
Custom env first, for the ground-truth check and the debugging material.
Gymnasium second, as the comparability check.
The environment is a constructor argument, not a rewrite — the algorithm files take a Gymnasium-style `reset()` / `step()` interface, and the custom envs implement that same interface.

## Layout

```
envs/          tiny environments, pure numpy, no deps, exact ground truth
algos/         one algorithm per file
broken/        working algorithms with one realistic bug each, plus FIX.md
diagnostics/   the debugging toolkit
experiments/   comparison runs that emit a text table and a PNG
cookbook/      the reference half: symptom tables, defaults, checklists
notebooks/     guided debugging walkthroughs
papers/        PDFs, organised by topic, each with notes.md
tests/         regression tests: each algo must still hit a known return
docs/          this plan and the write-ups
```

## `cookbook/` — the part you actually reopen

Short, dense, no narrative. Written last for each tier, from what actually went wrong while building it.

| File | Contents |
| --- | --- |
| `symptoms.md` | The lookup table. "Reward flat at zero", "Q values growing without bound", "entropy collapsed to zero in 200 steps", "value loss 1000× the policy loss", "works on seed 0 only" — each with the two or three causes worth checking, in order of likelihood. |
| `first_hour.md` | The ordered checklist for a new agent that does not learn. Overfit one batch, run the probe environments, check terminal masking, check observation scaling, check the action actually reaches the environment, plot the TD-error distribution. In that order, because that is cheapest-first. |
| `defaults.md` | Hyperparameters that work, per algorithm, with the range each one tolerates and which ones are worth tuning at all. Most are not. |
| `choosing.md` | One page: given discrete or continuous actions, a simulator or a fixed dataset, sparse or dense reward, and a step budget — which algorithm. With the honest answer that for many problems it is not RL. |
| `numbers.md` | What healthy looks like. Explained variance above 0.8. Approximate KL per update around 0.01. Clip fraction between 0.05 and 0.3. Gradient norm stable within an order of magnitude. Entropy decaying slowly, not falling off a cliff. |
| `gotchas.md` | The API-level traps. `terminated` vs `truncated`. Numpy views in a replay buffer. Seeding every one of numpy, torch, python and the environment. Vectorised environments auto-resetting under you. `.item()` in a hot loop. Evaluating with exploration still on. |

## Environments

All pure numpy, all under about 100 lines, all with an exact solver where the state space is finite.

| File | Purpose |
| --- | --- |
| `envs/bandit.py` | k-armed bandit. Exploration without bootstrapping or credit assignment. |
| `envs/gridworld.py` | N×N grid. Switchable reward mode (sparse / dense / shaped / misspecified), slip probability, horizon. Exposes `true_q()` by value iteration. |
| `envs/cliff.py` | Cliff Walking. The canonical on-policy vs off-policy demonstration. |
| `envs/chain.py` | Linear chain and Baird's counterexample. For bootstrapping, long-horizon credit assignment, and the deadly triad. |
| `envs/pointmass.py` | 2D continuous state and action. For DDPG, TD3, SAC and continuous PPO. Deliberately badly scaled (position ~1, velocity ~40) so observation normalisation has something to fix. Also where action-scaling bugs show up. |
| `envs/token_env.py` | ~10-token vocabulary, 6-token sequences, a hidden preference function. Small enough to enumerate every sequence, so RLHF and GRPO also get exact ground truth. |

### Gymnasium track

The standard set, used as the comparability check for each tier.

| Env | Used by |
| --- | --- |
| `FrozenLake-v1` | Tabular methods, and the slippery variant for stochastic transitions |
| `CliffWalking-v1` | SARSA vs Q-learning, cross-checked against our own `cliff.py` |
| `Taxi-v3` | Tabular with a larger state space, hierarchical structure |
| `CartPole-v1` | DQN, REINFORCE, A2C, PPO. The standard first target. |
| `MountainCar-v0` | Sparse reward with a hard exploration problem. ε-greedy fails here, which is the point. |
| `Acrobot-v1` | Longer horizon credit assignment |
| `Pendulum-v1` | Continuous control: DDPG, TD3, SAC, continuous PPO |
| `LunarLander-v3` | Harder discrete and continuous target, and where reward shaping matters |
| `MountainCarContinuous-v0` | Continuous plus sparse — the case that needs HER or RND |

MuJoCo (`HalfCheetah-v5`, `Hopper-v5`) is an optional third track for the continuous-control and offline tiers, gated on the install working.
Minari for standard offline datasets, likewise optional — our own gridworld dataset-quality axis does not depend on it.

`gridworld.py` is the workhorse.
Its reward modes drive the sparse-vs-dense material directly:

- **sparse** — +1 at the goal, 0 everywhere else. Nothing learns without exploration help.
- **dense** — small negative per step plus distance-to-goal. Learns fast.
- **shaped** — potential-based shaping, Ng et al. 1999. Provably leaves the optimal policy unchanged.
- **misspecified** — a shaping term that is *not* potential-based. The agent learns to farm the shaping term and never reaches the goal. This is the reward-hacking demo.

## Algorithms, by tier

Each file runs standalone: `python algos/ppo.py`.
Each prints a learning curve as text and writes a PNG.

**Tier 0 — no bootstrapping.**
`bandit_epsilon_greedy.py`, `bandit_ucb.py`, `bandit_thompson.py`, `monte_carlo.py` (first-visit prediction, then MC control).

**Tier 1 — tabular TD.**
`td0.py`, `sarsa.py`, `expected_sarsa.py`, `q_learning.py`, `double_q.py` (maximisation bias), `n_step_sarsa.py`, `sarsa_lambda.py` (eligibility traces), `dyna_q.py` (model-based planning).

**Tier 2 — function approximation.**
`linear_sarsa_tiles.py` — tile coding. This is the bridge from tables to networks and it is where most tutorials skip a step.
Then `dqn.py`, and as separate files rather than flags: `double_dqn.py`, `dueling_dqn.py`, `per_dqn.py`, `n_step_dqn.py`.

**Tier 3 — policy gradient.**
`reinforce.py`, `reinforce_baseline.py`, `a2c.py`, `ppo.py` (clipping and GAE), `ddpg.py`, `td3.py`, `sac.py`.

**Tier 4 — imitation, inverse and offline.**

The organising idea for this tier is not the algorithms, it is **what each method is allowed to touch**.
Almost every confusion in this area dissolves once the access columns are written down, because the algorithms differ less than the assumptions do.

| Method | Reward? | Env interaction? | Expert queries? | Can exceed the expert? |
| --- | --- | --- | --- | --- |
| Behaviour cloning | no | no | no | no |
| DAgger | no | yes | yes, interactive | no |
| GAIL | no | yes | no (fixed demos) | no |
| Inverse RL | learns one | yes | no | yes |
| Offline RL | yes, in the data | no | no | yes |
| Offline → online RL | yes | yes | no | yes |
| Online RL | yes | yes | no | n/a |

Read the columns, not the rows.

- **"Can exceed the expert" tracks the reward column exactly.** Nothing that only imitates can beat what it imitates, no matter how good the architecture. If a method has no notion of what is *good* — only of what was *done* — the demonstrator is a ceiling. That single observation explains why IRL and offline RL are worth their extra machinery and BC is not, when the demonstrations are mediocre.
- **BC and offline RL sit in the same box on access and in different boxes on outcome.** Both see a fixed dataset and never touch the environment. BC asks "what did the expert do here", offline RL asks "what is worth doing here", and only the second can stitch together a better trajectory than any single one in the data. `experiments/stitching.py` builds a dataset of two mediocre trajectories that share a state, and shows offline RL recovering a path that no demonstration contains.
- **DAgger's expert-query column is the expensive one.** It is the only method here that needs the expert *available during training* rather than recorded once. That is cheap with a scripted controller and ruinous with a human, and it is why DAgger is common in papers and rare in products.
- **GAIL sits between BC and IRL.** Fixed demos like BC, environment access like IRL, but it never produces a reward function you can read — the discriminator is a means, not an output. If you want the reward for its own sake, that is IRL.
- **Offline → online is where most real projects actually live** and is the least taught. Pretrain on logged data, then fine-tune with interaction. Its characteristic failure is its own: the first online updates destroy the pretrained policy, because the fresh on-policy data looks nothing like the buffer it was trained on.

Files, one per row plus the failure modes:

| File | Row |
| --- | --- |
| `bc.py` | Behaviour cloning, with the covariate-shift measurement attached |
| `dagger.py` | DAgger, with a query-budget sweep so the expert cost is visible |
| `gail.py` | Adversarial imitation, discriminator and policy |
| `max_ent_irl.py` | Tabular MaxEnt IRL — recovers the reward on the gridworld, compared cell by cell against the true one |
| `cql_tabular.py`, `cql.py` | Conservative Q-learning, tabular first |
| `bcq.py`, `iql.py` | Batch-constrained and implicit Q-learning |
| `offline_to_online.py` | Pretrain then fine-tune, and the collapse at the handover |

And the experiment that makes the table concrete: **the same gridworld, the same expert, and all seven rows scored on identical demonstration data**, with a column for environment steps used and a column for expert queries used. A method that wins on return while spending 10,000 expert queries has not won.

**Tier 5 — exploration, reward, curriculum, and getting unstuck.**

`her.py`, `count_based_exploration.py`, `rnd.py`, `reward_shaping.py`, plus the curriculum and local-optima material below.

**Curriculum learning** gets four files, because "curriculum" names four different things that people conflate.

| File | What varies | The measurement |
| --- | --- | --- |
| `curriculum_size.py` | Grid grows 4×4 → 6×6 → ... → 14×14, transferring the Q table each time | Total environment steps to solve 14×14, against training on 14×14 directly. The direct run mostly never solves it, which is the point. |
| `curriculum_reverse.py` | Start state moves outward from the goal (Florensa et al. 2017) | The strongest version on a sparse task: every early episode sees reward, so there is always signal. Needs the ability to reset to an arbitrary state — worth flagging, because real environments often cannot. |
| `curriculum_automatic.py` | The next task is chosen by learning progress rather than by a hand-written schedule | Compares against a fixed schedule and against random task selection. Random is a surprisingly strong baseline and is usually omitted from papers. |
| `curriculum_failure.py` | The same schedule, run too fast | Catastrophic forgetting: the agent solves stage 3 and can no longer solve stage 1. Measured by re-evaluating every earlier stage after each transition — a curve nobody plots and everybody should. |

The honest framing throughout: a curriculum is **an exploration method wearing different clothes**. It works by ensuring the agent sees reward early, which is the same thing HER, RND and count-based bonuses do by other means. `experiments/curriculum_vs_exploration.py` runs all four on the same sparse task and reports steps-to-threshold, so the comparison is explicit rather than implied.

**Escaping local optima** gets its own file set, because "the agent plateaus" is the most common real complaint and the least covered topic.

| File | Mechanism |
| --- | --- |
| `local_optima_zoo.py` | Four distinguishable plateau causes on one task: exhausted exploration, entropy collapse, a genuine reward local optimum, and a saturated network. Each has a different fix and the same-looking flat curve. |
| `entropy_floor.py` | A minimum-entropy constraint, and the automatic temperature tuning SAC uses. The difference between a policy that stopped exploring and one that stopped improving. |
| `restarts.py` | Random restarts and population-based training. Embarrassingly simple, routinely beats careful tuning of a single run. |
| `noisy_nets.py` | Parameter-space noise instead of action-space noise — exploration that is consistent within an episode rather than jittering every step. |
| `optimistic_init.py` | Optimistic initialisation. One line, no bonus terms, systematic exploration. Included partly because it is the cheapest thing that works and almost never tried. |

The diagnostic that ties it together, in `cookbook/plateau.md`: **a flat curve is at least four different bugs.** Check entropy, check state-visitation coverage, check the gradient norm, and check whether the exact optimal return is actually higher than what you have. That last one catches the case where the agent has already won and you are tuning against noise.

**Tier 6 — preference-based and LLM-style RL.**
Not optional. This is the tier closest to the work you actually do, and it inherits every failure mode from the tiers above plus a few of its own.

All of it runs on `envs/token_env.py`: a tiny sequence environment with a vocabulary of about 10 tokens, a 6-token horizon, and a hidden "true" preference function.
Small enough to enumerate every possible sequence, which means — as with the gridworld — there is exact ground truth to compare against.
A "model" here is a 2-layer network over a token prefix. Nothing is downloaded, nothing takes longer than a minute.

| File | Point it makes |
| --- | --- |
| `reward_model.py` | Bradley-Terry model over preference pairs. Where preference data comes from, and what a reward model actually is. |
| `best_of_n.py` | The baseline that is embarrassingly hard to beat. Report every method below against it. |
| `ppo_rlhf.py` | Full PPO with a KL-to-reference penalty. Shows why the KL term exists by removing it. |
| `grpo.py` | Group-relative advantage: sample a group per prompt, use the group mean as the baseline, drop the value network entirely. |
| `rloo.py` | REINFORCE leave-one-out. Nearly the same idea as GRPO, arrived at from the variance-reduction side. Comparing the two is more instructive than either alone. |
| `dpo.py` | No reward model, no sampling, no RL loop. Closed-form preference optimisation. |
| `reward_hacking_rm.py` | Over-optimise the reward model. True reward peaks and then falls while modelled reward keeps climbing. The Goodhart curve, reproduced in a minute on a toy problem. |

The comparisons that matter here:

- **GRPO vs PPO.** GRPO removes the critic. That kills a whole class of bugs (value loss scaling, GAE off-by-one, critic underfitting) and introduces another (group size too small, so the baseline is noise). Run both, show the variance of the advantage estimate for group sizes 2, 4, 8, 16.
- **DPO vs PPO-RLHF.** Same preference data, same reference model. DPO is far simpler and has no sampling loop, but it optimises against the *dataset's* distribution rather than the current policy's. Construct the case where that gap matters and show DPO losing.
- **KL penalty coefficient sweep.** Too low and the policy drifts to gibberish that the reward model loves. Too high and nothing changes. This is the single most-tuned number in RLHF and here you can see the whole curve.
- **Length bias.** Add a small per-token reward by accident and watch every method discover that longer is better. The best-known real RLHF pathology, in ten lines.

Also worth including, same environment: **process vs outcome reward** (per-step versus terminal-only — the sparse/dense axis from Tier 5, in its modern form) and **verifiable reward** (a rule-checkable correctness signal instead of a learned reward model, which is what removes the reward-hacking failure mode entirely).

## On-policy vs off-policy

This axis gets its own thread through the repo rather than one file.

1. `experiments/cliff_sarsa_vs_qlearning.py` — the classic picture. SARSA takes the safe path away from the cliff because its own ε-greedy exploration is what it evaluates. Q-learning takes the optimal path along the edge and falls off during training. Same environment, same ε, different target.
2. `experiments/expected_sarsa_bridges.py` — Expected SARSA sits between the two, and becomes Q-learning exactly when the target policy is greedy.
3. `experiments/importance_sampling_variance.py` — off-policy Monte Carlo with ordinary vs weighted importance sampling. Watch the ordinary-IS estimator's variance explode as the two policies separate. This explains why deep off-policy methods use replay plus a target network instead of IS ratios.
4. `experiments/ppo_replay_staleness.py` — PPO is on-policy, but every implementation reuses each batch for several epochs. Sweep epochs from 1 to 20 and watch the KL between the behaviour and current policy grow until the clipping can no longer contain it. This is the practical meaning of "on-policy".
5. `experiments/deadly_triad.py` — Baird's counterexample. Off-policy plus bootstrapping plus function approximation diverges. Remove any one of the three and it does not.
6. Offline RL is the limit case: the behaviour policy is fixed and you cannot collect more data. `cql_tabular.py` shows the overestimation of out-of-distribution actions that plain Q-learning suffers, and how the conservative penalty fixes it.

## The debugging half

### `broken/` — bug catalogue

Each file is a working algorithm with exactly one realistic bug.
Each ships with a `FIX.md`: symptom, how you would notice, the diagnosis, the fix.
The bug is never a typo. Every one of these is something that ships in real code.

| # | Bug | Symptom |
| --- | --- | --- |
| 1 | No terminal masking — bootstraps past `done` | Values grow without bound on an episodic task |
| 2 | Target network copied every step | Q diverges, loss looks fine |
| 3 | Truncation treated as termination | Agent learns to run out the clock |
| 4 | `next_value` wrong at the last step of a GAE rollout | Slow, biased, still "works" — the worst kind |
| 5 | Advantage normalised over one sample | Advantage is always 0, no learning, no error |
| 6 | PPO ratio computed with the current policy for both terms | Ratio is always 1, clipping never fires, it is now vanilla PG |
| 7 | Replay buffer stores numpy views, not copies | Every transition is identical to the newest one |
| 8 | ε decayed per step instead of per episode | Exploration is gone by step 200 |
| 9 | Optimiser constructed over the target network's parameters | Nothing learns, no error message |
| 10 | γ = 1 on a continuing task | Divergence |
| 11 | SAC without the tanh log-prob correction | Entropy term is wrong, policy collapses |
| 12 | Learning rate 10× too high | Entropy → 0 in a few hundred steps, policy deterministic and wrong |
| 13 | Single seed | A "result" that does not survive a second seed |

### Normalised vs unnormalised

Its own thread, because it is the single most common reason a correct implementation does not learn, and because the fix is invisible in the algorithm's equations.
Every one of these is a run that fails, a one-line change, and the same run succeeding.

1. **Observations.** `envs/pointmass.py` is deliberately badly scaled: position is in roughly `[-1, 1]`, velocity in roughly `[-40, 40]`. A network fed the raw observation learns almost nothing, because the first layer's weights need wildly different magnitudes per input. A running mean/std normaliser fixes it. `experiments/obs_normalisation.py` runs both and plots them together.
2. **Rewards.** Same environment, reward multiplied by 1000. The algorithm is unchanged and mathematically the optimal policy is identical, but the TD error is now 1000× larger, the gradients explode and Q diverges. Three fixes compared: reward scaling, reward clipping (what DQN did), and return normalisation (what PopArt does). Clipping is the interesting one — it is not policy-preserving, and on a reward structure with meaningful magnitude differences it changes what the agent learns.
3. **Advantages.** PPO with and without per-batch advantage normalisation. Also the failure at the other end: normalising over a batch of one, where the advantage becomes exactly zero and learning silently stops with no error and no NaN.
4. **Value targets.** The critic regressing on unnormalised returns while the policy loss expects a well-scaled advantage. Shows up as the value loss dominating the total loss by three orders of magnitude.
5. **Gradient norm.** Clipping the gradient norm versus normalising it. Not the same operation, and the difference matters when the gradient is usually small and occasionally huge.

The lesson to land: **normalisation changes nothing about the maths and everything about whether it works.** That is why it is invisible when you read the paper and fatal when you write the code.

### `diagnostics/` — the toolkit

- `probe_envs.py` — Andy Jones' probe environments. Five environments of one or two states that each isolate one part of the agent: does the value head learn a constant, does it condition on observation, does it discount, does the policy respond to action-dependent reward. If DQN fails on probe 3 you know exactly where to look. This is the highest-value tool in the repo and almost nobody teaches it.
- `value_error_vs_truth.py` — compare learned Q against exact value iteration on the gridworld. Plot max error and mean error over training. The only way to distinguish "learning slowly" from "learning the wrong thing".
- `overfit_one_batch.py` — can the network fit a single transition to zero loss? If not, nothing downstream matters.
- `health_panel.py` — the standard set of numbers to log every update: gradient norm, value-function explained variance, policy entropy, approximate KL, clip fraction, TD-error distribution, replay-buffer age. Plus what each one looks like when it goes wrong.
- `seed_sweep.py` — run N seeds, plot median with the inter-quartile range. Any claimed improvement has to clear the seed noise band. Deep RL results that ignore this are the norm and are mostly wrong.

### `notebooks/`

Three guided walkthroughs only, where inline plots earn their keep:

1. `01_debug_a_dqn.ipynb` — start from a broken DQN, work down the probe environments, find the bug.
2. `02_sparse_vs_dense.ipynb` — the same gridworld under four reward modes, including watching the misspecified one get hacked.
3. `03_reading_the_health_panel.ipynb` — a PPO run that collapses, diagnosed from the logged numbers alone.

## Papers

**The point of this directory is not the PDFs. It is: what did this paper actually solve?**

A paper's contribution is a problem that nothing before it could handle.
Read as prose that is abstract; run as a failing case that a one-line change fixes, it is obvious.
So every paper gets a `demo.py` structured the same way:

1. **The problem, failing.** The prior method, on the smallest environment that breaks it. Runs in seconds. You watch it fail.
2. **The fix.** The paper's idea, changed as little as possible from step 1. Ideally one function.
3. **The same measurement on both**, side by side, exact where the environment allows.

If the paper's claim cannot be made to show up on a toy problem, that is worth knowing too, and the notes say so rather than pretending.

Layout per paper: `papers/<tier>/<key>/` containing `paper.pdf` (gitignored), `notes.md`, and `demo.py`.

The demos, and the failing case each one starts from:

| Paper | The problem, failing | The fix |
| --- | --- | --- |
| Ng et al. 1999, shaping | A reasonable-looking progress bonus makes the optimal policy avoid the goal forever | The potential-based form. Already built, in `GridWorld` |
| van Hasselt 2010, Double Q | Q-learning's max over noisy estimates is biased upward, so it prefers a losing action on a stochastic MDP | Two Q tables, one selects and the other evaluates |
| Mnih 2015, DQN | Online Q-learning with a network on correlated sequential data diverges | Replay buffer and target network, added one at a time so you can see which does what |
| Schaul 2015, PER | Uniform replay spends almost every sample on transitions with no TD error | Sample by priority, correct with importance weights |
| Schulman 2015, GAE | TD(0) is biased, Monte Carlo has huge variance, and neither is best | The lambda dial, swept, with the optimum in the middle |
| Schulman 2017, PPO | A single large policy update collapses the policy and it never recovers | The clipped ratio, and what happens when you raise the epoch count until clipping cannot hold it |
| Lillicrap 2015, DDPG | Q-learning has no `max` over a continuous action space | A deterministic actor that provides the argmax |
| Fujimoto 2018, TD3 | The critic's overestimation compounds into the actor and the policy exploits its own critic's error | Twin critics, delayed actor, target smoothing, ablated separately |
| Haarnoja 2018, SAC | The policy becomes deterministic early and stops exploring | Entropy in the objective, with the temperature swept and then tuned automatically |
| van Hasselt 2016, PopArt | Multiply the reward by 1000. Nothing else changes. The agent stops learning | Normalise the value targets, preserving the outputs |
| Ross 2011, DAgger | Behaviour cloning drifts off the expert's states and its errors compound | Query the expert on the states the learner actually visits |
| Kumar 2020, CQL | Offline Q-learning assigns high value to actions the dataset never contains, and the policy chooses them | The conservative penalty on out-of-distribution actions |
| Andrychowicz 2017, HER | A sparse-reward task where the goal is never reached, so there is nothing to learn from | Relabel failures as successes for the goal that was reached |
| Burda 2018, RND | `Chain(n=20)` needs 500,000 random episodes to see the reward once | An intrinsic bonus for novelty |
| Baird 1995 | Off-policy, bootstrapping and linear function approximation, on a problem whose true value is exactly representable. It diverges anyway | Remove any one of the three |
| Shao 2024, GRPO | PPO needs a value network, and the value network is most of what goes wrong | Group-relative baseline, no critic |
| Rafailov 2023, DPO | RLHF needs a reward model, a sampling loop and a KL penalty | A closed form that needs none of them |
| Gao 2022, over-optimisation | Optimise the reward model harder. Modelled reward keeps rising, true reward turns over | Nothing fixes it. Early stopping and KL control manage it |
| Henderson 2017 | The same configuration, ten seeds, spanning "solved" to "never learned" | Report median and interquartile range, or report nothing |

Each `notes.md` has three sections: the idea in one paragraph, an equations-to-code table mapping the paper's numbered equations to `file:line` in this repo, and what the paper got wrong or what superseded it.

PDFs are fetched from arXiv and other public sources by `papers/fetch.py` and are gitignored, since the repo is public.
Original author code is linked, not vendored, with a note on whether it still runs.

Tabular: Sutton 1988 (TD), Watkins 1989 (Q-learning), Rummery & Niranjan 1994 (SARSA), Sutton 1990 (Dyna), van Hasselt 2010 (Double Q), Ng et al. 1999 (potential-based shaping).

Deep value: Mnih 2013 and 2015 (DQN), van Hasselt 2015 (Double DQN), Wang 2016 (Dueling), Schaul 2015 (PER), Hessel 2017 (Rainbow).

Policy gradient: Williams 1992 (REINFORCE), Sutton 1999 (policy gradient theorem), Schulman 2015 (TRPO), Schulman 2015 (GAE), Schulman 2017 (PPO), Mnih 2016 (A3C), Lillicrap 2015 (DDPG), Fujimoto 2018 (TD3), Haarnoja 2018 (SAC).

Imitation and offline: Pomerleau 1991 (ALVINN), Ross 2011 (DAgger), Ho & Ermon 2016 (GAIL), Ziebart 2008 (MaxEnt IRL), Fujimoto 2019 (BCQ), Kumar 2020 (CQL), Levine 2020 (offline RL survey), Kostrikov 2021 (IQL).

Exploration and curriculum: Andrychowicz 2017 (HER), Bellemare 2016 (count-based), Burda 2018 (RND), Bengio 2009 (curriculum learning), Florensa 2017 (reverse curriculum).

Practice: Henderson 2017 (Deep RL That Matters — the seeds paper), Engstrom 2020 (Implementation Matters in Deep Policy Gradients), Andrychowicz 2020 (What Matters in On-Policy RL). These three belong in the debugging half, not the reading half.

Preference-based and LLM RL: Christiano 2017 (deep RL from human preferences), Stiennon 2020 (summarisation from human feedback), Ouyang 2022 (InstructGPT), Bai 2022 (Constitutional AI), Rafailov 2023 (DPO), Ahmadian 2024 (back to basics — RLOO), Shao 2024 (DeepSeekMath — GRPO), Gao 2022 (scaling laws for reward model over-optimisation — the Goodhart curve), Singhal 2023 (length bias in RLHF).

Normalisation and scale: van Hasselt 2016 (PopArt — learning values across magnitudes), Ioffe & Szegedy 2015 (batch norm, for the idea rather than the method), and the relevant sections of Engstrom 2020, which found that observation and reward normalisation explained more of PPO's performance than the clipping the paper is named after.

Papers get fetched from arXiv and other public sources.
Anything not publicly downloadable gets a link and a note, not a scraped copy.

## What else could be added

Ordered by what I think returns most for the debugging half.

1. **Probe environments.** Listed above. Worth calling out separately because it changes how you debug: you stop staring at the reward curve and start bisecting the agent.
2. **Seeds and significance.** One experiment where the same config run 10 times spans the full range from "solved" to "never learned". Every later comparison then reports median and IQR. This is the lesson that transfers to everything else you do.
3. **Reward hacking.** The misspecified gridworld above. Small, reproducible, and the mechanism is visible in the Q-table.
4. **Covariate shift in BC, measured.** Plot state-visitation distance between the expert's states and the cloned policy's states as the episode goes on. That divergence *is* the argument for DAgger, and it turns a hand-wave into a number.
5. **Offline dataset quality axis.** Build four datasets from the same gridworld — expert, medium, random, mixed — and run BC and CQL on each. BC wins on expert data, collapses on mixed. This is the result that explains why offline RL exists.
6. **Sample efficiency as the reporting unit.** Report environment steps to threshold, not final return. Final return hides the entire point of most of these methods.
7. **A "same problem, six algorithms" page.** The same gridworld solved by MC, SARSA, Q-learning, DQN, REINFORCE and PPO, with steps-to-threshold side by side. Then the same table on CartPole. This is the map of the whole repo on one page, twice.
8. **MLflow logging throughout.** Every run logs params, metrics and the health panel to the local MLflow at :5003. Comparisons become queryable instead of a folder of PNGs, and seed sweeps stop needing bookkeeping.
9. **Regression tests.** `tests/` asserts each algorithm still reaches its known return on the gridworld within a step budget. Catches the case where a refactor quietly breaks an algorithm months later.
10. **Distributional RL (C51 or QR-DQN).** One file. Changes *what* the network predicts rather than *how* it is trained, so it is a clean contrast to everything else. Also the cleanest way to see risk-sensitivity: the same expected value can hide very different distributions.

### Further additions, grouped

**Method families not yet covered**

- **Model-based, properly.** Dyna-Q is in Tier 1, but a small MBPO-style loop — learn a dynamics model, generate short rollouts from it, train the agent on the mixture — shows the actual trade: enormous sample efficiency, bought with model bias that compounds with rollout length. Sweep the rollout length and watch it break.
- **Evolutionary, genetic and swarm methods.** Their own tier, not a footnote — they solve the same problems by discarding gradients entirely, and on low-dimensional tasks they routinely win.
  - `cem.py` — cross-entropy method. Ten lines of real content, no gradients, no discount, no credit assignment, and it beats a lot of deep RL on the pointmass. Worth running before reaching for PPO on any small problem.
  - `es.py` — OpenAI-style evolution strategies. Gradient estimate from perturbations, parallelises almost perfectly, and needs no backward pass at all.
  - `ga.py` — a genetic algorithm over policy parameters: mutation, crossover, tournament selection. The clearest demonstration that a population escapes local optima that a single trajectory cannot.
  - `pso.py` — particle swarm. Included for the contrast: swarm methods share information between candidates during the search, where ES and GA only share it between generations.
  - `novelty_search.py` — reward the agent for doing something *different* rather than something good. On a deceptive maze this beats optimising the actual objective, which is the most counter-intuitive result in the area and the sharpest illustration of what a local optimum costs.
  
  The comparison that matters: gradient-free methods scale badly with parameter count and superbly with parallelism, and they are indifferent to sparse reward, long horizons and non-differentiable objectives. `experiments/gradient_free_vs_pg.py` runs CEM, ES, GA and PPO on the same tasks and reports steps, wall-clock and parameter count together — because the crossover is a function of all three.

- **Adversarial and two-player learning.** Three distinct things share the word and they belong in one place so the difference is visible.
  - `self_play.py` — an agent trained against copies of itself on a small game. Shows non-stationarity as a first-class problem: the environment changes because the opponent learns, and every convergence guarantee from earlier tiers assumed it would not.
  - `adversarial_perturbation.py` — an adversary that perturbs the observation within a bounded budget. A policy that looks solid can be destroyed by a perturbation far smaller than the sensor noise it will meet in reality. This is the robustness half.
  - `domain_randomisation.py` — the cheap defence. Randomise the environment's parameters during training and measure the transfer gap to a held-out setting.
  - GAIL from Tier 4 is the third sense of the word, and is cross-referenced here rather than duplicated.
  
  The through-line: adversarial training is a **minimax** problem, and minimax problems do not converge the way minimisation problems do. Cycling is the expected behaviour, not a bug, and `experiments/minimax_cycling.py` shows a two-player game where both policies rotate forever while every loss curve looks stable.
- **Successor features / general value functions.** Decouple "where does the policy go" from "what is it worth", which makes transfer across reward functions almost free on the gridworld.
- **Options and hierarchy.** A two-level agent on a gridworld with rooms. Shows temporal abstraction earning its keep on exactly the sparse-reward problem where flat RL stalls.
- **Contextual bandits.** The step between bandits and full RL, and the setting a surprising number of "we need RL" problems actually are.
- **Multi-agent.** Self-play on a tiny matrix game, plus iterated prisoner's dilemma. Shows non-stationarity: the environment changes because the opponent learns, which breaks every convergence guarantee above.
- **Safe / constrained RL.** A Lagrangian-constrained variant on a gridworld with a cost signal. Reward and constraint in tension is the shape of most real deployments.

**Evaluation and rigour**

- **A results table that survives contact.** Every experiment writes a row: algorithm, environment, seeds, steps-to-threshold median and IQR, wall-clock, commit hash. Appended, never overwritten. This is how the repo stays honest six months in.
- **Deterministic replay.** Seed numpy, torch, python and the environment from one call, and assert that two runs with the same seed produce bit-identical returns. Until that holds, no debugging session can conclude anything.
- **Evaluation without exploration.** A separate eval loop with ε=0 and deterministic actions. Reporting training-time return with exploration on is extremely common and quietly wrong.
- **Wall-clock alongside sample count.** A method that is 2× more sample-efficient and 10× slower per step is not an improvement for a simulator you own.

**Practical engineering**

- **Vectorised environments.** A minimal `SyncVectorEnv`. Also the trap it introduces: auto-reset means the observation after a terminal step belongs to the *next* episode, and using it as `next_obs` in the bootstrap silently corrupts the value function. Belongs in `broken/`.
- **Checkpointing and resume.** Save optimiser state, replay buffer and RNG state, not just weights. Resume and assert the learning curve continues rather than restarts.
- **Hyperparameter sweeps done right.** A small random search, with the seed varied *inside* each configuration so the winner is not just the luckiest seed. This is the mistake that produces most unreproducible RL results.
- **Profiling.** Where the time actually goes. Usually the environment or the `.item()` calls, not the backward pass.

**Conceptual demos worth their own file**

- **Why discounting.** γ swept from 0.5 to 1.0 on the chain environment. Shows the horizon it implies, why γ=1 needs guaranteed termination, and why a too-low γ makes a solvable problem unsolvable.
- **Exploration is not ε.** ε-greedy on a chain of length 20 needs on the order of 2²⁰ steps to reach the reward once. Count-based exploration finds it immediately. The clearest possible argument that exploration is a design problem, not a hyperparameter.
- **The bias-variance dial.** n-step returns with n from 1 to ∞ on one plot: TD(0) at one end, Monte Carlo at the other, the optimum in the middle. One figure that explains eligibility traces, GAE's λ, and n-step DQN all at once.
- **Off-by-one in the discount.** Two implementations of a return calculation that differ by one power of γ. Both learn. One is subtly worse. Shows how a bug can hide inside a working agent.
- **When not to use RL.** A problem solved by RL in 100k steps and by a two-line heuristic instantly. Worth putting near the front.

## Build order

1. `envs/` plus the exact value-iteration solver, the shared `reset()` / `step()` interface, and `tests/` scaffolding.
2. Tier 0 and Tier 1 tabular algorithms. Everything checkable against ground truth, then cross-checked on FrozenLake and Taxi.
3. `experiments/cliff_sarsa_vs_qlearning.py` and the rest of the on-policy/off-policy thread. Run on both our `cliff.py` and `CliffWalking-v1`; the two curves should agree.
4. `diagnostics/` — probe environments and the health panel, built before the first neural network.
5. Tier 2 DQN family, debugged with the tools from step 4, then taken to CartPole and MountainCar.
6. `broken/` bugs 1 to 9, which are all reachable by this point.
7. Tier 3 policy gradient, plus `pointmass.py` and `Pendulum-v1` for the continuous methods.
8. Sparse / dense / shaped / misspecified reward experiments, plus HER. `MountainCarContinuous-v0` is the sparse case that needs it.
9. Tier 4 imitation and offline, plus the dataset quality axis.
10. Tier 5 curriculum and exploration.
11. `papers/` fetched and annotated alongside each tier as it is built, not in one batch at the end.

## Notes

- The repo is not currently a git repository. Worth `git init` before any of this lands, so that a broken refactor is recoverable.
- `sample.ipynb` is the PyCharm placeholder notebook and can be deleted once `notebooks/` exists. Not deleting it without asking.
- Core dependencies: numpy, torch, matplotlib, mlflow, gymnasium.
- Optional, each gated on the install working and never imported by a core file: `gymnasium[box2d]` for LunarLander, `mujoco` for the continuous-control track, `minari` for standard offline datasets.
- Wheel availability on Python 3.14 is verified for torch 2.13.0 only. The Gymnasium, Box2D, MuJoCo and Minari wheels still need checking, and Box2D in particular compiles from source more often than not. If any of them will not install on 3.14, the fallback is a second `.venv-rl` on Python 3.12 rather than dropping the track.
