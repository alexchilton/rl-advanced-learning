# Notes — Exploration, reward, curriculum

One section per paper.
Numbers are from runs on 2026-11-09; the scripts are named in each section.
Runnable version: [`notebooks/exploration_01_bonus_curriculum_her.ipynb`](../../notebooks/exploration_01_bonus_curriculum_her.ipynb).

The tier's one comparison, `experiments/curriculum_vs_exploration.py`: tabular Q-learning (α 0.5, ε 0.1, γ 0.99), environment steps until the greedy policy reaches 95% of the exact optimum from the real start, median [min, max] over 5 seeds, budget 200,000 steps.

| method | Chain(30) | GridWorld 14×14, sparse |
| --- | --- | --- |
| ε-greedy | never (0/5) | 23,000 [21,000, 30,300] |
| optimistic Q (Q₀ = 1) | 1,300 [1,100, 1,400] | 20,700 [16,700, 22,100] |
| count bonus β = 0.1 | 72,500 [59,000, 88,000] | never (0/5) |
| reverse curriculum | 16,500 [12,500, 20,000] | 20,000 [15,000, 20,500] |
| size curriculum | 850 [800, 1,200] | 1,400 [1,200, 1,550] |
| random task (start band chosen uniformly) | 58,800 [38,800, 89,300] | 15,000 [13,700, 17,900] |
| learning-progress teacher | 45,000 [34,100, 46,600] | 16,500 [12,800, 18,600] |

The reverse-curriculum row was rerun after a fix on 2026-11-09: `distance_to_goal` gave the absorbing goal state distance 1, so it was a start state, and an episode there "succeeded" in one step with no reward. It now excludes absorbing states (`tests/test_exploration.py`, `test_the_absorbing_goal_cell_is_never_a_start`). Before the fix the row read 17,500 and 19,000.

## Curriculum Learning (Yoshua Bengio, Jerome Louradour, Ronan Collobert, Jason Weston, 2009)

PDF: [`bengio-2009-curriculum.pdf`](./bengio-2009-curriculum.pdf)

### The idea in one paragraph

Train on easy examples first and harder ones later.
In the paper's supervised experiments this speeds up training and reaches better final solutions, and the authors read it as a continuation method: optimise a smoothed version of the objective first.
The RL version here grows the task: solve a short chain or a small grid, then a larger one, carrying the Q table across.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| stages of increasing size, each trained to 95% of its own optimum | the schedule | `algos/curriculum_size.py:45-66` |
| $Q_{\text{big}}(\text{cell } k \text{ steps from goal}) \leftarrow Q_{\text{small}}(\text{same})$ | goal-aligned transfer; new states start at 0 | `algos/curriculum_size.py:31-42` |

### What it got wrong / what came after

The fastest method in the table: 850 steps on Chain(30) and 1,400 on the 14×14 grid, every stage counted.
The caveat is large: every cell of a chain or an open grid looks the same relative to the goal, so a goal-aligned copy is close to a perfect transfer.
On tasks where a small instance is not a corner of the big one, there is no such mapping.

Every curriculum in the table uses a Q table, and a table cannot forget: each state has its own entry.
`algos/curriculum_failure.py` runs a curriculum with a network instead: DQN on a goal-conditioned 7×7 grid (observation = position and goal), the four corners as four stages of 30,000 steps each, every goal re-evaluated every 1,000 steps (`experiments/curriculum_failure.py`, 5 seeds).

| arm | seeds solving goals 1, 2, 3, 4 after the last stage |
| --- | --- |
| plain: replay of 2,000, current goal only | 0, 0, 0, 5 |
| rehearsal: a quarter of episodes on earlier goals | 5, 4, 4, 5 |
| big replay: 120,000, every stage kept | 5, 5, 5, 5 |

With the plain schedule each goal is solved when its own stage ends and lost by the end of the next one; only the last goal survives.
A curriculum that evaluates only the current stage reports success the whole way through.
A large replay buffer is rehearsal by another name, which is one reason ordinary DQN runs rarely show this.

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| goal of each episode: an earlier stage's with probability ρ | rehearsal | `algos/curriculum_failure.py:98-104` |
| greedy success on every goal, every 1,000 steps | the forgetting curve | `algos/curriculum_failure.py:69-83`, `133` |

## Unifying Count-Based Exploration and Intrinsic Motivation (Marc G. Bellemare et al., 2016)

PDF: [`bellemare-2016-count-based.pdf`](./bellemare-2016-count-based.pdf)

### The idea in one paragraph

Add $\beta / \sqrt{N(s,a)}$ to the reward, so rarely tried actions look valuable.
In Atari no state repeats, so the paper derives a pseudo-count from a density model of observations.
With a finite state space the real count exists, and the tabular bonus is the exact special case.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $N(s,a) \mathrel{+}= 1$ | the count | `algos/count_based_exploration.py:73` |
| $r + \beta/\sqrt{N(s,a)}$ | the bonus, added to the reward | `algos/count_based_exploration.py:75` |
| $Q_0 = 1$ everywhere | optimistic initialisation, the baseline | `algos/count_based_exploration.py:52` |

### What it got wrong / what came after

On Chain(30), where ε-greedy never reaches the goal, the bonus solves it in 72,500 steps, and optimistic initialisation in 1,300.
On the 14×14 grid it **never** solves within 200,000 steps, while ε-greedy needs 23,000.
Measured on two seeds: the bonus run first reaches the goal at 87,583 and 43,089 steps, and its Q at the start is 1.198 and 1.190 against an exact optimum of 0.778.
The bonus is part of the reward, so the greedy policy optimises goal plus bonus; 20,000 further steps with the bonus off do not bring the inflated values down.
Smaller β (0.01, 0.003) solves the grid in 73,000 to 140,500 steps, still slower than ε-greedy.
β = 0.01 and β = 0.003 give identical step counts on every seed: before the first reward every Q value is β times a constant, so the greedy choices depend only on the bonus's sign.
After: RND (below) replaces the count with a prediction error.

## Hindsight Experience Replay (Marcin Andrychowicz et al., 2017)

PDF: [`andrychowicz-2017-her.pdf`](./andrychowicz-2017-her.pdf)

### The idea in one paragraph

On a sparse goal-reaching task, store each transition again with the goal replaced by a state actually reached later in the episode, and the reward recomputed for that goal.
Failed episodes become successes for some goal, so a goal-conditioned value function learns from the first episode.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $g' = s_{t'}$, $t' \sim \mathcal{U}\{t, \dots, T\}$ | the "future" relabelling strategy, k = 4 copies | `algos/her.py:70-74` |
| $r' = \text{compute\_reward}(s_{t+1}, g')$, terminal if 0 | reward recomputed for the new goal | `algos/her.py:75-77` |
| $y \in [-1/(1-\gamma), 0]$ | target clipped to the achievable range | `algos/her.py:84` |
| bit flipping, n bits, reward −1 until the state equals the goal | the paper's motivating task | `envs/bit_flip.py` |

### What it got wrong / what came after

Measured (`python algos/her.py`), greedy success rate on 50 fresh goals after 4,000 episodes, seed 0:

| bits | DQN (k = 0) | DQN + HER (k = 4) |
| --- | --- | --- |
| 5 | 100% | 100% |
| 10 | 0% | 100% |
| 15 | 0% | 100% |
| 20 | 0% | 100% |

Above 5 bits a random policy essentially never matches a random goal, so plain DQN never sees a reward of 0.
HER needs a goal-conditioned task and a way to recompute the reward for any goal, which most environments do not expose.

## Reverse Curriculum Generation for Reinforcement Learning (Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel, 2017)

PDF: [`florensa-2017-reverse-curriculum.pdf`](./florensa-2017-reverse-curriculum.pdf)

### The idea in one paragraph

Start episodes next to the goal, where the agent reaches it by accident, and move the start states outward as it becomes reliable.
Every early episode ends in reward, and each new ring of starts sits next to states whose values are already known.
It requires resetting the environment to arbitrary states.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| BFS on the transition table from terminating states, absorbing states excluded | distance to the goal | `algos/curriculum_reverse.py:37-65` |
| starts = states with distance ≤ d | the current ring | `algos/curriculum_reverse.py:93` |
| d += 1 when greedy rollouts from those starts succeed ≥ 80% | the schedule | `algos/curriculum_reverse.py:68-81`, `algos/curriculum_reverse.py:100` |

The paper picks new starts by random walks from the goal; this file uses the BFS distance, which is exact on a finite MDP.

### What it got wrong / what came after

16,500 steps on Chain(30), where ε-greedy never succeeds, and 20,000 on the 14×14 grid against ε-greedy's 23,000.
It helps most where exploration is the whole problem, and costs little where it is not.
Its decision to move outward uses extra greedy rollouts that are not counted as steps; the automatic curricula below use only their training episodes.
The arbitrary-reset requirement is the real limit: a simulator allows it, most physical or production systems do not.

## Teacher-Student Curriculum Learning (Tambet Matiisen, Avital Oliver, Taco Cohen, John Schulman, 2017)

PDF: [`matiisen-2017-teacher-student.pdf`](./matiisen-2017-teacher-student.pdf)

### The idea in one paragraph

A teacher chooses which task the student trains on next, as a non-stationary bandit whose reward is learning progress: the slope of the student's score on that task.
The absolute value of the slope is used, so a task whose score is falling (forgetting) is practised again.
The paper's Window algorithm keeps the last K scores per task and fits the slope by regression against timesteps.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| tasks = start states grouped into 6 bands of distance to the goal | the task set | `algos/curriculum_automatic.py:52-59` |
| $LP_k = \lvert \overline{o}_{\text{newer half}} - \overline{o}_{\text{older half}} \rvert$ over the last 10 outcomes | learning progress | `algos/curriculum_automatic.py:67-76` |
| argmax LP, uniform with probability 0.2 | the teacher's choice | `algos/curriculum_automatic.py:78-82` |
| outcome = goal reached in that training episode | the score | `algos/curriculum_automatic.py:129` |

Simplified from the paper: the half-window difference replaces the regression slope, and there is no bandit smoothing.

### What it got wrong / what came after

Same table as the rest of this tier (top of this file): the teacher beats random band choice on Chain(30), 45,000 against 58,800 steps, and loses to it on the 14×14 grid, 16,500 against 15,000.
The paper does compare with uniform task sampling: it reports its teacher an order of magnitude faster on the Minecraft maze, and uniform sampling "competitive" on 2-D addition. Here it is the same split: on the grid uniform choice is the best of the three start-distribution methods, ahead of the hand-written reverse schedule (20,000).

## Exploration by Random Network Distillation (Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov, 2018)

PDF: [`burda-2018-rnd.pdf`](./burda-2018-rnd.pdf)

### The idea in one paragraph

Novelty as prediction error: a fixed random network $f$ and a predictor $\hat f$ trained to match it on visited states.
$\lVert \hat f(s) - f(s)\rVert^2$ is large for states not yet seen, and it generalises between similar states, which a count table cannot.
The paper gives the intrinsic reward its own value head and treats it as non-episodic.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $r_i = \lVert \hat f(s') - f(s')\rVert^2 / \sigma$ | novelty, normalised by a running std, recomputed per batch | `algos/rnd.py:120` |
| $y_E = r + \gamma(1-d)\max Q_E^-$, $y_I = r_i + \gamma_I \max Q_I^-$ | two heads; the intrinsic one ignores episode ends | `algos/rnd.py:117-121` |
| act on $Q_E + \beta Q_I$, evaluate $Q_E$ alone | behaviour vs the task's policy | `algos/rnd.py:94` |

### What it got wrong / what came after

Measured (`python experiments/rnd_chain.py`), Chain(n) with one-hot observations and the action indices scrambled per state (`envs.chain.ScrambledActions`), 50,000 steps, 5 seeds:

| | goal first reached (step) | greedy policy optimal at the end |
| --- | --- | --- |
| DQN, Chain(20) | 11,403 and 48,742 on 2 seeds; never on 3 | 2 / 5 |
| DQN + RND, Chain(20) | 1,701 to 3,802 | 5 / 5 |
| DQN, Chain(30) | 36,898 and 48,453 on 2 seeds; never on 3 | 0 / 5 |
| DQN + RND, Chain(30) | 4,287 to 4,903 | 5 / 5 |

The first version of this table was measured on the plain chain, where "right" is action 1 in every state, and had DQN optimal on 4 / 5 (Chain(20)) and 2 / 5 (Chain(30)).
Those successes were mostly initialisation, not exploration: before any training, 3 of 5 fresh DQN networks prefer "right" in every state of Chain(20).
Scrambling the action meaning per state removes that shortcut. DQN drops to 2 / 5 and 0 / 5; RND stays at 5 / 5 on both.

A single-head version, with the bonus added to the reward, was tried first and failed in an informative way (measured on the plain chain, before the scrambling).
On Chain(20) it reached the goal at 2,281 to 4,023 steps (DQN: 3,750 to 9,155 when it succeeds) and ended with a greedy return of 0 on 4 of 5 seeds.
Fading the bonus to zero and recomputing it per batch did not help.
Its learned Q was flat at about 1.438 for both actions in every state, above the largest achievable return of 1.0, so the greedy action was noise.
The same inflation is what stops the tabular count bonus on the grid.
