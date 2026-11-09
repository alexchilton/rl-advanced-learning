# Notes — Tabular

One section per paper.
Papers with code in this repo have the full three sections; the others say what is missing.
Runnable version: [`notebooks/tabular_01_q_learning.ipynb`](../../notebooks/tabular_01_q_learning.ipynb) and [`notebooks/00_ground_truth.ipynb`](../../notebooks/00_ground_truth.ipynb).

## Learning to Predict by the Methods of Temporal Differences (Richard S. Sutton, 1988)

PDF: [`sutton-1988-td.pdf`](./sutton-1988-td.pdf)

### The idea in one paragraph

Learn a prediction from the difference between successive predictions, not from the final outcome.
After each step, move $V(s)$ towards $r + \gamma V(s')$.
That bootstraps: it uses one estimate to improve another, so learning happens before the episode ends.
The paper proves TD(0) converges for linear predictors and argues it is more data-efficient than Monte Carlo on Markov problems.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\delta = r + \gamma V(s') - V(s)$ | the TD error | in its control form at `algos/q_learning.py:94-95` |
| $V^\pi = R^\pi + \gamma P^\pi V^\pi$ | the fixed point TD estimates, solved exactly | `envs/solvers.py:120` (`policy_evaluation`) |
| $\gamma V(s')$ only if not terminated | terminal masking | `algos/q_learning.py:92-94` |

### What it got wrong / what came after

There is no standalone TD(0) prediction file here; the TD error appears only inside Q-learning and the offline methods.
Notebook 00 shows a limit of tabular tests of the terminal mask: on GridWorld, deleting it changes nothing (max error 0.000000), because the absorbing goal has value 0 anyway.
After: TD(λ) and eligibility traces; Q-learning (below) for control.

## Learning from Delayed Rewards (Christopher J. C. H. Watkins, 1989)

PDF: [`watkins-1989-qlearning.pdf`](./watkins-1989-qlearning.pdf)

### The idea in one paragraph

Q-learning: learn the action-value function of the **optimal** policy while following any exploring policy.
The target takes the max over the next state's actions, whatever the agent actually does next, which makes it off-policy.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$ | the update | `algos/q_learning.py:94-95` |
| ε-greedy behaviour | exploration | `algos/q_learning.py:58-61` |
| argmax with random tie-breaking | not in the paper; needed on a zero-initialised table | `algos/q_learning.py:41-55` |
| $J(\pi_{\text{greedy}})$ computed exactly | the learning curve | `algos/q_learning.py:84`, `:100` |

### What it got wrong / what came after

The tie-break is an implementation detail with a large effect.
Measured in `notebooks/tabular_01_q_learning.ipynb`, sparse 8×8 GridWorld, 3 seeds: `np.argmax` ends at 0.0 on every seed; random ties end at the optimum 0.5133 on every seed.
The max causes two later problems: overestimation (Double Q-learning, below, `algos/double_q.py`) and extrapolation error once the data is fixed (`algos/offline_q.py`).
After: SARSA (on-policy, below), Double Q-learning, DQN.

## Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming (Richard S. Sutton, 1990)

PDF: [`sutton-1990-dyna.pdf`](./sutton-1990-dyna.pdf)

### The idea in one paragraph

Learn a model of the environment from real experience and use it to generate extra, simulated updates.
After each real step, Dyna-Q stores the transition in a table and replays n remembered transitions through the same Q-learning update.
Learning, planning and acting share one value function; planning is just more updates from imagined data.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| model[s, a] ← (r, s′, terminated) | the learned model: last outcome per pair | `algos/dyna_q.py:53` |
| n × Q-learning update on a random remembered (s, a) | planning | `algos/dyna_q.py:55-58` |

### What it got wrong / what came after

Measured (`python algos/dyna_q.py`, 10×10 sparse grid with a wall, 5 seeds), real steps until the greedy policy is optimal: 0 planning steps 9,300 (median), 5 → 1,800, 20 → 2,300, 50 → 1,000.
Planning cuts the real experience needed by up to 9 times; the 20-step median is pulled up by one seed at 9,000.
The model remembers one outcome per pair, which is exact here and wrong on a stochastic task; it also never notices when the world changes (Sutton & Barto's Dyna-Q+ adds a bonus for that).
After: prioritised sweeping, and learned-model methods such as MuZero.

## On-Line Q-Learning Using Connectionist Systems (G. A. Rummery, M. Niranjan, 1994)

PDF: [`rummery-niranjan-1994-sarsa.pdf`](./rummery-niranjan-1994-sarsa.pdf)

### The idea in one paragraph

SARSA bootstraps from the action the agent will actually take next, not from the best one.
That makes it on-policy: it learns the value of the epsilon-greedy policy it follows, exploration included.
The name is the tuple each update uses: (s, a, r, s', a').
The report called it "modified connectionist Q-learning"; Sutton coined "SARSA" later.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $a' \sim \epsilon\text{-greedy}(Q(s', \cdot))$, chosen before the update | the next action, which the target uses | `algos/sarsa.py:75` |
| $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma Q(s',a') - Q(s,a))$ | the update | `algos/sarsa.py:76-77` |
| $\pi_\epsilon(a \mid s)$ | the policy SARSA's Q is the value of | `envs/solvers.py:199` (`epsilon_greedy_policy`) |

### What it got wrong / what came after

Measured (`python experiments/cliff_sarsa_vs_qlearning.py`, 4×12 cliff, α = 0.2, ε = 0.1, 30,000 steps, 5 of 5 seeds): Q-learning's greedy path is the 13-step edge (exact return −12.248), SARSA's the 17-step safe path (−15.706).
The policies they actually follow while exploring score the other way round: Q-learning −46.6, SARSA −20.4 (medians).
SARSA's target contains the random next action, so it needs a smaller α: at α = 0.5 and 30,000 steps its greedy policy loops forever on 2 of 5 seeds.
After: Expected SARSA replaces $Q(s',a')$ with its expectation under the policy, removing that variance.

## Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping (Andrew Y. Ng, Daishi Harada, Stuart Russell, 1999)

PDF: [`ng-1999-shaping.pdf`](./ng-1999-shaping.pdf)

### The idea in one paragraph

Adding a shaping term $F(s, s') = \gamma \Phi(s') - \Phi(s)$ to the reward cannot change the optimal policy, for any potential function Φ.
The paper also shows the converse: any shaping term that is not of this form can change it, for some MDP.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $r' = r + \gamma \Phi(s') - \Phi(s)$, $\Phi(s) = -0.1\,d(s)$ | `shaped` mode | `envs/gridworld.py:197-203` |
| $\Phi(\text{goal}) = 0$ | required for the guarantee | `envs/gridworld.py:202` |
| $r' = r + 0.3 \max(0, d(s) - d(s'))$ | `misspecified`: one-sided, not potential-based | `envs/gridworld.py:205-209` |
| $r' = r - 0.01 + 0.1(d(s) - d(s'))$ | `dense`: not policy-invariant in general | `envs/gridworld.py:190-195` |

### What it got wrong / what came after

The guarantee is about the optimal policy, not about how fast it is learned.
Measured (`python experiments/sparse_vs_dense.py`, 3 seeds, steps to 95% of the sparse optimum): on 14×14, sparse takes 35,000 steps and shaped takes 50,000; dense takes 5,000 at every size.
The mechanism is in `shaping_margin` in that file: shaping inflates every value by $-\Phi(s)$ and leaves the margin between "solve" and "loiter" equal to the sparse return.
The converse half of the paper is the `misspecified` mode: the one-sided bonus is never solved at any size, and at γ ≥ 0.9 its optimal policy never enters the goal (notebook 00).

## Double Q-learning (Hado van Hasselt, 2010)

PDF: [`vanhasselt-2010-doubleq.pdf`](./vanhasselt-2010-doubleq.pdf)

### The idea in one paragraph

The max over noisy, unbiased estimates is biased upwards: $\mathbb{E}[\max_a \hat Q(a)] \ge \max_a \mathbb{E}[\hat Q(a)]$.
Keep two tables. Each step a coin picks one to update; that table chooses the best next action and the other one scores it.
Their errors are independent, so the lucky error that made an action look best is not the one used to value it.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $a^* = \arg\max_{a'} Q_1(s', a')$ | the updated table selects | `algos/double_q.py:67` |
| $Q_1(s,a) \leftarrow Q_1(s,a) + \alpha(r + \gamma Q_2(s', a^*) - Q_1(s,a))$ | the other table evaluates | `algos/double_q.py:68-69` |
| act ε-greedily on $Q_1 + Q_2$ | behaviour | `algos/double_q.py:55` |

The test MDP is Sutton & Barto's Example 6.7, in `envs/maximization_bias.py`: going left is worth −0.1, right 0.

### What it got wrong / what came after

Measured over 2,000 runs (`python algos/double_q.py`), share of episodes going left from A, where ε-greedy alone gives 5%:

| episodes | Q-learning | Double Q |
| --- | --- | --- |
| 1–10 | 69.8% | 50.3% |
| 11–50 | 89.7% | 26.0% |
| 51–100 | 61.9% | 10.2% |
| 201–300 | 13.5% | 6.8% |

Each table sees half the data, so Double Q learns each value more slowly.
After: Double DQN (`papers/deep-value/notes.md`) uses DQN's online and target networks as the two estimators.
