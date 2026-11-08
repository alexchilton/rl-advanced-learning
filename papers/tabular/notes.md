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
The max causes two later problems: overestimation (Double Q-learning, below, not built here) and extrapolation error once the data is fixed (`algos/offline_q.py`).
After: SARSA (on-policy, below), Double Q-learning, DQN.

## Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming (Richard S. Sutton, 1990)

PDF: [`sutton-1990-dyna.pdf`](./sutton-1990-dyna.pdf)

Not built in this repo yet. There is no Dyna file; the plan lists it in `docs/rl-curriculum-plan.md`.

## On-Line Q-Learning Using Connectionist Systems (G. A. Rummery, M. Niranjan, 1994)

PDF: [`rummery-niranjan-1994-sarsa.pdf`](./rummery-niranjan-1994-sarsa.pdf)

Not built in this repo yet.
`algos/q_learning.py:12` refers to `algos/sarsa.py`, which does not exist.

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

Not built in this repo yet.
