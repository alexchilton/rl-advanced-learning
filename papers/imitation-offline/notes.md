# Notes — Imitation and offline

One section per paper.
Every number below is an exact return on the default `Racetrack` (7,126 states, exact optimum −6.54 at γ = 0.99), from running the file named.
Runnable versions: [`notebooks/imitation_01_bc_dagger_irl_gail.ipynb`](../../notebooks/imitation_01_bc_dagger_irl_gail.ipynb) and [`notebooks/offline_01_q_cql_bcq_iql.ipynb`](../../notebooks/offline_01_q_cql_bcq_iql.ipynb).

## Efficient Training of Artificial Neural Networks for Autonomous Navigation (Dean A. Pomerleau, 1991)

PDF: [`pomerleau-1991-alvinn.pdf`](./pomerleau-1991-alvinn.pdf)

### The idea in one paragraph

Drive a car by supervised learning on (camera image, human steering) pairs: behaviour cloning.
The paper's own finding is the important one.
A network trained only on the human's good driving never sees what to do after a mistake, so Pomerleau synthesised shifted, off-road views and labelled them with the correcting action.
That is the covariate-shift problem, and a hand-made fix for it, twenty years before DAgger made the fix general.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\pi(s) = a_{\text{expert}}(\arg\min_{s_i \in D} \lVert \phi(s) - \phi(s_i) \rVert)$ | nearest-neighbour BC: copy the action of the closest demonstrated state | `algos/bc.py:58-81` |
| $\phi(s) = (\text{row}, \text{col}, v_\uparrow, v_\rightarrow) / \text{scale}$ | the features the distance is measured in | `algos/bc.py:84-96` |
| $\text{shift} = \mathbb{E}_{s \sim \pi}\ \min_{s_i \in D} \lVert \phi(s) - \phi(s_i) \rVert$ | covariate shift: how far the clone drifts from the demonstrations | `algos/bc.py:99-112` |

### What it got wrong / what came after

BC has no idea what is good, only what was done, so it cannot exceed the demonstrator and its errors compound.
Measured in `experiments/bc_vs_dagger.py`, 5 seeds: BC's median is −12.93 at both 160 and 320 labels; DAgger's is −6.77 and −6.64.
One seed can look almost optimal (seed 1 reaches −6.87 at 160 labels), which is why a single BC run proves nothing.
Ross et al. 2011 (below) turned the synthetic-correction idea into DAgger.

## Maximum Entropy Inverse Reinforcement Learning (Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey, 2008)

PDF: [`ziebart-2008-maxent-irl.pdf`](./ziebart-2008-maxent-irl.pdf)

### The idea in one paragraph

Recover a reward that explains the demonstrations, instead of copying the actions.
Many rewards explain any behaviour, so pick the one whose trajectory distribution has maximum entropy while matching the expert's feature counts.
The gradient of the demonstrations' likelihood is "expert feature counts minus the feature counts the current reward's soft-optimal policy would produce".
It needs the transition model.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $r(s) = \phi(s)^\top \theta$ | reward linear in features; one indicator per track cell, 118 parameters | `algos/max_ent_irl.py:181`, features `:195-213` |
| $V(s) = \log \sum_a \exp Q(s,a)$ | soft value iteration: `logsumexp` where value iteration has `max` | `algos/max_ent_irl.py:69-85` (the `logsumexp` is line 82) |
| $\mu_{t+1}(s') = \sum_{s,a} \mu_t(s)\,\pi(a \mid s)\,P(s' \mid s,a)$ | expected state visitation of the soft-optimal policy | `algos/max_ent_irl.py:88-98` |
| $\nabla_\theta = \phi^\top (\mu_{\text{expert}} - \mu_\theta)$ | the whole learning rule | `algos/max_ent_irl.py:184` |
| $\theta \leftarrow \theta - \bar\theta$ | pin the additive constant the reward is not identified by | `algos/max_ent_irl.py:187` |

### What it got wrong / what came after

Measured (`python algos/max_ent_irl.py`), demonstrator −8.32 (40% random actions), optimum −5.83 at γ = 0.95: with 10 demonstrations BC scores −13.52 and IRL-then-plan −5.87; with 300, −5.84 and −5.83.
The win comes from structure (118 parameters instead of 7,126 per-state choices), not from imitation as such: the file records that on a 7×7 grid BC needed 5 demonstrations and IRL 20.
A detail the paper does not stress and that broke the first version here: demonstrations must be padded to the full horizon, or the goal looks over-visited and its reward is pushed down (`algos/max_ent_irl.py:116-148`; measured 0.0000 against a demonstrator worth 0.1370 before the fix).
After: GAIL (below) drops the transition model; deep IRL methods (guided cost learning, AIRL) replace the linear reward with a network.

## A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, 2011)

PDF: [`ross-2011-dagger.pdf`](./ross-2011-dagger.pdf)

### The idea in one paragraph

Behaviour cloning trains on the expert's states and is tested on its own; errors therefore compound, and the expected cost grows with the square of the horizon.
DAgger lets the learner drive, asks the expert what it would have done in each state the learner reached, adds those labels to the dataset, and refits.
Training on the learner's own state distribution makes the cost grow linearly in the horizon.
The price is that the expert must be available during training, not recorded once.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\pi_i = \beta_i\,\pi^* + (1 - \beta_i)\,\hat\pi_i$ | mixed collection policy: follow the expert with probability β | `algos/dagger.py:101-103` |
| $\beta_i = 0.6^i$ | the decay schedule | `algos/dagger.py:92`, `:97` |
| $D \leftarrow D \cup \{(s, \pi^*(s)) : s \sim \pi_i\}$ | aggregate expert labels on the learner's states | `algos/dagger.py:123-124` |
| fixed labels per round | subsample so each round costs the same expert queries | `algos/dagger.py:119-121` |

### What it got wrong / what came after

The β schedule reads like a detail and is not: with β = 0 from round one, DAgger needed 2,426 labels to reach what the decayed schedule reaches in 80 (recorded in `algos/dagger.py:161-165`).
On a matched label budget (5 seeds) DAgger reaches −6.81 at 80 labels where BC's median is −17.39.
The real-world limit is the expert query: free with a scripted controller, costly with a human.
After: HG-DAgger and SafeDAgger let a human intervene only when needed; GAIL (below) replaces the expert queries with environment interaction.

## Generative Adversarial Imitation Learning (Jonathan Ho, Stefano Ermon, 2016)

PDF: [`ho-ermon-2016-gail.pdf`](./ho-ermon-2016-gail.pdf)

### The idea in one paragraph

Train a discriminator to tell expert (state, action) pairs from the learner's, and train the policy with RL on a reward built from the discriminator.
At the optimum the learner's occupancy measure matches the expert's.
GAIL needs environment interaction but no reward, no transition model, and no expert on call.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\max_D\ \mathbb{E}_{\pi_E}[\log D] + \mathbb{E}_{\pi}[\log(1 - D)]$ | logistic discriminator; expert = 1, learner = 0, with weight decay | `algos/gail.py:168-181` |
| $D(s,a) = \sigma(w_{\text{cell}(s), a})$ | low-capacity discriminator: track cell and action, velocity ignored | `algos/gail.py:56-83` |
| $r = -\log(1 - D)$, $\log D$, or $\log D - \log(1 - D)$ | three forms of the manufactured reward | `algos/gail.py:106-128` |
| $Q \leftarrow Q + \alpha(r + \gamma \max Q' - Q)$ | inner RL loop: Q-learning on the manufactured reward | `algos/gail.py:183-189` |

### What it got wrong / what came after

The sign of the reward is a prior on episode length, not part of imitation.
Measured, 30 demonstrations, 3 seeds: the positive form $-\log(1 - D)$ scores −100.00 on every seed (the car never finishes, because every step pays); the negative form scores −11.37, −12.94 and −18.83.
Kostrikov et al. 2019 (DAC) fix this by learning a reward for the absorbing state.
GAIL wins with very little data (3 demonstrations: −11.92 against BC's −68.78) and gets worse with more (100 demonstrations: −27.03 against BC's −7.02), because the policy chases a reward that moves every round.
After: AIRL recovers a transferable reward; DAC and ValueDICE fix the absorbing-state bias and the sample cost.

## Off-Policy Deep Reinforcement Learning without Exploration (Scott Fujimoto, David Meger, Doina Precup, 2019)

PDF: [`fujimoto-2019-bcq.pdf`](./fujimoto-2019-bcq.pdf)

### The idea in one paragraph

Offline Q-learning fails because the max in the target picks actions the data never took, whose values are pure extrapolation.
BCQ restricts the max to actions the behaviour policy plausibly takes at that state.
The paper names the problem *extrapolation error*; the tabular version is batch-constrained Q-learning.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\mathcal{A}(s) = \{a : \hat b(a \mid s) \ge \tau \max_{a'} \hat b(a' \mid s)\}$ | admissible actions, relative to the most-taken one | `algos/bcq_tabular.py:57-67` |
| $y = r + \gamma \max_{a' \in \mathcal{A}(s')} Q(s',a')$ | the constrained backup | `algos/bcq_tabular.py:99-102` |
| $\pi(s) = \arg\max_{a \in \mathcal{A}(s)} Q(s,a)$ | constrained policy extraction | `algos/bcq_tabular.py:113-125` |
| $V(\text{never acted from}) = -1/(1-\gamma)$ | pessimistic value for states with no data | `algos/bcq_tabular.py:54` |

### What it got wrong / what came after

The ablation (`python algos/bcq_tabular.py`, medium data) shows that the action constraint does all the work, −98.73 to −6.61, and the pessimism setting changes nothing.
The reason: in trajectory data, no bootstrapped next state is ever "never acted from" (0 such states in the log).
A common silent bug: train with the constraint, then extract with an unrestricted argmax.
After: CQL and IQL (below) reach the same place by other mechanisms; BEAR uses a distribution-matching constraint instead of a hard support.

## Conservative Q-Learning for Offline Reinforcement Learning (Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine, 2020)

PDF: [`kumar-2020-cql.pdf`](./kumar-2020-cql.pdf)

### The idea in one paragraph

Instead of restricting the max, push the Q values of all actions down and the data's actions up.
The learned Q is then a lower bound on the true value of the policy, so the max no longer selects extrapolated optimism.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\alpha\,\mathbb{E}_s[\log \sum_a \exp Q(s,a) - Q(s, a_{\text{data}})]$ | the CQL(H) penalty | docstring `algos/cql_tabular.py:9` |
| $\nabla_Q = \alpha(\mathrm{softmax}\,Q(s,\cdot) - \mathbf{1}_{a_{\text{data}}})$ | its gradient, applied each sweep | `algos/cql_tabular.py:70-81` |
| ordinary TD target $r + \gamma \max Q'$ | unchanged; α = 0 is naive offline Q | `algos/cql_tabular.py:58-64` |

### What it got wrong / what came after

α is a dial with no safe end.
Measured on medium data (behaviour policy −13.50): α = 0 gets −98.73 while believing −1.84; α = 0.5 to 5 gets −6.54 to −6.56; α = 100 gets −45.08 while "believing" +536, on a task where no return can exceed 0.
So a conservative Q is not a value estimate, and α cannot be tuned without the online evaluation offline RL forbids.
After: Cal-QL calibrates the conservative values for online fine-tuning; IQL avoids the penalty altogether.

## Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems (Sergey Levine, Aviral Kumar, George Tucker, Justin Fu, 2020)

PDF: [`levine-2020-offline-rl-survey.pdf`](./levine-2020-offline-rl-survey.pdf)

### The idea in one paragraph

A survey that frames offline RL as distribution shift: the learned policy is evaluated on actions the data does not contain.
It sorts the fixes into policy constraints (BCQ, BEAR), conservative values (CQL) and model-based methods, and argues that offline RL is a separate problem, not online RL with collection removed.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $y = r + \gamma \max_{a'} Q(s',a')$ over a fixed log | naive fitted Q iteration, the failure the survey starts from | `algos/offline_q.py:44-62` |
| greedy action outside the data's support | the diagnostic, weighted by state visits | `algos/offline_q.py:65-83` |
| believed value against exact value | extrapolation error in return units | `algos/offline_q.py:86-96` |

### What it got wrong / what came after

Measured (`python algos/offline_q.py`, 20,000 transitions): naive offline Q believes −1.02 to −2.35 on all four datasets and actually gets −100.00, −98.73, −93.07 and −59.08.
Its section on dataset quality is the one `experiments/offline_dataset_quality.py` reproduces: on expert data behaviour cloning is as good as anything.
After: IQL (below), Decision Transformer, and offline-to-online methods (`algos/offline_to_online.py`).

## Offline Reinforcement Learning with Implicit Q-Learning (Ilya Kostrikov, Ashvin Nair, Sergey Levine, 2021)

PDF: [`kostrikov-2021-iql.pdf`](./kostrikov-2021-iql.pdf)

### The idea in one paragraph

Never evaluate an action outside the data.
Learn $V(s)$ by expectile regression on the Q values of the actions the data took; as the expectile τ → 1 this approaches the max over in-sample actions.
The Q target is $r + \gamma V(s')$, with no max anywhere.
Extract the policy by advantage-weighted regression over the data's actions.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $L_\tau(u) = \lvert \tau - \mathbf{1}(u < 0) \rvert\, u^2$, $u = Q(s,a) - V(s)$ | expectile regression for V | `algos/iql_tabular.py:74-78` |
| $Q(s,a) \leftarrow r + \gamma V(s')$ | the max-free target | `algos/iql_tabular.py:82-88` |
| $\pi(a \mid s) \propto \exp(\beta (Q - V))$ over seen actions | advantage-weighted extraction | `algos/iql_tabular.py:93-120` |

### What it got wrong / what came after

Measured on random data (behaviour policy −32.05): τ = 0.5 to 0.9 gets −8.75 to −9.46; τ = 0.95 collapses to −32.50 and τ = 0.99 to −98.51, with $V(\text{start}) = -1.47$, the same optimistic lie as naive offline Q.
IQL moves the failure from unseen actions to thinly seen ones, where τ controls it.
β = 0 is behaviour cloning and scores −99.96 on this data.
After: IQL is the usual starting point for offline-to-online fine-tuning; see `algos/offline_to_online.py` for the handover problem with a CQL start.
