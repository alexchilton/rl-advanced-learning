# Notes — Deep value

One section per paper.
Numbers are from `experiments/deep_value_sweep.py` on 2026-11-09: CartPole-v1, 100,000 steps, 5 seeds per configuration, greedy return over 10 fresh episodes.
"Mean over curve" is the greedy return averaged over the 20 evaluations, which is less noisy than the final one.
Runnable version: [`notebooks/deep_value_01_dqn_family.ipynb`](../../notebooks/deep_value_01_dqn_family.ipynb).

## Playing Atari with Deep Reinforcement Learning (Volodymyr Mnih et al., 2013)

PDF: [`mnih-2013-atari-dqn.pdf`](./mnih-2013-atari-dqn.pdf)

### The idea in one paragraph

Q-learning with a convolutional network reading raw pixels, trained from a replay memory of past transitions sampled uniformly.
Replay breaks the correlation between consecutive samples and reuses each transition many times.
The 2013 version has no separate target network; that came in 2015.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $y = r + \gamma (1 - d) \max_{a'} Q(s', a')$ | the Q-learning target | `algos/dqn.py:155-157` |
| $\mathrm{Huber}(Q(s,a) - y)$ | the loss (the paper clips the error, which is the same thing) | `algos/dqn.py:158-159` |
| uniform sample from a ring buffer | experience replay | `algos/dqn.py:62-89` |
| ε decays linearly 1 → 0.05 | exploration | `algos/dqn.py:96-98` |

### What it got wrong / what came after

Without a target network, bootstrapping chases its own moving output.
Measured with replay but no target network (`target_update=1`): mean Q diverges to about $3 \times 10^{10}$ on all 5 seeds and the return collapses to 9.2.
The 2015 paper adds the target network.

## Human-level control through deep reinforcement learning (Volodymyr Mnih et al., 2015)

PDF: [`mnih-2015-dqn-nature.pdf`](./mnih-2015-dqn-nature.pdf)

### The idea in one paragraph

DQN as it is usually meant: the 2013 agent plus a target network $Q_{\bar\theta}$, a copy of the online network refreshed every C steps and used only to compute targets.
Between refreshes the regression target holds still.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $y = r + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s', a')$ | target from the frozen copy | `algos/dqn.py:156-157` |
| $\bar\theta \leftarrow \theta$ every C = 500 steps | target refresh | `algos/dqn.py:166-167` |
| store `terminated`, not `terminated or truncated` | a time limit still bootstraps | `algos/dqn.py:146-148` |
| predicted max Q vs discounted return actually received | the overestimation measurement | `algos/dqn.py:190-218` (`value_vs_return`) |

### What it got wrong / what came after

Ablation (`experiments/deep_value_sweep.py ablation`), final greedy return, median [min, max]:

| config | final | mean over curve | predicted Q | actual |
| --- | --- | --- | --- | --- |
| replay + target | 500.0 [46.2, 500.0] | 231.3 | 100.09 | 99.94 |
| no replay | 49.8 [10.1, 500.0] | 90.6 | 238.92 | 25.25 |
| no target | 9.2 [9.2, 9.5] | 9.2 | 3.3e10 | 4.91 |
| neither | 9.2 [9.2, 9.5] | 9.2 | 9.4e9 | 4.91 |

On this setup the target network is what prevents divergence; replay is what makes results agree across seeds.
Even full DQN is unstable: 3 of 5 seeds end at 500 and 2 collapse late, to 110.2 and 46.2.
The probe environments (`diagnostics/probe_envs.py`) show one interface bug cleanly: report termination as truncation and probe 1 (reward +1, then end) learns Q ≈ 24.6 instead of 1.
After: Double DQN, PER, Dueling, n-step, combined in Rainbow (below).

## Deep Reinforcement Learning with Double Q-learning (Hado van Hasselt, Arthur Guez, David Silver, 2015)

PDF: [`vanhasselt-2015-double-dqn.pdf`](./vanhasselt-2015-double-dqn.pdf)

### The idea in one paragraph

The max over noisy Q estimates is biased upwards.
Let the online network choose the next action and the target network evaluate it, so the chosen action's error is not the one used to score it.
It is tabular Double Q-learning (`algos/double_q.py`) with the two networks DQN already has.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $a^* = \arg\max_{a'} Q_\theta(s', a')$ | online network selects | `algos/double_dqn.py:66` |
| $y = r + \gamma (1 - d) Q_{\bar\theta}(s', a^*)$ | target network evaluates | `algos/double_dqn.py:67-68` |

### What it got wrong / what came after

On CartPole, Double DQN is the only variant whose prediction sits **below** what it actually gets: 88.05 against 97.51 (medians); plain DQN is 100.09 against 99.94.
HYPOTHESIS (unverified): the correction overshoots into underestimation because CartPole's rewards are deterministic, so there is little upward bias to remove.
It still has the second-best mean over the curve: 287.1 against DQN's 231.3.
The bias it targets is clearer in the tabular case: on Sutton & Barto's maximisation-bias MDP, Q-learning goes down the losing branch in 89.7% of episodes 11–50, Double Q in 26.0% (2,000 runs; `algos/double_q.py`).

## Prioritized Experience Replay (Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, 2015)

PDF: [`schaul-2015-per.pdf`](./schaul-2015-per.pdf)

### The idea in one paragraph

Replay transitions in proportion to their TD error, so the agent studies what it predicts badly.
That changes the distribution the loss is averaged over, so importance-sampling weights correct it, fully by the end of training.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $P(i) = p_i^\alpha / \sum_k p_k^\alpha$, $p_i = \lvert\delta_i\rvert + \epsilon$ | sampling probability; α = 0.6 | `algos/per_dqn.py:81-91` (sum-tree descent) |
| $w_i = (N P(i))^{-\beta} / \max_j w_j$, β: 0.4 → 1 | importance weights | `algos/per_dqn.py:92-94`, β schedule `algos/per_dqn.py:136` |
| new transitions enter at the max priority | each is replayed at least once | `algos/per_dqn.py:75-77` |
| $\sum_i w_i\,\mathrm{Huber}(\delta_i)$, then update $p_i$ | weighted loss and priority update | `algos/per_dqn.py:141-148` |

### What it got wrong / what came after

On CartPole PER has a higher mean over the curve than DQN (300.2 against 231.3) and a lower final median (316.1 [115.9, 496.5] against 500.0 [46.2, 500.0]).
It learns sooner and ends less consistently; five seeds cannot say which of those is the real difference.
After: Rainbow's ablation ranks PER among the two most important components on Atari.

## Dueling Network Architectures for Deep Reinforcement Learning (Ziyu Wang et al., 2016)

PDF: [`wang-2016-dueling.pdf`](./wang-2016-dueling.pdf)

### The idea in one paragraph

Split the Q head into a state value $V(s)$ and per-action advantages $A(s,a)$, recombined as $Q = V + A - \bar A$.
In states where the action barely matters, V is learned once instead of separately for every action.
Subtracting the mean advantage makes V and A identifiable.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $Q(s,a) = V(s) + A(s,a) - \frac{1}{\lvert\mathcal{A}\rvert}\sum_{a'} A(s,a')$ | the dueling head | `algos/dueling_dqn.py:34-45` |

### What it got wrong / what came after

No gain on CartPole: final 125.8 [90.9, 500.0], mean over curve 226.9, against DQN's 231.3.
HYPOTHESIS (unverified): with two actions there is little for the value stream to share, and the paper's gains are on games with many actions.

## Rainbow: Combining Improvements in Deep Reinforcement Learning (Matteo Hessel et al., 2017)

PDF: [`hessel-2017-rainbow.pdf`](./hessel-2017-rainbow.pdf)

### The idea in one paragraph

Combine six DQN extensions — Double, PER, Dueling, n-step returns, distributional value heads (C51) and noisy networks — and ablate each.
The combination beat every single extension on Atari, and the ablation ranked PER and n-step as the most important.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\bar\theta}(s_{t+n}, a')$ | n-step target, n = 3 | `algos/n_step_dqn.py:58-64` (stored), `algos/n_step_dqn.py:85-87` (bootstrap) |
| Double, PER, Dueling | the other three built components | `algos/double_dqn.py`, `per_dqn.py`, `dueling_dqn.py` |

Not built: distributional heads, noisy networks, and a file combining all six.

### What it got wrong / what came after

On CartPole the 3-step target is the clear winner: mean over curve 407.3 against DQN's 231.3, and every seed ends at 98.3 or above.
That matches Rainbow's ranking of n-step among the most important components, on a much smaller problem.
Reward reaches back three states per update instead of one.
