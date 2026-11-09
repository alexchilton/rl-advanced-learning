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
| Double, PER, Dueling | three more components | `algos/double_dqn.py`, `per_dqn.py`, `dueling_dqn.py` |
| C51, Noisy | the last two, sections below | `algos/c51_dqn.py`, `algos/noisy_dqn.py` |

All six are built as separate files; there is no file combining them.

### What it got wrong / what came after

On CartPole the 3-step target is the clear winner: mean over curve 407.3 against DQN's 231.3, and every seed ends at 98.3 or above.
That matches Rainbow's ranking of n-step among the most important components, on a much smaller problem.
Reward reaches back three states per update instead of one.

## A Distributional Perspective on Reinforcement Learning (Marc G. Bellemare, Will Dabney, Rémi Munos, 2017)

PDF: [`bellemare-2017-distributional.pdf`](./bellemare-2017-distributional.pdf)

### The idea in one paragraph

Learn the distribution of the return, not only its mean.
C51 represents it as probabilities on N = 51 fixed atoms between V_MIN and V_MAX.
The Bellman update shifts and shrinks the target distribution, $r + \gamma z_j$, and each shifted atom's probability is split between its two nearest grid points.
The loss is the cross-entropy between that projected target and the prediction; Q is the mean, so acting is unchanged.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $Q(s,a) = \sum_i p_i(s,a)\, z_i$ | the mean, used for acting | `algos/c51_dqn.py:55-56` |
| $\hat T z_j = \mathrm{clip}(r + \gamma (1-d) z_j, V_{MIN}, V_{MAX})$ | shifted target atoms | `algos/c51_dqn.py:63` |
| split by distance to $\lfloor b \rfloor$, $\lceil b \rceil$ | the projection Φ | `algos/c51_dqn.py:59-73` |
| $a^* = \arg\max_a \sum_i p_i(s',a) z_i$ on the target net | next action | `algos/c51_dqn.py:111` |
| $-\sum_i m_i \log p_i(s,a)$ | cross-entropy loss | `algos/c51_dqn.py:114` |

The paper uses V_MAX = −V_MIN = 10 on Atari with clipped rewards. CartPole's discounted returns lie in [0, 100], so the support here is [0, 100].

### What it got wrong / what came after

On a coin flip with an exactly known return distribution (+1 with probability 1/4, −1 otherwise; `diagnostics/probe_envs.py`, `CoinFlip`), C51 recovers both masses within 0.06 (`tests/test_rainbow_trpo.py`).
It passes all five probes with error 0.0, because every probe's return falls exactly on an atom.
On CartPole, 5 seeds: curve average 377.7, second to 3-step (407.3) and ahead of DQN (231.3); final 500 on 4 of 5 seeds; predicted Q 94.92 against 99.00 received.
Why distributional learning helps is still argued about: the mean is the same quantity DQN learns, and what changes is the training signal.

## Noisy Networks for Exploration (Meire Fortunato et al., 2017)

PDF: [`fortunato-2017-noisy-nets.pdf`](./fortunato-2017-noisy-nets.pdf)

### The idea in one paragraph

Replace ε-greedy with noise in the weights, $w = \mu + \sigma \varepsilon$, and learn σ by the same loss as μ.
The greedy action of the current weight sample is the exploratory action.
Factorised Gaussian noise needs one random number per input and per output instead of one per weight.
For DQN the paper resamples the parameters before every action and makes every fully connected layer noisy.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\varepsilon_{ij} = f(e_i) f(e_j)$, $f(x) = \mathrm{sgn}(x)\sqrt{\lvert x\rvert}$ | factorised noise | `algos/noisy_dqn.py:58-61` |
| $w = \mu + \sigma \odot \varepsilon$ | noisy weights in training mode; μ alone in evaluation | `algos/noisy_dqn.py:63-67` |
| resample before every action, or once per episode | `resample="step"` or `"episode"` | `algos/noisy_dqn.py:116-119` |

Here the first layer is plain and the last two are noisy.

### What it got wrong / what came after

On CartPole, 5 seeds: the only variant ending at 500 on every seed (curve average 307.5).
The learned noise does not shrink: mean |σ| rises from 0.044 to between 0.065 and 0.072 on all 5 seeds.

On long chains it fails (`experiments/noisy_chain.py`, 50,000 steps, 5 seeds, action indices scrambled per state):
Noisy DQN never reaches the goal on Chain(20) or Chain(30), whether the noise is resampled every step or held for an episode; ε-greedy DQN reaches it on 2 of 5 seeds on each.
At initialisation the noise changes the greedy action in 6.6% to 28.3% of (state, sample) pairs, against 50% for ε-greedy at ε = 1.
HYPOTHESIS (unverified): that lower randomness at the start is why it never reaches the goal; a larger σ₀ would test it.

Without the scrambling, the first run of that experiment had Noisy DQN reach the Chain(20) goal at steps 41 and 103.
Those were not exploration: before any training, 3 of 5 fresh networks prefer "right" in every chain state, and on the plain chain "right" is the same action index everywhere (`envs/chain.py`, `ScrambledActions`).

