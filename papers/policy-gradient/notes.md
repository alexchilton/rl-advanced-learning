# Notes — Policy gradient

One section per paper.
Only PPO and GAE have code in this repo, inside the JAX notebook; the others say what is missing.
Runnable version: [`notebooks/jax_01_lunar_lander.ipynb`](../../notebooks/jax_01_lunar_lander.ipynb), built by [`notebooks/build_jax_01.py`](../../notebooks/build_jax_01.py). Line numbers below are in the build script.

## Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Ronald J. Williams, 1992)

PDF: [`williams-1992-reinforce.pdf`](./williams-1992-reinforce.pdf)

Not built in this repo yet. The PPO loss below uses the same score-function gradient, $\nabla \log \pi(a \mid s)\, A$.

## Policy Gradient Methods for Reinforcement Learning with Function Approximation (Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour, 1999)

PDF: [`sutton-1999-policy-gradient.pdf`](./sutton-1999-policy-gradient.pdf)

Not built in this repo yet.

## Trust Region Policy Optimization (John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel, 2015)

PDF: [`schulman-2015-trpo.pdf`](./schulman-2015-trpo.pdf)

Not built in this repo yet. PPO (below) replaces TRPO's KL constraint with a clipped ratio; the JAX notebook logs the approximate KL TRPO would constrain.

## High-Dimensional Continuous Control Using Generalized Advantage Estimation (John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel, 2015)

PDF: [`schulman-2015-gae.pdf`](./schulman-2015-gae.pdf)

### The idea in one paragraph

Estimate the advantage as an exponentially weighted sum of TD errors.
λ = 0 is the one-step TD error: low variance, biased by the critic. λ = 1 is the Monte Carlo return minus a baseline: unbiased, high variance.
λ between them trades one for the other.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ | TD error, cut at terminations | `notebooks/build_jax_01.py:256` |
| $\hat A_t = \delta_t + \gamma\lambda(1 - d_t)\hat A_{t+1}$ | the GAE recursion, as a reverse `lax.scan` | `notebooks/build_jax_01.py:251-260` |
| $\hat R_t = \hat A_t + V(s_t)$ | value target | `notebooks/build_jax_01.py:260` |

### What it got wrong / what came after

A time limit is not a termination.
The notebook folds the bootstrap for truncated episodes into the reward before GAE (comment at line 252); treating truncation as termination teaches the lander that hovering is fatal.
Settings used on LunarLander: γ = 0.999, λ = 0.98.

## Continuous Control with Deep Reinforcement Learning (Timothy P. Lillicrap et al., 2015)

PDF: [`lillicrap-2015-ddpg.pdf`](./lillicrap-2015-ddpg.pdf)

Not built in this repo yet.

## Asynchronous Methods for Deep Reinforcement Learning (Volodymyr Mnih et al., 2016)

PDF: [`mnih-2016-a3c.pdf`](./mnih-2016-a3c.pdf)

Not built in this repo yet.

## Proximal Policy Optimization Algorithms (John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, 2017)

PDF: [`schulman-2017-ppo.pdf`](./schulman-2017-ppo.pdf)

### The idea in one paragraph

Take several gradient steps on the same batch of on-policy data without letting the policy move too far.
Instead of TRPO's KL constraint, clip the probability ratio $r_t = \pi_\theta(a \mid s) / \pi_{\text{old}}(a \mid s)$ to $[1 - \epsilon, 1 + \epsilon]$ and take the pessimistic minimum of the clipped and unclipped objectives.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $r_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})$ | the probability ratio | `notebooks/build_jax_01.py:266` |
| $-\mathbb{E}[\min(r_t \hat A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)]$ | the clipped surrogate | `notebooks/build_jax_01.py:268` |
| $\hat A \leftarrow (\hat A - \mu) / \sigma$ | per-minibatch advantage normalisation | `notebooks/build_jax_01.py:267` |
| $L = L^{\text{clip}} + c_v L^{V} - c_e H$ | value loss and entropy bonus | `notebooks/build_jax_01.py:269-271` |
| $\mathbb{E}[(r - 1) - \log r]$, $\mathbb{E}[\lvert r - 1 \rvert > \epsilon]$ | approx KL and clip fraction, the health panel | `notebooks/build_jax_01.py:272` |

### What it got wrong / what came after

The clip only matters if the policy moves far enough to reach it.
Measured in the JAX notebook: 32 gradient steps per 4,096 samples never learns (greedy 23.8, approx KL about 1e-3); 256 steps solves LunarLander (271.9) in 92 s.
At an equal 1M-step budget over 5 seeds, SB3 PPO averaged 249.0 and the from-scratch version 242.6 to 254.1; the spreads (6.5 to 20.2) are as large as the gaps, so the scores do not rank.
Engstrom et al. 2020 and Andrychowicz et al. 2020 (in `papers/practice/`) show that much of PPO's advantage over TRPO comes from implementation details like these.
A residual-PPO run on the Smart Buildings simulator that does not learn is in `notebooks/case_smart_buildings_residual_ppo.ipynb`.

## Addressing Function Approximation Error in Actor-Critic Methods (Scott Fujimoto, Herke van Hoof, David Meger, 2018)

PDF: [`fujimoto-2018-td3.pdf`](./fujimoto-2018-td3.pdf)

Not built in this repo yet.

## Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine, 2018)

PDF: [`haarnoja-2018-sac.pdf`](./haarnoja-2018-sac.pdf)

Not built in this repo yet. The soft value iteration in `algos/max_ent_irl.py:69-85` is the tabular form of the same maximum-entropy backup.
