# Notes — Policy gradient

One section per paper.
Numbers are from `experiments/policy_gradient_sweep.py` and `experiments/pg_variance.py` on 2026-11-09.
CartPole-v1: 100,000 steps, 5 seeds (3 for the λ sweep), greedy return over 10 episodes every 10,000 steps.
Pendulum-v1: 20,000 steps, 5 seeds, deterministic action over 5 episodes every 2,000 steps; random is about −1,200.
"Mean over curve" averages every evaluation, so it rewards learning sooner.
Runnable versions: [`notebooks/policy_gradient_01_reinforce_to_sac.ipynb`](../../notebooks/policy_gradient_01_reinforce_to_sac.ipynb), and the JAX PPO in [`notebooks/jax_01_lunar_lander.ipynb`](../../notebooks/jax_01_lunar_lander.ipynb) (built by [`notebooks/build_jax_01.py`](../../notebooks/build_jax_01.py)).

## Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Ronald J. Williams, 1992)

PDF: [`williams-1992-reinforce.pdf`](./williams-1992-reinforce.pdf)

### The idea in one paragraph

REINFORCE: play an episode with a stochastic policy, then raise the log-probability of each action in proportion to the return that followed it.
The estimate is unbiased and needs no model and no value function.
Section 4 of the paper already notes that a reinforcement baseline can be subtracted without bias.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $G_t = \sum_{k \ge t} \gamma^{k-t} r_k$ | reward-to-go | `algos/reinforce.py:47-52` |
| $-\frac{1}{T}\sum_t \log\pi(a_t \mid s_t)\, G_t$ | the loss whose gradient is the estimator | `algos/reinforce.py:96` |
| $G_t - V(s_t)$, V fitted to $G_t$ | the learned baseline | `algos/reinforce_baseline.py:63`, `algos/reinforce_baseline.py:70` |

### What it got wrong / what came after

The variance, measured at one frozen policy over 300 one-episode gradients (`experiments/pg_variance.py`):

| weighting | relative variance | cosine of one estimate to the mean |
| --- | --- | --- |
| whole return $G_0$ (the original form) | 1.000 | 0.042 |
| reward-to-go $G_t$ | 0.293 | 0.184 |
| $G_t$ − mean return | 0.115 | 0.304 |
| $G_t - V(s_t)$, V explained variance 0.67 | 0.082 | 0.451 |

A single original-form gradient is almost orthogonal to the true direction (cosine 0.04).
On CartPole the learned baseline lifts the curve average from 389.6 to 406.6, and the final medians from 448.4 to 457.9.
After: actor-critic methods bootstrap the baseline (A2C, below).

## Policy Gradient Methods for Reinforcement Learning with Function Approximation (Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour, 1999)

PDF: [`sutton-1999-policy-gradient.pdf`](./sutton-1999-policy-gradient.pdf)

### The idea in one paragraph

The policy-gradient theorem: $\nabla J = \mathbb{E}_{s \sim d^\pi, a \sim \pi}[\nabla \log\pi(a \mid s)\, Q^\pi(s,a)]$.
The gradient does not involve the derivative of the state distribution, so it can be estimated from samples.
The paper also shows a "compatible" critic can replace $Q^\pi$ without bias, which is the basis of actor-critic methods.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\nabla\log\pi(a \mid s)\, \hat Q(s,a)$ with $\hat Q = G_t$ | the theorem, sampled | `algos/reinforce.py:96` |
| $\nabla\log\pi(a \mid s)\, \hat A(s,a)$ with a learned critic | the actor-critic form | `algos/a2c.py:133` |

### What it got wrong / what came after

The compatible-features condition is rarely met by neural critics, so practical actor-critics are biased; they trade the bias for much lower variance.
A2C and PPO (below) are that trade.

## Trust Region Policy Optimization (John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel, 2015)

PDF: [`schulman-2015-trpo.pdf`](./schulman-2015-trpo.pdf)

Not built in this repo. PPO (below) replaces TRPO's KL constraint with a clipped ratio; both PPO files log the approximate KL that TRPO would constrain (`algos/ppo.py:79`).

## High-Dimensional Continuous Control Using Generalized Advantage Estimation (John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel, 2015)

PDF: [`schulman-2015-gae.pdf`](./schulman-2015-gae.pdf)

### The idea in one paragraph

Estimate the advantage as an exponentially weighted sum of TD errors.
λ = 0 is the one-step TD error: low variance, biased by the critic. λ = 1 is the Monte Carlo return minus a baseline: unbiased, high variance.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ | TD error, cut at episode ends | `algos/a2c.py:110` |
| $\hat A_t = \delta_t + \gamma\lambda(1 - d_t)\hat A_{t+1}$ | the recursion, backwards | `algos/a2c.py:111` |
| $r_T \leftarrow r_T + \gamma V(s_{T+1})$ on truncation | a time limit is not the end of the MDP | `algos/a2c.py:85-87` |
| the same in JAX, as a reverse `lax.scan` | | `notebooks/build_jax_01.py:251-260` |

### What it got wrong / what came after

PPO on CartPole, 3 seeds per λ, curve average: λ = 0 → 247.0, 0.5 → 363.2, 0.9 → 479.3, 0.95 → 481.8, 1.0 → 475.8.
λ = 0 is clearly worst, and one of its seeds ends at 130.7. From 0.9 to 1 the differences are within seed noise.
`tests/test_policy_gradient.py` checks both limits exactly.

## Continuous Control with Deep Reinforcement Learning (Timothy P. Lillicrap et al., 2015)

PDF: [`lillicrap-2015-ddpg.pdf`](./lillicrap-2015-ddpg.pdf)

### The idea in one paragraph

DQN for continuous actions: a deterministic actor $\mu(s)$ stands in for the argmax, trained by following the critic's gradient with respect to the action.
Replay, soft-updated target networks, and Gaussian action noise for exploration.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $y = r + \gamma(1-d)\,Q_{\bar\phi}(s', \mu_{\bar\theta}(s'))$ | critic target | `algos/ddpg.py:125` |
| $\max_\theta Q_\phi(s, \mu_\theta(s))$ | actor: climb the critic | `algos/ddpg.py:131` |
| $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$, τ = 0.005 | soft target update | `algos/ddpg.py:72-75` |

### What it got wrong / what came after

The actor maximises the critic, so it seeks out critic errors.
On Pendulum this does not show: DDPG reaches −200 in 4,000 to 6,000 steps and ends at −104.2 [−110.5, −98.2], against TD3's −97.1.
TD3 (below) is the fix for the failure; Pendulum is too easy to show the failure.

## Asynchronous Methods for Deep Reinforcement Learning (Volodymyr Mnih et al., 2016)

PDF: [`mnih-2016-a3c.pdf`](./mnih-2016-a3c.pdf)

### The idea in one paragraph

A3C: many actor-learners on CPU threads, each computing n-step advantage actor-critic gradients and applying them asynchronously to shared parameters.
The parallel actors decorrelate the data that replay decorrelates in DQN.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $-\log\pi(a_t \mid s_t)\hat A_t + c_v (V - \hat R)^2 - c_e H(\pi)$ | the actor-critic loss with entropy bonus | `algos/a2c.py:131-136` |
| n-step advantage (here GAE with λ = 0.95, 32 steps) | | `algos/a2c.py:103-114` |

`algos/a2c.py` is the synchronous version with one environment: same loss, no threads.

### What it got wrong / what came after

The asynchrony turned out not to be the point: synchronous A2C matches it and is simpler.
On CartPole A2C ends at 500 on all 5 seeds but has a lower curve average than REINFORCE (340.2 against 389.6): it starts slower.
After: PPO reuses each batch for several epochs.

## Proximal Policy Optimization Algorithms (John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, 2017)

PDF: [`schulman-2017-ppo.pdf`](./schulman-2017-ppo.pdf)

### The idea in one paragraph

Reuse each batch for several epochs of minibatch updates, correcting with the probability ratio $r_t = \pi_\theta / \pi_{\text{old}}$.
Clip the ratio to $[1 - \epsilon, 1 + \epsilon]$ and take the pessimistic minimum, so the objective is flat once the policy has moved far enough.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $r_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})$ | the ratio | `algos/ppo.py:70` |
| $-\mathbb{E}[\min(r_t \hat A_t, \mathrm{clip}(r_t, 1 \pm \epsilon)\hat A_t)]$ | the clipped surrogate | `algos/ppo.py:71` |
| epochs × minibatches on one batch | the reuse | `algos/ppo.py:63` |
| $\mathbb{E}[(r-1) - \log r]$, share with $\lvert r - 1\rvert > \epsilon$ | approx KL and clip fraction | `algos/ppo.py:79-80` |
| the same loss in JAX | | `notebooks/build_jax_01.py:262-272` |

### What it got wrong / what came after

On CartPole PPO is the best of the four: 500 on all 5 seeds, curve average 467.9.
The clip only matters if the policy moves far enough to reach it; the JAX notebook measures that 32 gradient steps per 4,096 samples never learn LunarLander (greedy 23.8) and 256 solve it (271.9).
Engstrom et al. 2020 (`papers/practice/`) show that much of PPO's advantage over TRPO comes from code-level details like these.

## Addressing Function Approximation Error in Actor-Critic Methods (Scott Fujimoto, Herke van Hoof, David Meger, 2018)

PDF: [`fujimoto-2018-td3.pdf`](./fujimoto-2018-td3.pdf)

### The idea in one paragraph

DDPG's actor exploits the critic's overestimation. TD3 adds three fixes:
clipped double Q (the target uses the smaller of two critics), a delayed actor (updated every second critic step), and target policy smoothing (clipped noise on the target action).

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\tilde a = \mathrm{clip}(\mu_{\bar\theta}(s') + \mathrm{clip}(\epsilon, -c, c), -1, 1)$ | target smoothing | `algos/td3.py:78-79` |
| $y = r + \gamma(1-d)\min_{i=1,2} Q_{\bar\phi_i}(s', \tilde a)$ | clipped double Q | `algos/td3.py:81-82` |
| actor and targets every `policy_delay` = 2 critic updates | the delay | `algos/td3.py:88-92` |

### What it got wrong / what came after

Ablation on Pendulum, 5 seeds (`experiments/policy_gradient_sweep.py td3_ablation`): every variant ends at about −97, so none of the fixes changes the final result here.
The delay is the only one with a measurable effect, and it is a cost: full TD3 reaches −200 at 8,000 steps on all 5 seeds; without the delay, 4,000 to 6,000.
Removing all three is as fast as removing only the delay (curve average −238.6 against −241.1).
The fixes do show in the critic: final mean Q on the replay batch is −63.3 for full TD3, −59.2 without smoothing, −52.2 without clipped double Q, −49.7 without the delay, and −23.0 with none of them (every seed between −28.2 and −22.8).
Each fix pulls the values down, as designed; whether the higher values are overestimates needs the true values of the buffer states, which these runs do not compute.
HYPOTHESIS (unverified): on Pendulum the actor does not turn the critic's optimism into a worse return within 20,000 steps. The paper's gains are on MuJoCo locomotion, which is not installed here.

## Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine, 2018)

PDF: [`haarnoja-2018-sac.pdf`](./haarnoja-2018-sac.pdf)

### The idea in one paragraph

Maximise reward plus α times the policy's entropy.
The actor is a tanh-squashed Gaussian trained through the reparameterisation trick; the critics learn soft Q values with TD3's clipped double Q.
The follow-up paper learns α to hold the entropy near a target.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\log\pi(a) = \log\mathcal{N}(u) - \sum\log(1 - \tanh^2 u)$ | tanh correction, in stable form | `algos/sac.py:58` |
| $y = r + \gamma(1-d)(\min_i Q_{\bar\phi_i}(s', a') - \alpha\log\pi(a' \mid s'))$ | soft target | `algos/sac.py:110` |
| $\mathbb{E}[\alpha\log\pi(a \mid s) - \min_i Q_i(s, a)]$, $a$ reparameterised | actor loss | `algos/sac.py:117` |
| $-\log\alpha\,(\log\pi + \bar{\mathcal{H}})$, target $\bar{\mathcal{H}} = -1$ | learned temperature | `algos/sac.py:121` |

`tests/test_policy_gradient.py` checks the tanh correction against the direct change-of-variables formula.

### What it got wrong / what came after

On Pendulum SAC ends at −97.1 [−98.9, −96.2] and reaches −200 in 4,000 to 6,000 steps.
Temperature sweep, final median: learned α −97.1, α = 0.01 −98.1, α = 0.1 −98.4, α = 1 −121.3.
Only α = 1 hurts: it pays the policy to stay random (entropy +0.33 at the end).
The learned α settles at 0.022 and holds the entropy at −0.97, against the target of −1, so the automatic temperature does what it is for.
