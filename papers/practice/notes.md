# Notes — Practice (the debugging half)

One section per paper.
No algorithm in this tier has its own code in this repo yet; each section says what exists, if anything.

## Deep Reinforcement Learning that Matters (Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger, 2017)

PDF: [`henderson-2017-deep-rl-matters.pdf`](./henderson-2017-deep-rl-matters.pdf)

This is a practice paper, so there are no equations to map. Where its rules show up in this repo:

| Rule from the paper | Where it is applied |
| --- | --- |
| report several seeds, never one curve | `experiments/bc_vs_dagger.py:27` (5 seeds), `experiments/sparse_vs_dense.py:35`, `experiments/offline_dataset_quality.py:35`, `algos/offline_to_online.py:41` (3 seeds each) |
| report the spread, not only the mean | the median `[min, max]` columns printed by `experiments/bc_vs_dagger.py` and `algos/gail.py` |
| evaluation noise hides real differences | removed entirely for finite environments: every score is an exact `policy_return` (`envs/solvers.py:164`) |

What the repo adds: exact evaluation removes measurement noise but not policy volatility. In `algos/offline_to_online.py` naive fine-tuning starts at −6.54; its worst point is −38.30 on one seed and −9.30 and −8.93 on the other two.

## Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO (Logan Engstrom et al., 2020)

PDF: [`engstrom-2020-implementation-matters.pdf`](./engstrom-2020-implementation-matters.pdf)

No standalone code. The JAX notebook is a small instance of the paper's point: the same PPO loss with 32 or 256 gradient steps per batch gives greedy 23.8 or 271.9 on LunarLander (`notebooks/jax_01_lunar_lander.ipynb`).

## What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study (Marcin Andrychowicz et al., 2020)

PDF: [`andrychowicz-2020-what-matters.pdf`](./andrychowicz-2020-what-matters.pdf)

No standalone code. The JAX notebook compares PPO settings (run A against run B, SB3's settings against ours) at an equal 1M-step budget over 5 seeds; see its summary table.
