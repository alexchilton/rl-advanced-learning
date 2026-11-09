# Notes — Normalisation and scale

One section per paper.
Numbers are from `experiments/reward_normalisation.py` and `experiments/obs_normalisation.py` on 2026-11-09.
Runnable version: [`notebooks/normalisation_01_popart_obs.ipynb`](../../notebooks/normalisation_01_popart_obs.ipynb).

## Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (Sergey Ioffe, Christian Szegedy, 2015)

PDF: [`ioffe-szegedy-2015-batchnorm.pdf`](./ioffe-szegedy-2015-batchnorm.pdf)

### The idea in one paragraph

Normalise each layer's inputs to zero mean and unit variance over the mini-batch, then let the layer learn a scale and shift.
Training becomes faster and less sensitive to initialisation and learning rate.

### Equations to code

Batch normalisation inside a network is **not built** here.
What is built is its input-side relative used throughout deep RL: a running mean/std normaliser on observations.

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\hat x = \mathrm{clip}((x - \mu)/\sigma, -10, 10)$, μ and σ running (Welford) | observation normaliser | `experiments/obs_normalisation.py`, `RunningNorm` |
| $\hat x$ applied to every sampled batch with the current μ, σ | sample-time normalisation | `algos/td3.py`, `obs_transform` |

### What it got wrong / what came after

TD3 on PointMass, 30,000 steps, 3 seeds; the PD controller in `PointMass.expert_action` scores −13.34.
What differs between the three modes is when the normaliser is applied.

| mode | final return per seed | return at 9,000 steps |
| --- | --- | --- |
| raw | −7.57, −5.41, −5.95 | −15.9, −13.1, −15.1 |
| normalised when stored | −71.85, −17.58, −5.11 | −102.1, −149.9, −96.8 |
| normalised when sampled | −4.25, −4.10, −4.27 | −10.6, −4.6, −5.2 |

Normalising at store time is the worst of the three, and normalising at sample time the best, on every seed.
The difference is the stale scale: a replay entry stored early keeps the statistics of that moment, so the same state appears in the buffer at several scales while the statistics move.
The first run of this experiment had only raw and store-time, and normalisation looked harmful; the hypothesis written then (stale scales in replay) was tested with the sample-time arm and held.
The raw agent learns too, which contradicted the original claim in `envs/pointmass.py`; that docstring now gives these numbers.
In RL, BatchNorm itself is awkward for a related reason: batch statistics differ between replay batches and single-state action selection.

## Learning Values Across Many Orders of Magnitude (Hado van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, David Silver, 2016)

PDF: [`vanhasselt-2016-popart.pdf`](./vanhasselt-2016-popart.pdf)

### The idea in one paragraph

PopArt: the network predicts normalised values; running statistics μ, σ of the targets turn them back into real values.
"ART" adapts the statistics; "POP" rescales the last layer whenever they change so the unnormalised outputs stay exactly the same.
DQN on Atari had clipped rewards to [−1, 1] instead, which keeps the scale fixed but changes the task when reward sizes matter.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $Q = \sigma f(s,a) + \mu$ | unnormalised output | `algos/popart_dqn.py:58` |
| $W \leftarrow W\sigma_{old}/\sigma$, $b \leftarrow (\sigma_{old} b + \mu_{old} - \mu)/\sigma$ | preserve outputs | `algos/popart_dqn.py:68-69` |
| $\mathrm{Huber}(f - (y - \mu)/\sigma)$ | regress normalised targets | `algos/popart_dqn.py:106` |

Notebook section 2 (`torch.manual_seed(0)`, 256 inputs): after one statistics jump to μ = 529, σ = 615, no Q value moved by more than 3.6e-4;
after five jumps to μ = 4,007, σ = 1,348, by no more than 2.7e-3. Both are float32 rounding at that scale. Without POP every Q would move by about the change in μ.

### What it got wrong / what came after

CartPole, every reward × 1,000, DQN, 100,000 steps, 3 seeds, return in original units:

| config | final, median [min, max] | mean over curve | peak \|mean Q\| |
| --- | --- | --- | --- |
| scale 1 (reference) | 500.0 [110.2, 500.0] | 253.7 | 97.8 |
| ×1000, nothing done | 9.2 [9.2, 9.5] | 9.2 | 8.7e7 |
| ×1000, clipped to [−1, 1] | 500.0 [110.2, 500.0] | 253.7 | 97.8 |
| ×1000, divided by 1,000 | 500.0 [110.2, 500.0] | 253.7 | 97.8 |
| ×1000, PopArt | 113.6 [106.3, 453.6] | 248.9 | 1.1e5 |

Untreated, DQN diverges (Q around 1e8) and never learns.
Clipping and dividing are identical to the reference on every seed, by arithmetic rather than by luck: every CartPole reward is +1, so both map the scaled reward back to exactly the original one. This task cannot show clipping's cost, which needs rewards of different sizes.
PopArt recovers the reference learning curve (248.9 against 253.7) without being told the scale, with its Q at the true magnitude, about 1,000 × 100. Its final returns vary more (106.3 to 453.6); the reference also has a seed ending at 110.2.
