# Notes — Preference-based and LLM RL

One section per paper.
**Everything here runs on a toy, not a language model.** A tiny GRU writes token sequences, and the true reward is a formula (`envs/preference_toy.py`): +1 per requested token, −0.3 per other token, at most 12 tokens.
Methods see only preference labels, as in real RLHF; the formula only scores the results, which is what makes over-optimisation measurable.
No real LLM was trained; the cached Hugging Face models on this machine have no weights, and downloading one was not approved.
Numbers are from `experiments/preference_sweep.py` on 2026-11-09. The reference (SFT) policy scores 1.18.
Runnable version: [`notebooks/preference_01_rlhf_dpo_grpo.ipynb`](../../notebooks/preference_01_rlhf_dpo_grpo.ipynb).

## Deep Reinforcement Learning from Human Preferences (Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei, 2017)

PDF: [`christiano-2017-human-preferences.pdf`](./christiano-2017-human-preferences.pdf)

### The idea in one paragraph

People compare two short trajectory clips; a reward model is fitted to those comparisons with the Bradley-Terry likelihood; an RL agent optimises the learned reward; new comparisons are requested as the policy changes.
About an hour of human feedback taught MuJoCo and Atari behaviours, including a backflip no one could write a reward for.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ | Bradley-Terry | `envs/preference_toy.py:72` (labels), `algos/reward_model.py:61` |
| $-\log\sigma(r(y_w) - r(y_l))$ | reward-model loss | `algos/reward_model.py:62` |

### What it got wrong / what came after

Reward-model accuracy against the true reward: 0.690 (100 pairs), 0.753 (300), 0.870 (1,000), 0.893 (3,000), 0.899 (10,000).
It saturates because the labels are noisy: 10.7% of labels prefer the response with the lower true reward.
This toy uses a fixed set of comparisons; the paper's online querying, which keeps the reward model accurate where the policy goes, is not built.

## Learning to Summarize from Human Feedback (Nisan Stiennon et al., 2020)

PDF: [`stiennon-2020-summarize-hf.pdf`](./stiennon-2020-summarize-hf.pdf)

### The idea in one paragraph

The Christiano recipe applied to a language model: supervised fine-tuning, a reward model from human comparisons of summaries, then PPO against the reward model with a KL penalty to the SFT policy.
The KL term keeps the policy where the reward model is trustworthy.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $R = r_{RM}(x, y) - \beta \log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)}$ | reward with the KL penalty folded in | `algos/rlhf.py:63`, `algos/rlhf.py:68` |
| SFT by maximum likelihood | the starting point | `algos/sequence_policy.py:78` |

### What it got wrong / what came after

The paper itself shows that optimising the reward model too hard makes summaries worse in human ratings.
Here that is measured exactly (Gao et al. 2022, below).

## Training Language Models to Follow Instructions with Human Feedback (Long Ouyang et al., 2022)

PDF: [`ouyang-2022-instructgpt.pdf`](./ouyang-2022-instructgpt.pdf)

### The idea in one paragraph

InstructGPT: SFT on demonstrations, a reward model from rankings, PPO with a KL penalty (plus a pretraining-loss term).
A 1.3B RLHF model was preferred by labellers over the 175B GPT-3.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $\max_\pi \mathbb{E}[r_{RM}] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{ref})$ | the objective | `algos/rlhf.py:63-72` |

PPO's clip and value network are not used: each batch takes one gradient step, so the ratio is 1 (`algos/rlhf.py` docstring).

### What it got wrong / what came after

On the toy, RLHF with the human reward model (3,000 pairs, β 0.05, 300 steps) lifts the true reward from 1.18 to 8.1–8.8 (3 seeds × 3 estimators), against an optimum of 12.
The true reward peaks around step 125–150 at about 9 and then declines while the reward-model score keeps rising.
After: DPO removes the reward model; RLOO and GRPO remove the value network.

## Constitutional AI: Harmlessness from AI Feedback (Yuntao Bai et al., 2022)

PDF: [`bai-2022-constitutional-ai.pdf`](./bai-2022-constitutional-ai.pdf)

### The idea in one paragraph

Replace human harmlessness labels with labels from a model guided by a written list of principles (RLAIF), plus a supervised phase of self-critique and revision.
Feedback becomes cheap and the principles become explicit and auditable.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| labels from a rule: "mention the requested token; repetition is bad" | an AI/rubric labeler | `envs/preference_toy.py:64-69` |

The paper's labeler is a language model reading principles; this one is a fixed rule. Only the consequence is reproduced, not the method.

### What it got wrong / what came after

The rule sounds reasonable and penalises exactly the repetition the true reward wants.
It agrees with the true reward on 40.9% of pairs; its reward model orders fresh pairs correctly 44.9% of the time.
RLHF on it reaches a reward-model score of 15.4 and a true reward of **0.48, below the SFT's 1.18**, with 2.8-token responses.
The general point: AI feedback is only as aligned with the goal as its principles, and optimisation finds where they differ.

## Scaling Laws for Reward Model Overoptimization (Leo Gao, John Schulman, Jacob Hilton, 2022)

PDF: [`gao-2022-scaling-laws-overoptimization.pdf`](./gao-2022-scaling-laws-overoptimization.pdf)

### The idea in one paragraph

Use a large "gold" reward model as ground truth and train smaller proxy reward models on its labels.
As a policy is optimised against the proxy, the gold score first rises and then falls, along a curve whose shape depends on the proxy's size and the data, measured against KL from the initial policy.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| gold reward = the exact formula $r^*$ | ground truth, without a gold model | `envs/preference_toy.py` (`true_reward`) |
| proxy = reward model from 1,000 pairs (accuracy 0.870) | the weaker proxy | `experiments/preference_sweep.py` (`overopt`) |

### What it got wrong / what came after

RLOO, 600 steps, 2 seeds per β:

| β | peak true reward (step) | final true | final proxy | final KL |
| --- | --- | --- | --- | --- |
| 0 | 9.55 (200) | 7.98 | 16.68 | 17.9 |
| 0.01 | 9.53 (200) | 7.86 | 16.72 | 18.0 |
| 0.03 | 9.04 (200) | 7.60 | 16.66 | 14.6 |
| 0.1 | 8.27 (150) | 6.99 | 16.42 | 9.4 |
| 0.3 | 7.34 (100) | 6.36 | 15.85 | 5.9 |

At every β the true reward peaks and then falls by 1 to 1.6 while the proxy stays high.
The KL penalty lowered both the peak and the end point; it did not prevent the decline. Stopping at the peak would have, and that needs the true reward.
Gao et al. report the same (Section 3, Figure 9): the KL penalty "does not affect the KL_RL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping". Here too, every β traces the same rising curve of true reward against √KL.

## Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn, 2023)

PDF: [`rafailov-2023-dpo.pdf`](./rafailov-2023-dpo.pdf)

### The idea in one paragraph

The KL-regularised objective's optimum is $\pi^* \propto \pi_{ref}\exp(r/\beta)$, so $r = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z$.
Substitute into Bradley-Terry and $Z$ cancels: preference learning becomes a classification loss on the policy itself, with no reward model and no sampling.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $-\log\sigma\big(\beta[\log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \log\frac{\pi(y_l)}{\pi_{ref}(y_l)}]\big)$ | the DPO loss | `algos/dpo.py:53-54` |
| $\log\pi(y \mid x) = \sum_t \log\pi(y_t \mid x, y_{<t})$ | sequence log-probability | `algos/sequence_policy.py:42-56` |

### What it got wrong / what came after

On the same 3,000 human pairs, 4 epochs: true reward 6.81 to 7.52 across β ∈ {0.03, 0.1, 0.3} (2 seeds each), against 8.56 for RLHF with RLOO.
DPO's curves dip at most 0.3 below their best; RLHF's fall by 1 to 1.6 in the over-optimisation runs.
HYPOTHESIS (unverified): DPO only moves probability among responses in the pairs, all from the reference policy, while RLHF samples and scores new ones.

## A Long Way to Go: Investigating Length Correlations in RLHF (Prasann Singhal, Tanya Goyal, Jiacheng Xu, Greg Durrett, 2023)

PDF: [`singhal-2023-length-bias.pdf`](./singhal-2023-length-bias.pdf)

### The idea in one paragraph

Much of what RLHF improves, by reward-model score, is response length: preference data and reward models favour longer answers, and policies learn to write more.
Controlling for length removes a large share of the measured gains.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| labeler score $= r^* + 0.5 \times \text{length}$ | a length-biased labeler | `envs/preference_toy.py:62-63` |

### What it got wrong / what came after

The length-biased labeler agrees with the true reward on 79.3% of pairs. RLHF on its reward model ends at a true reward of 6.62, against 8.20 with the unbiased one.
Length hardly differs (10.4 against 10.2 tokens) because on this task the true optimum is also the longest response, so the bias costs reward per token rather than showing up as length.
This toy cannot reproduce the paper's main observation, rising length, for that reason.

## Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs (Arash Ahmadian et al., 2024)

PDF: [`ahmadian-2024-rloo.pdf`](./ahmadian-2024-rloo.pdf)

### The idea in one paragraph

In RLHF the policy starts from a strong pretrained model and the reward comes once per completion, so PPO's machinery (clipping, a value network, per-token credit) buys little.
REINFORCE with a leave-one-out baseline over k samples per prompt (RLOO) matches or beats PPO, with less memory.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $A_i = R_i - \frac{1}{k-1}\sum_{j \ne i} R_j$ | leave-one-out baseline, k = 4 | `algos/rlhf.py:70` |

### What it got wrong / what came after

Final true reward, 3 seeds each: batch-mean baseline 8.07, 8.11, 8.69; RLOO 8.20, 8.80, 8.67; GRPO 8.51, 8.48, 8.59. The three are within seed noise on this task.

## DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Zhihong Shao et al., 2024)

PDF: [`shao-2024-deepseekmath-grpo.pdf`](./shao-2024-deepseekmath-grpo.pdf)

### The idea in one paragraph

GRPO: sample a group of answers per question, normalise each reward by the group's mean and standard deviation, and use that as the advantage, with no value network.
The KL to the reference is a separate loss term, estimated with $\frac{\pi_{ref}}{\pi} - \log\frac{\pi_{ref}}{\pi} - 1$.
Used with verifiable rewards (answer checking) on mathematics.

### Equations to code

| Equation | What it is | file:line in this repo |
| --- | --- | --- |
| $A_i = (R_i - \bar R_{\text{group}}) / \sigma_{\text{group}}$ | group-relative advantage | `algos/rlhf.py:66` |
| $\beta\,(\frac{\pi_{ref}}{\pi} - \log\frac{\pi_{ref}}{\pi} - 1)$ | the k3 KL estimator, as a loss term | `algos/rlhf.py:77-78` |

### What it got wrong / what came after

On the toy GRPO matches the other estimators on true reward (8.48 to 8.59) and moves further from the reference: final KL 16.4 to 17.7, against 11.3 to 13.5 for RLOO and the mean baseline.
Per-group standardisation rescales every prompt's advantages to unit size whatever the reward spread, which later work has questioned; not measured here.
The verifiable-reward setting of the paper (a real model checking maths answers) is not built.
