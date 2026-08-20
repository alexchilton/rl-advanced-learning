# Reference implementations

**What is verified here and what is not.** Every URL below was checked to resolve.
The language, era and maintenance notes come from reading the repository. The
"worth reading / worth running" lines are a JUDGEMENT from inspecting the code --
none of these repos has been cloned, installed or executed as part of this repo.
Treat them as a reading order, not as a claim that they build.

Where this repo has its own demo of a paper's idea, it is linked from
`papers/README.md`. Those are run and tested; these are not.

# Code links

One entry per paper that has a public reference implementation.
Every URL below was checked live (`curl -sI` / GitHub API) on 2026-08-20, not recalled from memory.
Papers with no verified public code are omitted here; they are marked "none public" in `README.md`.

No repository is vendored into this repo.
These are links only, per the plan.

## DQN family

**`google-deepmind/dqn`** — https://github.com/google-deepmind/dqn
Lua/Torch7, released 2015, archived (read-only) since 2017.
Official code for Mnih 2015 (Nature DQN).
Looks worth reading for the exact preprocessing and replay-buffer details the paper elides, but do not try to run it: Torch7 is dead and the repo has had no updates in 9 years.

**`google-deepmind/dqn_zoo`** — https://github.com/google-deepmind/dqn_zoo
Python/JAX, released 2020, still maintained.
Covers Double DQN (van Hasselt 2015), Dueling DQN (Wang 2016), Prioritized Experience Replay (Schaul 2015), and Rainbow (Hessel 2017) — DeepMind never released separate repos for these, this is the closest thing to official code and it actually runs today.
Worth reading; it is the best single reference for how these four papers compose.

**`Kaixhin/Rainbow`** — https://github.com/Kaixhin/Rainbow
Python/PyTorch, released 2017, last updated 2022.
Unofficial but widely cited PyTorch reimplementation of Hessel 2017.
Worth reading as a second, more readable opinion alongside `dqn_zoo` — smaller, single-file-per-component, easier to diff against your own implementation.

## Policy gradient family

**`joschu/modular_rl`** — https://github.com/joschu/modular_rl
Python/Keras+Theano, released 2016, unmaintained since 2018.
John Schulman's own repo, implementing both TRPO (Schulman 2015) and GAE (Schulman 2015) — the two papers share this codebase.
Looks worth reading for the GAE computation and the conjugate-gradient TRPO step, but Theano is dead; treat it as reference pseudocode, not something to run.

**`openai/baselines`** — https://github.com/openai/baselines
Python/TensorFlow 1.x, released 2017, in maintenance mode (no active development, still receives no updates but the repo is not archived).
Reference implementation used for PPO (Schulman 2017), DDPG (Lillicrap 2015), and HER (Andrychowicz 2017) — all three ship from the same repo.
Looks worth reading for PPO's practical tricks (reward/observation normalisation, value clipping) that are not in the paper; TF1 makes it painful to actually run in 2026.

**`miyosuda/async_deep_reinforce`** — https://github.com/miyosuda/async_deep_reinforce
Python/TensorFlow, released 2016, unmaintained since 2018.
Unofficial reimplementation of A3C (Mnih 2016) — DeepMind never released official code.
Worth a skim for the multi-threaded actor-learner structure; not worth running, everyone reproduces A3C via a synchronous approximation (A2C) now anyway.

**`sfujim/TD3`** — https://github.com/sfujim/TD3
Python/PyTorch, released 2018, still lightly maintained.
Scott Fujimoto's own repo for TD3 (Fujimoto 2018).
Worth reading and worth running — clean, short, still current PyTorch idioms.

**`haarnoja/sac`** — https://github.com/haarnoja/sac
Python/TensorFlow 1.x, released 2017, explicitly deprecated by the author.
Original SAC (Haarnoja 2018) code; the README says to use the successor, `rail-berkeley/softlearning`, instead.
Looks worth reading the original once for the temperature auto-tuning derivation; do not try to run either repo, use a modern PyTorch SAC (e.g. from `sfujim` or CleanRL) instead.

## Imitation and offline RL family

**`openai/imitation`** — https://github.com/openai/imitation
Python/TensorFlow + rllab, released 2016, archived.
Jonathan Ho's own repo for GAIL (Ho & Ermon 2016).
Looks worth reading for the discriminator/generator loop; the rllab dependency makes it effectively unrunnable today.

**`sfujim/BCQ`** — https://github.com/sfujim/BCQ
Python/PyTorch, released 2018, lightly maintained (both the continuous ICML 2019 and discrete NeurIPS-workshop 2019 variants are in the same repo).
Official code for Fujimoto 2019 (BCQ).
Looks worth reading and running (not executed here) — short, current PyTorch.

**`aviralkumar2907/CQL`** — https://github.com/aviralkumar2907/CQL
Mixed: `atari/` is TensorFlow 1.x built on Google's `batch_rl`/Dopamine, `d4rl/` is PyTorch built on `rlkit`.
Official code for Kumar 2020 (CQL).
Looks worth reading the `d4rl/` (PyTorch) half; the Atari half drags in an old Dopamine dependency tree that is not worth fighting with.

**`ikostrikov/implicit_q_learning`** — https://github.com/ikostrikov/implicit_q_learning
Python/JAX, released 2021, lightly maintained.
Official code for Kostrikov 2021 (IQL).
Looks worth reading and running (not executed here) — small, JAX makes the expectile-regression loss easy to see directly.

## Exploration family

**`openai/random-network-distillation`** — https://github.com/openai/random-network-distillation
Python/TensorFlow 1.x, released 2018, archived.
Official code for Burda 2018 (RND).
Looks worth reading for the observation-normalisation details around the random/predictor networks — they matter more than the paper's headline idea and are easy to get wrong; not worth running given the TF1/MPI dependency stack.

## Practice / debugging-half family

**`MadryLab/implementation-matters`** — https://github.com/MadryLab/implementation-matters
Python/PyTorch, released 2020, lightly maintained.
Official code for Engstrom 2020 (Implementation Matters in Deep Policy Gradients).
Looks worth reading and running (not executed here) — it is built specifically to isolate each PPO implementation trick one at a time, which is exactly the debugging use case this repo cares about.

## Preference-based and LLM RL family

**`nottombrown/rl-teacher`** — https://github.com/nottombrown/rl-teacher
Python/TensorFlow, released 2017, unmaintained since 2023 (no commits, not formally archived).
Reference implementation associated with Christiano 2017 (deep RL from human preferences).
Looks worth reading for the reward-predictor-plus-PPO loop structure; treat as historical, do not try to run it.

**`openai/summarize-from-feedback`** — https://github.com/openai/summarize-from-feedback
Python/TensorFlow, released 2020, archived.
Official code for Stiennon 2020.
Looks worth reading for how the reward model and PPO fine-tuning loop are wired together at (then-)scale; archived and TF-based, not meant to run today.

**`anthropics/ConstitutionalHarmlessnessPaper`** — https://github.com/anthropics/ConstitutionalHarmlessnessPaper
Not a training implementation — contains prompts, evals, and sample transcripts only, released 2022, archived.
Companion release for Bai 2022 (Constitutional AI).
Looks worth reading the constitution prompts themselves; there is no RL training code here, Anthropic did not release one.

**`eric-mitchell/direct-preference-optimization`** — https://github.com/eric-mitchell/direct-preference-optimization
Python/PyTorch + HuggingFace Transformers, released 2023, lightly maintained.
Official code for Rafailov 2023 (DPO), by the paper's own author.
Looks worth reading and running (not executed here) — the closed-form loss is a few lines, and this repo is the clearest place to see how it replaces the PPO loop entirely.

**`huggingface/trl`** — https://github.com/huggingface/trl
Python/PyTorch + HuggingFace, actively maintained (commits as recent as this year).
Not the paper authors' own repo, but contains an `RLOOTrainer` implementing Ahmadian 2024 (RLOO), and is the practical way most people run RLOO, DPO, PPO-RLHF and GRPO-style training today.
Worth reading as the "how this actually gets run in production" counterpart to the smaller from-scratch repos above; large and fast-moving, expect the API to have drifted since any given paper.

**`deepseek-ai/DeepSeek-Math`** — https://github.com/deepseek-ai/DeepSeek-Math
Python, released 2024, actively maintained.
Official code accompanying Shao 2024, which introduces GRPO.
Looks worth reading for the group-relative advantage computation directly from the people who proposed it.

**`PrasannS/rlhf-length-biases`** — https://github.com/PrasannS/rlhf-length-biases
Python/Jupyter notebooks, released 2023, lightly maintained.
Prasann Singhal's own repo for Singhal 2023 (length bias in RLHF).
Looks worth reading for the length-correlation measurement code specifically — it is the part of this repo most directly reusable for the length-bias demo in Tier 6.

## Normalisation family

**`google-deepmind/scalable_agent`** — https://github.com/google-deepmind/scalable_agent
Python/TensorFlow 1.x, released 2018, lightly maintained.
This is the IMPALA codebase, not a standalone PopArt repo — DeepMind never released one — but it contains a working PopArt implementation (van Hasselt 2016) inside a real large-scale agent.
Worth reading specifically for the PopArt layer, once you know where to look in a much bigger codebase; not worth standing up the whole IMPALA stack just for this.

## Not included: no verified public code found

These have code=None in `fetch.py`.
Some plausible-looking URLs exist (thesis-era code, course reimplementations, unofficial GitHub repos with no connection to the authors), but none resolved to something I could call a reference implementation with confidence, so they are left out rather than guessed at:

- Sutton 1988 (TD), Watkins 1989, Rummery & Niranjan 1994, Sutton 1990 (Dyna), van Hasselt 2010 (tabular Double Q), Ng 1999 (reward shaping) — all pre-date the convention of releasing code with a paper.
- Williams 1992 (REINFORCE), Sutton 1999 (policy gradient theorem) — same.
- Pomerleau 1991 (ALVINN), Ziebart 2008 (MaxEnt IRL), Ross 2011 (DAgger) — no repo traceable to the authors; DAgger in particular is simple enough that every imitation-learning library implements it directly rather than pointing back to one canonical repo.
- Levine 2020 (offline RL survey) — a survey, not an algorithm.
- Bellemare 2016 (count-based / CTS density model), Bengio 2009 (curriculum learning), Florensa 2017 (reverse curriculum) — searched GitHub and the papers' own pages; nothing surfaced that is clearly the authors' own code.
- Henderson 2017 (Deep RL That Matters), Andrychowicz 2020 (What Matters in On-Policy RL) — empirical studies over other people's algorithms, not their own codebase.
- Ouyang 2022 (InstructGPT), Gao 2022 (reward-model overoptimization scaling laws) — OpenAI did not release training code for either.
- Ioffe & Szegedy 2015 (batch norm) — original implementation was internal (Caffe/Google); the method is now built into every deep learning framework, so there is nothing separate worth linking.
