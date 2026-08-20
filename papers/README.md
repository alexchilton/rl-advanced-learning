# Papers

Reading list for `rl_advanced_learning`, grouped by tier as in `docs/rl-curriculum-plan.md`.
PDFs are fetched by `fetch.py` into the matching subdirectory and are gitignored (`papers/**/*.pdf`).
This table is the tracked index.

Each tier subdirectory also has a `notes.md` stub for mapping equations to line numbers in this repo, per the plan.

Run `python papers/fetch.py` to download everything.
See `code_links.md` for notes on the reference implementations linked below.

## Tabular (`tabular/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Learning to Predict by the Methods of Temporal Differences | 1988 | Richard S. Sutton | [`sutton-1988-td.pdf`](tabular/sutton-1988-td.pdf) | [abs](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) | none public |
| Learning from Delayed Rewards | 1989 | Christopher J. C. H. Watkins | [`watkins-1989-qlearning.pdf`](tabular/watkins-1989-qlearning.pdf) | [abs](http://www.cs.rhul.ac.uk/~chrisw/thesis.html) | none public |
| Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming | 1990 | Richard S. Sutton | [`sutton-1990-dyna.pdf`](tabular/sutton-1990-dyna.pdf) | [abs](http://incompleteideas.net/papers/sutton-90.pdf) | none public |
| On-Line Q-Learning Using Connectionist Systems | 1994 | G. A. Rummery, M. Niranjan | [`rummery-niranjan-1994-sarsa.pdf`](tabular/rummery-niranjan-1994-sarsa.pdf) | [abs](http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf) | none public |
| Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping | 1999 | Andrew Y. Ng, Daishi Harada, Stuart Russell | [`ng-1999-shaping.pdf`](tabular/ng-1999-shaping.pdf) | [abs](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) | none public |
| Double Q-learning | 2010 | Hado van Hasselt | [`vanhasselt-2010-doubleq.pdf`](tabular/vanhasselt-2010-doubleq.pdf) | [abs](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) | none public |

## Deep value (`deep-value/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Playing Atari with Deep Reinforcement Learning | 2013 | Volodymyr Mnih et al. | [`mnih-2013-atari-dqn.pdf`](deep-value/mnih-2013-atari-dqn.pdf) | [abs](https://arxiv.org/abs/1312.5602) | none public |
| Human-level control through deep reinforcement learning | 2015 | Volodymyr Mnih et al. | [`mnih-2015-dqn-nature.pdf`](deep-value/mnih-2015-dqn-nature.pdf) | [abs](https://www.nature.com/articles/nature14236) | [link](https://github.com/google-deepmind/dqn) |
| Deep Reinforcement Learning with Double Q-learning | 2015 | Hado van Hasselt, Arthur Guez, David Silver | [`vanhasselt-2015-double-dqn.pdf`](deep-value/vanhasselt-2015-double-dqn.pdf) | [abs](https://arxiv.org/abs/1509.06461) | [link](https://github.com/google-deepmind/dqn_zoo) |
| Prioritized Experience Replay | 2015 | Tom Schaul, John Quan, Ioannis Antonoglou, David Silver | [`schaul-2015-per.pdf`](deep-value/schaul-2015-per.pdf) | [abs](https://arxiv.org/abs/1511.05952) | [link](https://github.com/google-deepmind/dqn_zoo) |
| Dueling Network Architectures for Deep Reinforcement Learning | 2016 | Ziyu Wang et al. | [`wang-2016-dueling.pdf`](deep-value/wang-2016-dueling.pdf) | [abs](https://arxiv.org/abs/1511.06581) | [link](https://github.com/google-deepmind/dqn_zoo) |
| Rainbow: Combining Improvements in Deep Reinforcement Learning | 2017 | Matteo Hessel et al. | [`hessel-2017-rainbow.pdf`](deep-value/hessel-2017-rainbow.pdf) | [abs](https://arxiv.org/abs/1710.02298) | [link](https://github.com/google-deepmind/dqn_zoo) |

## Policy gradient (`policy-gradient/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | 1992 | Ronald J. Williams | [`williams-1992-reinforce.pdf`](policy-gradient/williams-1992-reinforce.pdf) | [abs](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | none public |
| Policy Gradient Methods for Reinforcement Learning with Function Approximation | 1999 | Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour | [`sutton-1999-policy-gradient.pdf`](policy-gradient/sutton-1999-policy-gradient.pdf) | [abs](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | none public |
| Trust Region Policy Optimization | 2015 | John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel | [`schulman-2015-trpo.pdf`](policy-gradient/schulman-2015-trpo.pdf) | [abs](https://arxiv.org/abs/1502.05477) | [link](https://github.com/joschu/modular_rl) |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | 2015 | John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel | [`schulman-2015-gae.pdf`](policy-gradient/schulman-2015-gae.pdf) | [abs](https://arxiv.org/abs/1506.02438) | [link](https://github.com/joschu/modular_rl) |
| Continuous Control with Deep Reinforcement Learning | 2015 | Timothy P. Lillicrap et al. | [`lillicrap-2015-ddpg.pdf`](policy-gradient/lillicrap-2015-ddpg.pdf) | [abs](https://arxiv.org/abs/1509.02971) | [link](https://github.com/openai/baselines) |
| Asynchronous Methods for Deep Reinforcement Learning | 2016 | Volodymyr Mnih et al. | [`mnih-2016-a3c.pdf`](policy-gradient/mnih-2016-a3c.pdf) | [abs](https://arxiv.org/abs/1602.01783) | [link](https://github.com/miyosuda/async_deep_reinforce) |
| Proximal Policy Optimization Algorithms | 2017 | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov | [`schulman-2017-ppo.pdf`](policy-gradient/schulman-2017-ppo.pdf) | [abs](https://arxiv.org/abs/1707.06347) | [link](https://github.com/openai/baselines) |
| Addressing Function Approximation Error in Actor-Critic Methods | 2018 | Scott Fujimoto, Herke van Hoof, David Meger | [`fujimoto-2018-td3.pdf`](policy-gradient/fujimoto-2018-td3.pdf) | [abs](https://arxiv.org/abs/1802.09477) | [link](https://github.com/sfujim/TD3) |
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | 2018 | Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine | [`haarnoja-2018-sac.pdf`](policy-gradient/haarnoja-2018-sac.pdf) | [abs](https://arxiv.org/abs/1801.01290) | [link](https://github.com/haarnoja/sac) |

## Imitation and offline (`imitation-offline/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Efficient Training of Artificial Neural Networks for Autonomous Navigation | 1991 | Dean A. Pomerleau | [`pomerleau-1991-alvinn.pdf`](imitation-offline/pomerleau-1991-alvinn.pdf) | [abs](https://www.ri.cmu.edu/publications/efficient-training-of-artificial-neural-networks-for-autonomous-navigation/) | none public |
| Maximum Entropy Inverse Reinforcement Learning | 2008 | Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey | [`ziebart-2008-maxent-irl.pdf`](imitation-offline/ziebart-2008-maxent-irl.pdf) | [abs](https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf) | none public |
| A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | 2011 | Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell | [`ross-2011-dagger.pdf`](imitation-offline/ross-2011-dagger.pdf) | [abs](https://arxiv.org/abs/1011.0686) | none public |
| Generative Adversarial Imitation Learning | 2016 | Jonathan Ho, Stefano Ermon | [`ho-ermon-2016-gail.pdf`](imitation-offline/ho-ermon-2016-gail.pdf) | [abs](https://arxiv.org/abs/1606.03476) | [link](https://github.com/openai/imitation) |
| Off-Policy Deep Reinforcement Learning without Exploration | 2019 | Scott Fujimoto, David Meger, Doina Precup | [`fujimoto-2019-bcq.pdf`](imitation-offline/fujimoto-2019-bcq.pdf) | [abs](https://arxiv.org/abs/1812.02900) | [link](https://github.com/sfujim/BCQ) |
| Conservative Q-Learning for Offline Reinforcement Learning | 2020 | Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine | [`kumar-2020-cql.pdf`](imitation-offline/kumar-2020-cql.pdf) | [abs](https://arxiv.org/abs/2006.04779) | [link](https://github.com/aviralkumar2907/CQL) |
| Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems | 2020 | Sergey Levine, Aviral Kumar, George Tucker, Justin Fu | [`levine-2020-offline-rl-survey.pdf`](imitation-offline/levine-2020-offline-rl-survey.pdf) | [abs](https://arxiv.org/abs/2005.01643) | none public |
| Offline Reinforcement Learning with Implicit Q-Learning | 2021 | Ilya Kostrikov, Ashvin Nair, Sergey Levine | [`kostrikov-2021-iql.pdf`](imitation-offline/kostrikov-2021-iql.pdf) | [abs](https://arxiv.org/abs/2110.06169) | [link](https://github.com/ikostrikov/implicit_q_learning) |

## Exploration, reward, curriculum (`exploration-curriculum/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Curriculum Learning | 2009 | Yoshua Bengio, Jerome Louradour, Ronan Collobert, Jason Weston | [`bengio-2009-curriculum.pdf`](exploration-curriculum/bengio-2009-curriculum.pdf) | [abs](https://doi.org/10.1145/1553374.1553380) | none public |
| Unifying Count-Based Exploration and Intrinsic Motivation | 2016 | Marc G. Bellemare et al. | [`bellemare-2016-count-based.pdf`](exploration-curriculum/bellemare-2016-count-based.pdf) | [abs](https://arxiv.org/abs/1606.01868) | none public |
| Hindsight Experience Replay | 2017 | Marcin Andrychowicz et al. | [`andrychowicz-2017-her.pdf`](exploration-curriculum/andrychowicz-2017-her.pdf) | [abs](https://arxiv.org/abs/1707.01495) | [link](https://github.com/openai/baselines) |
| Reverse Curriculum Generation for Reinforcement Learning | 2017 | Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel | [`florensa-2017-reverse-curriculum.pdf`](exploration-curriculum/florensa-2017-reverse-curriculum.pdf) | [abs](https://arxiv.org/abs/1707.05300) | none public |
| Exploration by Random Network Distillation | 2018 | Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov | [`burda-2018-rnd.pdf`](exploration-curriculum/burda-2018-rnd.pdf) | [abs](https://arxiv.org/abs/1810.12894) | [link](https://github.com/openai/random-network-distillation) |

## Practice (the debugging half) (`practice/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Deep Reinforcement Learning that Matters | 2017 | Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger | [`henderson-2017-deep-rl-matters.pdf`](practice/henderson-2017-deep-rl-matters.pdf) | [abs](https://arxiv.org/abs/1709.06560) | none public |
| Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO | 2020 | Logan Engstrom et al. | [`engstrom-2020-implementation-matters.pdf`](practice/engstrom-2020-implementation-matters.pdf) | [abs](https://arxiv.org/abs/2005.12729) | [link](https://github.com/MadryLab/implementation-matters) |
| What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study | 2020 | Marcin Andrychowicz et al. | [`andrychowicz-2020-what-matters.pdf`](practice/andrychowicz-2020-what-matters.pdf) | [abs](https://arxiv.org/abs/2006.05990) | none public |

## Preference-based and LLM RL (`preference-llm/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Deep Reinforcement Learning from Human Preferences | 2017 | Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei | [`christiano-2017-human-preferences.pdf`](preference-llm/christiano-2017-human-preferences.pdf) | [abs](https://arxiv.org/abs/1706.03741) | [link](https://github.com/nottombrown/rl-teacher) |
| Learning to Summarize from Human Feedback | 2020 | Nisan Stiennon et al. | [`stiennon-2020-summarize-hf.pdf`](preference-llm/stiennon-2020-summarize-hf.pdf) | [abs](https://arxiv.org/abs/2009.01325) | [link](https://github.com/openai/summarize-from-feedback) |
| Training Language Models to Follow Instructions with Human Feedback | 2022 | Long Ouyang et al. | [`ouyang-2022-instructgpt.pdf`](preference-llm/ouyang-2022-instructgpt.pdf) | [abs](https://arxiv.org/abs/2203.02155) | none public |
| Constitutional AI: Harmlessness from AI Feedback | 2022 | Yuntao Bai et al. | [`bai-2022-constitutional-ai.pdf`](preference-llm/bai-2022-constitutional-ai.pdf) | [abs](https://arxiv.org/abs/2212.08073) | [link](https://github.com/anthropics/ConstitutionalHarmlessnessPaper) |
| Scaling Laws for Reward Model Overoptimization | 2022 | Leo Gao, John Schulman, Jacob Hilton | [`gao-2022-scaling-laws-overoptimization.pdf`](preference-llm/gao-2022-scaling-laws-overoptimization.pdf) | [abs](https://arxiv.org/abs/2210.10760) | none public |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | 2023 | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn | [`rafailov-2023-dpo.pdf`](preference-llm/rafailov-2023-dpo.pdf) | [abs](https://arxiv.org/abs/2305.18290) | [link](https://github.com/eric-mitchell/direct-preference-optimization) |
| A Long Way to Go: Investigating Length Correlations in RLHF | 2023 | Prasann Singhal, Tanya Goyal, Jiacheng Xu, Greg Durrett | [`singhal-2023-length-bias.pdf`](preference-llm/singhal-2023-length-bias.pdf) | [abs](https://arxiv.org/abs/2310.03716) | [link](https://github.com/PrasannS/rlhf-length-biases) |
| Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs | 2024 | Arash Ahmadian et al. | [`ahmadian-2024-rloo.pdf`](preference-llm/ahmadian-2024-rloo.pdf) | [abs](https://arxiv.org/abs/2402.14740) | [link](https://github.com/huggingface/trl) |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2024 | Zhihong Shao et al. | [`shao-2024-deepseekmath-grpo.pdf`](preference-llm/shao-2024-deepseekmath-grpo.pdf) | [abs](https://arxiv.org/abs/2402.03300) | [link](https://github.com/deepseek-ai/DeepSeek-Math) |

## Normalisation and scale (`normalisation/`)

| Title | Year | Authors | Local file | Abstract / landing page | Original code |
| --- | --- | --- | --- | --- | --- |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | 2015 | Sergey Ioffe, Christian Szegedy | [`ioffe-szegedy-2015-batchnorm.pdf`](normalisation/ioffe-szegedy-2015-batchnorm.pdf) | [abs](https://arxiv.org/abs/1502.03167) | none public |
| Learning Values Across Many Orders of Magnitude | 2016 | Hado van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, David Silver | [`vanhasselt-2016-popart.pdf`](normalisation/vanhasselt-2016-popart.pdf) | [abs](https://arxiv.org/abs/1602.07714) | [link](https://github.com/google-deepmind/scalable_agent) |

## Notes on availability

All 48 papers above have a public PDF and are fetched by `fetch.py`.
None of the papers named in the plan required a paywall workaround.
Watkins 1989 is the freely-hosted PhD thesis, not the paywalled 1992 *Machine Learning* journal article.
Rummery & Niranjan 1994 is the freely-hosted Cambridge technical report.
Pomerleau's ALVINN entry uses the 1991 *Neural Computation* version hosted by CMU's Robotics Institute, not the paywalled journal page.

