#!/usr/bin/env python3
"""Fetch the reading list for rl_advanced_learning into papers/<tier>/.

Downloads every PDF listed in PAPERS into its tier subdirectory. Source is
always a public location: arXiv, a NeurIPS/ICML proceedings page, an author
homepage, or a university repository. Nothing is scraped from a paywall.

Usage:
    python papers/fetch.py                  # fetch everything not already present
    python papers/fetch.py --force           # re-download everything
    python papers/fetch.py --tier deep-value # fetch one tier only
    python papers/fetch.py --dry-run         # print what would happen, do nothing

Stdlib only: urllib, argparse, pathlib, json.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# arXiv (and some university sites) reject the default urllib User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (compatible; rl-advanced-learning-papers-fetch/1.0; "
    "+https://github.com/) urllib"
)

PAPERS_DIR = Path(__file__).resolve().parent

TIERS = (
    "tabular",
    "deep-value",
    "policy-gradient",
    "imitation-offline",
    "exploration-curriculum",
    "practice",
    "preference-llm",
    "normalisation",
)

# Each record: key (kebab-case slug -> filename stem), title, authors, year,
# tier (subdirectory), url (direct PDF), abs_url (landing page), code (URL of
# a reference implementation, or None if none is public / none was found).
PAPERS: list[dict] = [
    # -- tabular -----------------------------------------------------------
    dict(
        key="sutton-1988-td",
        title="Learning to Predict by the Methods of Temporal Differences",
        authors="Richard S. Sutton",
        year=1988,
        tier="tabular",
        url="http://incompleteideas.net/papers/sutton-88-with-erratum.pdf",
        abs_url="http://incompleteideas.net/papers/sutton-88-with-erratum.pdf",
        code=None,
    ),
    dict(
        key="watkins-1989-qlearning",
        title="Learning from Delayed Rewards",
        authors="Christopher J. C. H. Watkins",
        year=1989,
        tier="tabular",
        url="http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf",
        abs_url="http://www.cs.rhul.ac.uk/~chrisw/thesis.html",
        code=None,
    ),
    dict(
        key="rummery-niranjan-1994-sarsa",
        title="On-Line Q-Learning Using Connectionist Systems",
        authors="G. A. Rummery, M. Niranjan",
        year=1994,
        tier="tabular",
        url="http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf",
        abs_url="http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf",
        code=None,
    ),
    dict(
        key="sutton-1990-dyna",
        title=(
            "Integrated Architectures for Learning, Planning, and Reacting "
            "Based on Approximating Dynamic Programming"
        ),
        authors="Richard S. Sutton",
        year=1990,
        tier="tabular",
        url="http://incompleteideas.net/papers/sutton-90.pdf",
        abs_url="http://incompleteideas.net/papers/sutton-90.pdf",
        code=None,
    ),
    dict(
        key="vanhasselt-2010-doubleq",
        title="Double Q-learning",
        authors="Hado van Hasselt",
        year=2010,
        tier="tabular",
        url="https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf",
        abs_url="https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html",
        code=None,
    ),
    dict(
        key="ng-1999-shaping",
        title=(
            "Policy Invariance Under Reward Transformations: Theory and "
            "Application to Reward Shaping"
        ),
        authors="Andrew Y. Ng, Daishi Harada, Stuart Russell",
        year=1999,
        tier="tabular",
        url="https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf",
        abs_url="https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf",
        code=None,
    ),
    # -- deep-value ----------------------------------------------------------
    dict(
        key="mnih-2013-atari-dqn",
        title="Playing Atari with Deep Reinforcement Learning",
        authors="Volodymyr Mnih et al.",
        year=2013,
        tier="deep-value",
        url="https://arxiv.org/pdf/1312.5602",
        abs_url="https://arxiv.org/abs/1312.5602",
        code=None,
    ),
    dict(
        key="mnih-2015-dqn-nature",
        title="Human-level control through deep reinforcement learning",
        authors="Volodymyr Mnih et al.",
        year=2015,
        tier="deep-value",
        url="https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf",
        abs_url="https://www.nature.com/articles/nature14236",
        code="https://github.com/google-deepmind/dqn",
    ),
    dict(
        key="vanhasselt-2015-double-dqn",
        title="Deep Reinforcement Learning with Double Q-learning",
        authors="Hado van Hasselt, Arthur Guez, David Silver",
        year=2015,
        tier="deep-value",
        url="https://arxiv.org/pdf/1509.06461",
        abs_url="https://arxiv.org/abs/1509.06461",
        code="https://github.com/google-deepmind/dqn_zoo",
    ),
    dict(
        key="wang-2016-dueling",
        title="Dueling Network Architectures for Deep Reinforcement Learning",
        authors="Ziyu Wang et al.",
        year=2016,
        tier="deep-value",
        url="https://arxiv.org/pdf/1511.06581",
        abs_url="https://arxiv.org/abs/1511.06581",
        code="https://github.com/google-deepmind/dqn_zoo",
    ),
    dict(
        key="schaul-2015-per",
        title="Prioritized Experience Replay",
        authors="Tom Schaul, John Quan, Ioannis Antonoglou, David Silver",
        year=2015,
        tier="deep-value",
        url="https://arxiv.org/pdf/1511.05952",
        abs_url="https://arxiv.org/abs/1511.05952",
        code="https://github.com/google-deepmind/dqn_zoo",
    ),
    dict(
        key="hessel-2017-rainbow",
        title="Rainbow: Combining Improvements in Deep Reinforcement Learning",
        authors="Matteo Hessel et al.",
        year=2017,
        tier="deep-value",
        url="https://arxiv.org/pdf/1710.02298",
        abs_url="https://arxiv.org/abs/1710.02298",
        code="https://github.com/google-deepmind/dqn_zoo",
    ),
    # -- policy-gradient -------------------------------------------------
    dict(
        key="williams-1992-reinforce",
        title=(
            "Simple Statistical Gradient-Following Algorithms for "
            "Connectionist Reinforcement Learning"
        ),
        authors="Ronald J. Williams",
        year=1992,
        tier="policy-gradient",
        url="https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf",
        abs_url="https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf",
        code=None,
    ),
    dict(
        key="sutton-1999-policy-gradient",
        title=(
            "Policy Gradient Methods for Reinforcement Learning with "
            "Function Approximation"
        ),
        authors="Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour",
        year=1999,
        tier="policy-gradient",
        url="http://incompleteideas.net/papers/SMSM-NIPS99.pdf",
        abs_url="https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html",
        code=None,
    ),
    dict(
        key="schulman-2015-trpo",
        title="Trust Region Policy Optimization",
        authors="John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel",
        year=2015,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1502.05477",
        abs_url="https://arxiv.org/abs/1502.05477",
        code="https://github.com/joschu/modular_rl",
    ),
    dict(
        key="schulman-2015-gae",
        title="High-Dimensional Continuous Control Using Generalized Advantage Estimation",
        authors="John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel",
        year=2015,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1506.02438",
        abs_url="https://arxiv.org/abs/1506.02438",
        code="https://github.com/joschu/modular_rl",
    ),
    dict(
        key="schulman-2017-ppo",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov",
        year=2017,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1707.06347",
        abs_url="https://arxiv.org/abs/1707.06347",
        code="https://github.com/openai/baselines",
    ),
    dict(
        key="mnih-2016-a3c",
        title="Asynchronous Methods for Deep Reinforcement Learning",
        authors="Volodymyr Mnih et al.",
        year=2016,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1602.01783",
        abs_url="https://arxiv.org/abs/1602.01783",
        code="https://github.com/miyosuda/async_deep_reinforce",
    ),
    dict(
        key="lillicrap-2015-ddpg",
        title="Continuous Control with Deep Reinforcement Learning",
        authors="Timothy P. Lillicrap et al.",
        year=2015,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1509.02971",
        abs_url="https://arxiv.org/abs/1509.02971",
        code="https://github.com/openai/baselines",
    ),
    dict(
        key="fujimoto-2018-td3",
        title="Addressing Function Approximation Error in Actor-Critic Methods",
        authors="Scott Fujimoto, Herke van Hoof, David Meger",
        year=2018,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1802.09477",
        abs_url="https://arxiv.org/abs/1802.09477",
        code="https://github.com/sfujim/TD3",
    ),
    dict(
        key="haarnoja-2018-sac",
        title=(
            "Soft Actor-Critic: Off-Policy Maximum Entropy Deep "
            "Reinforcement Learning with a Stochastic Actor"
        ),
        authors="Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine",
        year=2018,
        tier="policy-gradient",
        url="https://arxiv.org/pdf/1801.01290",
        abs_url="https://arxiv.org/abs/1801.01290",
        code="https://github.com/haarnoja/sac",
    ),
    # -- imitation-offline ------------------------------------------------
    dict(
        key="pomerleau-1991-alvinn",
        title="Efficient Training of Artificial Neural Networks for Autonomous Navigation",
        authors="Dean A. Pomerleau",
        year=1991,
        tier="imitation-offline",
        url="https://www.ri.cmu.edu/pub_files/pub3/pomerleau_dean_1991_1/pomerleau_dean_1991_1.pdf",
        abs_url="https://www.ri.cmu.edu/publications/efficient-training-of-artificial-neural-networks-for-autonomous-navigation/",
        code=None,
    ),
    dict(
        key="ross-2011-dagger",
        title=(
            "A Reduction of Imitation Learning and Structured Prediction "
            "to No-Regret Online Learning"
        ),
        authors="Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell",
        year=2011,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/1011.0686",
        abs_url="https://arxiv.org/abs/1011.0686",
        code=None,
    ),
    dict(
        key="ho-ermon-2016-gail",
        title="Generative Adversarial Imitation Learning",
        authors="Jonathan Ho, Stefano Ermon",
        year=2016,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/1606.03476",
        abs_url="https://arxiv.org/abs/1606.03476",
        code="https://github.com/openai/imitation",
    ),
    dict(
        key="ziebart-2008-maxent-irl",
        title="Maximum Entropy Inverse Reinforcement Learning",
        authors="Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey",
        year=2008,
        tier="imitation-offline",
        url="https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf",
        abs_url="https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf",
        code=None,
    ),
    dict(
        key="fujimoto-2019-bcq",
        title="Off-Policy Deep Reinforcement Learning without Exploration",
        authors="Scott Fujimoto, David Meger, Doina Precup",
        year=2019,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/1812.02900",
        abs_url="https://arxiv.org/abs/1812.02900",
        code="https://github.com/sfujim/BCQ",
    ),
    dict(
        key="kumar-2020-cql",
        title="Conservative Q-Learning for Offline Reinforcement Learning",
        authors="Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine",
        year=2020,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/2006.04779",
        abs_url="https://arxiv.org/abs/2006.04779",
        code="https://github.com/aviralkumar2907/CQL",
    ),
    dict(
        key="levine-2020-offline-rl-survey",
        title=(
            "Offline Reinforcement Learning: Tutorial, Review, and "
            "Perspectives on Open Problems"
        ),
        authors="Sergey Levine, Aviral Kumar, George Tucker, Justin Fu",
        year=2020,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/2005.01643",
        abs_url="https://arxiv.org/abs/2005.01643",
        code=None,
    ),
    dict(
        key="kostrikov-2021-iql",
        title="Offline Reinforcement Learning with Implicit Q-Learning",
        authors="Ilya Kostrikov, Ashvin Nair, Sergey Levine",
        year=2021,
        tier="imitation-offline",
        url="https://arxiv.org/pdf/2110.06169",
        abs_url="https://arxiv.org/abs/2110.06169",
        code="https://github.com/ikostrikov/implicit_q_learning",
    ),
    # -- exploration-curriculum ------------------------------------------
    dict(
        key="andrychowicz-2017-her",
        title="Hindsight Experience Replay",
        authors="Marcin Andrychowicz et al.",
        year=2017,
        tier="exploration-curriculum",
        url="https://arxiv.org/pdf/1707.01495",
        abs_url="https://arxiv.org/abs/1707.01495",
        code="https://github.com/openai/baselines",
    ),
    dict(
        key="bellemare-2016-count-based",
        title="Unifying Count-Based Exploration and Intrinsic Motivation",
        authors="Marc G. Bellemare et al.",
        year=2016,
        tier="exploration-curriculum",
        url="https://arxiv.org/pdf/1606.01868",
        abs_url="https://arxiv.org/abs/1606.01868",
        code=None,
    ),
    dict(
        key="burda-2018-rnd",
        title="Exploration by Random Network Distillation",
        authors="Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov",
        year=2018,
        tier="exploration-curriculum",
        url="https://arxiv.org/pdf/1810.12894",
        abs_url="https://arxiv.org/abs/1810.12894",
        code="https://github.com/openai/random-network-distillation",
    ),
    dict(
        key="bengio-2009-curriculum",
        title="Curriculum Learning",
        authors="Yoshua Bengio, Jerome Louradour, Ronan Collobert, Jason Weston",
        year=2009,
        tier="exploration-curriculum",
        url="https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf",
        abs_url="https://doi.org/10.1145/1553374.1553380",
        code=None,
    ),
    dict(
        key="florensa-2017-reverse-curriculum",
        title="Reverse Curriculum Generation for Reinforcement Learning",
        authors="Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel",
        year=2017,
        tier="exploration-curriculum",
        url="https://arxiv.org/pdf/1707.05300",
        abs_url="https://arxiv.org/abs/1707.05300",
        code=None,
    ),
    # -- practice ----------------------------------------------------------
    dict(
        key="henderson-2017-deep-rl-matters",
        title="Deep Reinforcement Learning that Matters",
        authors="Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger",
        year=2017,
        tier="practice",
        url="https://arxiv.org/pdf/1709.06560",
        abs_url="https://arxiv.org/abs/1709.06560",
        code=None,
    ),
    dict(
        key="engstrom-2020-implementation-matters",
        title="Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO",
        authors="Logan Engstrom et al.",
        year=2020,
        tier="practice",
        url="https://arxiv.org/pdf/2005.12729",
        abs_url="https://arxiv.org/abs/2005.12729",
        code="https://github.com/MadryLab/implementation-matters",
    ),
    dict(
        key="andrychowicz-2020-what-matters",
        title="What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study",
        authors="Marcin Andrychowicz et al.",
        year=2020,
        tier="practice",
        url="https://arxiv.org/pdf/2006.05990",
        abs_url="https://arxiv.org/abs/2006.05990",
        code=None,
    ),
    # -- preference-llm ------------------------------------------------
    dict(
        key="christiano-2017-human-preferences",
        title="Deep Reinforcement Learning from Human Preferences",
        authors="Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei",
        year=2017,
        tier="preference-llm",
        url="https://arxiv.org/pdf/1706.03741",
        abs_url="https://arxiv.org/abs/1706.03741",
        code="https://github.com/nottombrown/rl-teacher",
    ),
    dict(
        key="stiennon-2020-summarize-hf",
        title="Learning to Summarize from Human Feedback",
        authors="Nisan Stiennon et al.",
        year=2020,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2009.01325",
        abs_url="https://arxiv.org/abs/2009.01325",
        code="https://github.com/openai/summarize-from-feedback",
    ),
    dict(
        key="ouyang-2022-instructgpt",
        title="Training Language Models to Follow Instructions with Human Feedback",
        authors="Long Ouyang et al.",
        year=2022,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2203.02155",
        abs_url="https://arxiv.org/abs/2203.02155",
        code=None,
    ),
    dict(
        key="bai-2022-constitutional-ai",
        title="Constitutional AI: Harmlessness from AI Feedback",
        authors="Yuntao Bai et al.",
        year=2022,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2212.08073",
        abs_url="https://arxiv.org/abs/2212.08073",
        code="https://github.com/anthropics/ConstitutionalHarmlessnessPaper",
    ),
    dict(
        key="rafailov-2023-dpo",
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        authors="Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn",
        year=2023,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2305.18290",
        abs_url="https://arxiv.org/abs/2305.18290",
        code="https://github.com/eric-mitchell/direct-preference-optimization",
    ),
    dict(
        key="ahmadian-2024-rloo",
        title=(
            "Back to Basics: Revisiting REINFORCE-Style Optimization for "
            "Learning from Human Feedback in LLMs"
        ),
        authors="Arash Ahmadian et al.",
        year=2024,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2402.14740",
        abs_url="https://arxiv.org/abs/2402.14740",
        code="https://github.com/huggingface/trl",
    ),
    dict(
        key="shao-2024-deepseekmath-grpo",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        authors="Zhihong Shao et al.",
        year=2024,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2402.03300",
        abs_url="https://arxiv.org/abs/2402.03300",
        code="https://github.com/deepseek-ai/DeepSeek-Math",
    ),
    dict(
        key="gao-2022-scaling-laws-overoptimization",
        title="Scaling Laws for Reward Model Overoptimization",
        authors="Leo Gao, John Schulman, Jacob Hilton",
        year=2022,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2210.10760",
        abs_url="https://arxiv.org/abs/2210.10760",
        code=None,
    ),
    dict(
        key="singhal-2023-length-bias",
        title="A Long Way to Go: Investigating Length Correlations in RLHF",
        authors="Prasann Singhal, Tanya Goyal, Jiacheng Xu, Greg Durrett",
        year=2023,
        tier="preference-llm",
        url="https://arxiv.org/pdf/2310.03716",
        abs_url="https://arxiv.org/abs/2310.03716",
        code="https://github.com/PrasannS/rlhf-length-biases",
    ),
    # -- normalisation ---------------------------------------------------
    dict(
        key="vanhasselt-2016-popart",
        title="Learning Values Across Many Orders of Magnitude",
        authors="Hado van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, David Silver",
        year=2016,
        tier="normalisation",
        url="https://arxiv.org/pdf/1602.07714",
        abs_url="https://arxiv.org/abs/1602.07714",
        code="https://github.com/google-deepmind/scalable_agent",
    ),
    dict(
        key="ioffe-szegedy-2015-batchnorm",
        title=(
            "Batch Normalization: Accelerating Deep Network Training by "
            "Reducing Internal Covariate Shift"
        ),
        authors="Sergey Ioffe, Christian Szegedy",
        year=2015,
        tier="normalisation",
        url="https://arxiv.org/pdf/1502.03167",
        abs_url="https://arxiv.org/abs/1502.03167",
        code=None,
    ),
]

assert set(p["tier"] for p in PAPERS) <= set(TIERS)
assert len(PAPERS) == len(set(p["key"] for p in PAPERS)), "duplicate key in PAPERS"


def target_path(paper: dict) -> Path:
    return PAPERS_DIR / paper["tier"] / f"{paper['key']}.pdf"


def download(paper: dict, force: bool, dry_run: bool) -> str:
    """Returns one of 'downloaded', 'skipped', 'failed'."""
    dest = target_path(paper)
    if dest.exists() and not force:
        print(f"skip     {dest}")
        return "skipped"

    if dry_run:
        print(f"would fetch {paper['url']} -> {dest}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(paper["url"], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"FAILED   {paper['key']}: {exc}", file=sys.stderr)
        return "failed"

    if not data.startswith(b"%PDF"):
        print(
            f"FAILED   {paper['key']}: response did not look like a PDF "
            f"(first bytes: {data[:16]!r})",
            file=sys.stderr,
        )
        return "failed"

    dest.write_bytes(data)
    print(f"wrote    {dest.resolve()}")
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--tier", choices=TIERS, default=None, help="fetch only this tier")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, do nothing")
    args = parser.parse_args()

    papers = [p for p in PAPERS if args.tier is None or p["tier"] == args.tier]

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []

    for paper in papers:
        result = download(paper, force=args.force, dry_run=args.dry_run)
        counts[result] += 1
        if result == "failed":
            failures.append(paper["key"])

    print()
    print(
        f"Summary: {counts['downloaded']} downloaded, "
        f"{counts['skipped']} skipped, {counts['failed']} failed "
        f"(of {len(papers)} attempted)"
    )
    if failures:
        print("Failed: " + ", ".join(failures))

    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
