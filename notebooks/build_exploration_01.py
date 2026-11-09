"""Build notebooks/exploration_01_bonus_curriculum_her.ipynb. Imports from this repo; run from the repo root."""

import json
import pathlib

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ----------------------------------------------------------------------
md(r"""
# Exploration: bonuses, curricula and hindsight on sparse rewards

**Code:** [`count_based_exploration.py`](../algos/count_based_exploration.py) · [`curriculum_reverse.py`](../algos/curriculum_reverse.py) · [`curriculum_size.py`](../algos/curriculum_size.py) · [`curriculum_automatic.py`](../algos/curriculum_automatic.py) · [`curriculum_failure.py`](../algos/curriculum_failure.py) · [`noisy_dqn.py`](../algos/noisy_dqn.py) · [`rnd.py`](../algos/rnd.py) · [`her.py`](../algos/her.py) · [`envs/bit_flip.py`](../envs/bit_flip.py) · [`experiments/curriculum_vs_exploration.py`](../experiments/curriculum_vs_exploration.py) · [`experiments/rnd_chain.py`](../experiments/rnd_chain.py) · [`experiments/noisy_chain.py`](../experiments/noisy_chain.py) · [`experiments/curriculum_failure.py`](../experiments/curriculum_failure.py)
**Papers:** Bengio 2009 ([PDF](../papers/exploration-curriculum/bengio-2009-curriculum.pdf)), Bellemare 2016 ([PDF](../papers/exploration-curriculum/bellemare-2016-count-based.pdf)), Andrychowicz 2017 HER ([PDF](../papers/exploration-curriculum/andrychowicz-2017-her.pdf)), Florensa 2017 ([PDF](../papers/exploration-curriculum/florensa-2017-reverse-curriculum.pdf)), Matiisen 2017 ([PDF](../papers/exploration-curriculum/matiisen-2017-teacher-student.pdf)), Fortunato 2017 Noisy ([PDF](../papers/deep-value/fortunato-2017-noisy-nets.pdf)), Burda 2018 RND ([PDF](../papers/exploration-curriculum/burda-2018-rnd.pdf)) ·
**Notes:** [`papers/exploration-curriculum/notes.md`](../papers/exploration-curriculum/notes.md)

On a sparse task nothing is learned until the reward is seen once. Every method here is a way of seeing it sooner.

What you will see:

1. Why ε-greedy is not exploration: on a 30-state chain it reaches the goal with probability $2^{-29}$ per episode.
2. Where each method sends the agent, as a picture of state visits.
3. **Seven methods on the same two tasks, one metric**: four curricula (two hand-written, two chosen by a teacher or at random), a count bonus, optimistic initialisation, and ε-greedy.
4. **The bonus trap:** a bonus added to the reward inflates the values the policy is read from. Measured on the count bonus and on a one-head RND.
5. **RND with two value heads** on long chains, with the action indices scrambled per state so that an untrained network cannot walk to the goal by accident.
6. **HER** on bit flipping: 0% for plain DQN from 10 bits, 100% with hindsight up to 20.
7. **Noisy networks** against ε-greedy on the same scrambled chains.
8. **A curriculum that forgets:** four goals trained in turn by one network, every goal re-evaluated throughout.

Sections 1, 3, part of 5 and part of 6 run live (about 2 minutes on CPU). The rest reads `experiments/results/exploration/`;
`python experiments/curriculum_vs_exploration.py`, `rnd_chain.py`, `noisy_chain.py`, `curriculum_failure.py` and `python algos/her.py` regenerate it.
""")

code(r"""
import json, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import torch

from algos import count_based_exploration as counts, curriculum_reverse, curriculum_size, her
from diagnostics.plots import grid_figure
from envs.bit_flip import BitFlip
from envs.chain import Chain
from envs.gridworld import GridWorld

torch.set_num_threads(1)
plt.rcParams["figure.dpi"] = 110
GAMMA = 0.99
ROOT = pathlib.Path("..") if pathlib.Path("../algos").exists() else pathlib.Path(".")
RESULTS = ROOT / "experiments/results/exploration"
""")

# ----------------------------------------------------------------------
md(r"""
## 0. The problems

**Chain(n)**: n states in a line, start at the left. "Right" moves one step; "left" sends the agent back to the start. The only reward is +1 for reaching the right end.

**GridWorld 14×14, sparse**: +1 for entering the goal corner, nothing else.

**Bit flipping (n bits)**: the state and a random goal are both n bits; each action flips one bit; reward −1 per step until the state equals the goal.

| task | states | what a random policy does |
| --- | --- | --- |
| Chain(30) | 30 | reaches the goal with probability $2^{-29} \approx 1.9 \times 10^{-9}$ per episode |
| GridWorld 14×14 | 196 | reaches the goal by chance often enough: ε-greedy solves it |
| bit flipping, 20 bits | $2^{20}$ states × $2^{20}$ goals | matches the goal with probability about $2^{-20}$ per step |
""")

code(r"""
print(f"{'n':>4} {'P(random walk reaches goal in one episode)':>44}")
for n in (5, 10, 20, 30):
    print(f"{n:>4} {Chain(n=n).random_walk_success_probability():>44.2e}")

fig, axs = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
ax = axs[0]; n = 12
for s in range(n):
    ax.add_patch(plt.Rectangle((s, 0), 0.9, 0.9, fc="#2e7d32" if s == n - 1 else ("#1565c0" if s == 0 else "#e8e8e8")))
    if s < n - 1:
        ax.annotate("", xy=(s + 1.0, 0.65), xytext=(s + 0.85, 0.65), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("", xy=(0.45, 0.95), xytext=(7.45, 0.95), arrowprops=dict(arrowstyle="->", lw=1, color="#c62828", connectionstyle="arc3,rad=0.3"))
ax.text(4, 1.55, "'left' from anywhere: back to the start", color="#c62828", ha="center")
ax.text(0.45, -0.3, "start", ha="center"); ax.text(n - 0.55, -0.3, "+1", ha="center", color="#2e7d32")
ax.set_xlim(-0.3, n + 0.2); ax.set_ylim(-0.6, 2); ax.axis("off"); ax.set_title("Chain(n), drawn for n = 12")
grid_figure(GridWorld(size=14, reward_mode="sparse"), axs[1], title="GridWorld 14x14, sparse")
plt.show()
""")

# ----------------------------------------------------------------------
md(r"""
## 1. Where each method sends the agent

20,000 steps of tabular Q-learning on Chain(30), counting visits per state.
ε-greedy is the random walk until it has found something, so it stays near the start.
Optimistic initialisation (every Q starts at 1, the largest possible return) and the count bonus both make untried actions look best, so the greedy policy itself walks to the frontier.
""")

code(r"""
class Counting(Chain):
    def __init__(self, **kw):
        super().__init__(**kw); self.visits = np.zeros(self.n)
    def step(self, action):
        out = super().step(action); self.visits[self.state] += 1; return out

methods = {"ε-greedy": {}, "optimistic Q (Q0 = 1)": dict(q_init=1.0), "count bonus β = 0.1": dict(bonus=0.1)}
fig, axs = plt.subplots(len(methods), 1, figsize=(11, 3.6), sharex=True)
for ax, (name, kw) in zip(axs, methods.items()):
    env = Counting(n=30, max_steps=120)
    _, _, first = counts.train(env, n_steps=20_000, eval_every=20_000, seed=0, **kw)
    ax.imshow(np.log10(env.visits + 1)[None], aspect="auto", cmap="magma", vmin=0, vmax=4)
    ax.set_yticks([]); ax.set_title(f"{name}: goal first reached at step {first}", fontsize=9, loc="left")
axs[-1].set_xlabel("chain state (start = 0, goal = 29)"); plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------
md(r"""
## 2. Seven methods, two tasks, one metric

Steps until the greedy policy, evaluated **exactly** from the real start, reaches 95% of the optimum. Every step counts, including the curricula's easy stages. 5 seeds, budget 200,000.

| method | how it sees the reward sooner | what it needs |
| --- | --- | --- |
| optimistic Q | untried actions look best until tried | nothing; one line |
| count bonus | $r + \beta/\sqrt{N(s,a)}$ | counts, so a finite or pseudo-counted state space |
| reverse curriculum | start next to the goal, move outward ([`curriculum_reverse.py`](../algos/curriculum_reverse.py)) | **resetting to any state** |
| size curriculum | solve a small instance, copy Q aligned at the goal ([`curriculum_size.py`](../algos/curriculum_size.py)) | **a family of tasks and a mapping between them** |
| random task | start each episode in a random band of distance to the goal | resetting to any state |
| learning progress | a teacher picks the band whose success rate is changing fastest ([`curriculum_automatic.py`](../algos/curriculum_automatic.py)) | resetting to any state |
""")

code(r"""
data = json.loads((RESULTS / "curriculum_vs_exploration.json").read_text())
methods = ["epsilon-greedy", "optimistic Q", "count bonus", "reverse curriculum", "size curriculum", "random task", "learning progress"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, task in zip(axs, data["table"]):
    print(task)
    for i, m in enumerate(methods):
        hits = data["table"][task][m]
        found = [h for h in hits if h is not None]
        label = f"{int(np.median(found)):,} [{min(found):,}, {max(found):,}]" if len(found) >= 3 else f"never ({len(found)}/5)"
        print(f"  {m:>19}: {label}")
        ax.scatter([h if h is not None else data['budget'] * 1.3 for h in hits], [i] * len(hits), s=25,
                   c=["tab:blue" if h is not None else "tab:red" for h in hits])
    ax.set_xscale("log"); ax.set_yticks(range(len(methods)), methods); ax.set_title(task)
    ax.axvline(data["budget"], color="k", ls=":", lw=1); ax.set_xlabel("steps to 95% of optimum (red = never, plotted past the budget)")
plt.tight_layout(); plt.show()
""")

md(r"""
- **Chain(30):** ε-greedy never solves it. The size curriculum (850) and optimistic Q (1,300) are fastest, then the reverse curriculum (16,500), the learning-progress teacher (45,000), random bands (58,800) and the count bonus (72,500).
- **GridWorld 14×14:** ε-greedy is fine here (23,000). The size curriculum is still fastest (1,400). Random bands (15,000) beat the learning-progress teacher (16,500) and the hand-written reverse schedule (20,000). The count bonus **never** finishes, which section 3 explains.
- **The teacher against random:** the teacher wins on the chain and loses on the grid; on the grid random task choice is the best of the three start-distribution methods. Matiisen et al. report the same split: their teacher an order of magnitude faster than uniform sampling on a Minecraft maze, uniform sampling "competitive" on 2-D addition.
- The reverse schedule decides when to move outward with extra greedy rollouts that are not counted as steps. The teacher uses only its training episodes. The comparison favours the reverse schedule.
- The cheapest method, optimistic initialisation, is as good as anything that needs no extra machinery. Most tutorials skip it.
- The size curriculum's lead depends on an unusually easy transfer: every cell of a chain or an open grid looks alike relative to the goal.
""")

# ----------------------------------------------------------------------
md(r"""
## 3. The bonus trap

A bonus added to the reward becomes part of what Q estimates. The greedy policy then optimises goal **plus** bonus, and Q at the start can exceed any achievable return.
Measured on the count bonus (β = 0.1, 200,000 steps on the 14×14 grid), then with the same table trained 20,000 more steps with the bonus switched off.
""")

code(r"""
grid = GridWorld(size=14, reward_mode="sparse", max_steps=300)
optimal = grid.optimal_return(GAMMA)
t0 = time.time()
Q, curve, first = counts.train(grid, n_steps=200_000, bonus=0.1, eval_every=10_000, seed=0)
Q_off, curve_off, _ = counts.train(grid, n_steps=20_000, bonus=0.0, eval_every=2_000, seed=100, Q=Q)
print(f"({time.time() - t0:.0f} s)  goal first reached at step {first}")
start = grid.index(grid.start)
print(f"max Q at the start: {Q[start].max():.3f}   exact optimal value: {optimal:.3f}")
print(f"greedy return during the bonus run: {[round(v, 3) for _, v in curve[::4]]}")
print(f"after 20,000 more steps with the bonus off: {[round(v, 3) for _, v in curve_off]}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
grid_figure(grid, axs[0], values=Q.max(1), policy=Q, title="count bonus: learned max Q (inflated)")
grid_figure(grid, axs[1], values=grid.true_v(GAMMA), policy=grid.optimal_policy(GAMMA), title="exact V* and optimal policy")
for ax in axs: ax.images[0].set_clim(0, 1.25)
plt.show()
""")

md(r"""
The left map is above 1 almost everywhere, though no return here can exceed 1: those are bonus values, and the arrows chase them.
Switching the bonus off does not repair it within 20,000 steps; values that high decay slowly.

The same thing happened to RND written the obvious way, with one value head and the bonus added to the reward.
On Chain(20) (the plain chain, before the scrambling of section 6) it ended with a greedy return of 0 on 4 of 5 seeds, its Q flat at about 1.438 for both actions in every state, above the largest achievable return of 1.0.
Fading the bonus and recomputing it per batch did not help. The fix is the paper's design: a second value head.
""")

# ----------------------------------------------------------------------
md(r"""
## 4. RND with two value heads

`algos/rnd.py`: a fixed random network $f$ and a predictor $\hat f$; novelty $\lVert \hat f(s') - f(s')\rVert^2$, normalised.
The extrinsic head $Q_E$ learns the task reward; the intrinsic head $Q_I$ learns the novelty and ignores episode ends.
The agent acts on $Q_E + \beta Q_I$; the policy that is evaluated is $Q_E$ alone, so the bonus never enters it.
Chain(20) and Chain(30), one-hot observations, 50,000 steps, 5 seeds. The action indices are scrambled per state (section 6 shows why).
""")

code(r"""
rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("rnd_chain__*.json"))]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, n in zip(axs, (20, 30)):
    for name, colour in (("DQN", "tab:grey"), ("DQN + RND", "tab:green")):
        sel = [r for r in rows if r["n"] == n and r["name"] == name]
        firsts = [r["first_goal"] for r in sel]
        optimal_end = sum(r["log"][-1]["score"] >= 0.95 * r["optimum"] for r in sel)
        print(f"Chain({n}) {name:>10}: first goal {firsts}; greedy policy optimal at the end on {optimal_end}/{len(sel)}")
        for r in sel:
            ax.plot([e["step"] for e in r["log"]], [e["score"] for e in r["log"]], "o-", color=colour, alpha=0.6, ms=3,
                    label=name if r is sel[0] else None)
    ax.axhline(sel[0]["optimum"], color="k", ls="--", lw=1); ax.set_title(f"Chain({n}): exact greedy return, each seed")
    ax.set_xlabel("environment steps"); ax.legend(fontsize=8)
plt.show()
""")

md(r"""
RND reaches the goal on every seed, by step 3,802 on Chain(20) and 4,903 on Chain(30), and ends with the optimal policy on all 10 runs.
DQN reaches it on 2 of 5 seeds on each chain, late (11,403 to 48,742 steps), and ends optimal on 2 of 5 and 0 of 5.

The first run of this section used the plain chain, where "right" is the same action index in every state, and had DQN optimal on 4 of 5 and 2 of 5.
Most of that was a fresh network that already preferred "right", not exploration; section 6 measures it.
""")

# ----------------------------------------------------------------------
md(r"""
## 5. Hindsight Experience Replay

A failed bit-flipping episode still reached *some* state. HER stores each transition 4 more times, with the goal replaced by a state reached later in the episode and the reward recomputed ([`algos/her.py:70-77`](../algos/her.py)).
Some of those copies are successes, so the goal-conditioned network gets signal from the first episode.

First live, 10 bits, 1,000 episodes each:
""")

code(r"""
make = lambda: BitFlip(10)
t0 = time.time()
live = {k: her.train(make(), n_episodes=1_000, k=k, evaluate=her.success_rate(make, episodes=30), eval_every=100, seed=0)[1]
        for k in (0, 4)}
print(f"({time.time() - t0:.0f} s)")
for k, log in live.items():
    plt.plot([r["episode"] for r in log], [r["success"] for r in log], "o-", ms=3, label="DQN" if k == 0 else "DQN + HER (k=4)")
plt.xlabel("episodes"); plt.ylabel("greedy success on fresh goals"); plt.title("bit flipping, 10 bits"); plt.legend(); plt.show()
""")

md(r"""
Then the full sweep from `python algos/her.py`, 4,000 episodes, 5 to 20 bits:
""")

code(r"""
sweep = json.loads((RESULTS / "her_bits.json").read_text())
print(f"{'bits':>5} {'DQN':>6} {'DQN + HER':>10}")
for n in (5, 10, 15, 20):
    print(f"{n:>5} {sweep[f'{n} bits, k=0'][-1]['success']:>6.0%} {sweep[f'{n} bits, k=4'][-1]['success']:>10.0%}")
fig, axs = plt.subplots(1, 4, figsize=(15, 2.8), sharey=True)
for ax, n in zip(axs, (5, 10, 15, 20)):
    for k, label in ((0, "DQN"), (4, "HER")):
        log = sweep[f"{n} bits, k={k}"]
        ax.plot([r["episode"] for r in log], [r["success"] for r in log], "o-", ms=3, label=label)
    ax.set_title(f"{n} bits"); ax.set_xlabel("episodes")
axs[0].set_ylabel("success"); axs[0].legend(fontsize=8); plt.show()
""")

# ----------------------------------------------------------------------
md(r"""
## 6. Noisy networks against ε-greedy, on chains that cannot be solved by accident

On the plain chain, "right" is action 1 in **every** state. A fresh network whose argmax happens to favour index 1 walks to the goal without exploring at all.
Measured below: the share of Chain(20)'s states in which a freshly initialised network already prefers "right", for 5 seeds.
""")

code(r"""
from algos.dqn import QNetwork
from algos.noisy_dqn import NoisyQNetwork
from envs.chain import ScrambledActions
onehot = torch.eye(20)
for name, make in (("DQN", lambda: QNetwork(20, 2)), ("Noisy DQN, mean weights", lambda: NoisyQNetwork(20, 2))):
    shares = []
    for seed in range(5):
        torch.manual_seed(seed); net = make(); net.eval()
        with torch.no_grad():
            shares.append(float((net(onehot).argmax(1) == 1).float().mean()))
    print(f"{name:>24}: share of states preferring 'right' before training {[round(x, 2) for x in shares]}")
scrambled = ScrambledActions(Chain(n=20, max_steps=80, obs_mode="onehot"), seed=0)
print(f"\nscrambled chain, seed 0: action index meaning 'right' per state {[int(np.argsort(p)[1]) for p in scrambled.perm]}")
print(f"return of 'always action 0' {scrambled.policy_return(np.zeros(20, int), GAMMA):.3f}, "
      f"'always action 1' {scrambled.policy_return(np.ones(20, int), GAMMA):.3f}, optimum {scrambled.optimal_return(GAMMA):.3f}")
""")

md(r"""
Three of five fresh DQN networks prefer "right" in every state. The RND runs of section 4 and the runs below therefore use `envs.chain.ScrambledActions`: per state and per seed, a random permutation of what the action indices mean (the construction of Osband et al.'s Deep Sea).
No constant action solves it, and the exact optimum is unchanged.

The comparison ([`experiments/noisy_chain.py`](../experiments/noisy_chain.py)): DQN with ε decaying from 1 to 0.05, against Noisy DQN with ε = 0 and the weight noise resampled every step (the paper's DQN agent) or held for a whole episode. 50,000 steps, 5 seeds.
""")

code(r"""
noisy_rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("noisy_chain__*.json"))]
agents = ["DQN (epsilon-greedy)", "Noisy DQN", "Noisy DQN, per episode"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.2))
for ax, n in zip(axs, (20, 30)):
    for k, name in enumerate(agents):
        sel = [r for r in noisy_rows if r["n"] == n and r["name"] == name]
        firsts = [r["first_goal"] for r in sel]
        solved = sum(r["log"][-1]["score"] >= 0.95 * r["optimum"] for r in sel)
        print(f"Chain({n}) {name:>23}: first goal {firsts}; optimal at the end on {solved}/{len(sel)}")
        ax.scatter([f if f is not None else 60_000 for f in firsts], [k] * len(firsts), s=30,
                   c=["tab:blue" if f is not None else "tab:red" for f in firsts])
    ax.set_yticks(range(len(agents)), agents); ax.axvline(50_000, color="k", ls=":", lw=1)
    ax.set_title(f"Chain({n}): step the goal was first reached (red = never)"); ax.set_xlabel("environment steps")
plt.tight_layout(); plt.show()
""")

md(r"""
Noisy networks never reach the goal on either chain, whether the noise is resampled every step or held for the episode. ε-greedy reaches it on 2 of 5 seeds.

Measured at initialisation: the weight noise changes the greedy action in 6.6% to 28.3% of (state, noise sample) pairs, over 5 seeds. ε-greedy at ε = 1 changes it in 50%.
So at the start the noise explores less than ε-greedy's schedule does, and on a chain where one wrong step returns you to the start that difference compounds over every step.
That this is the whole reason is a HYPOTHESIS (unverified); a larger σ₀ would test it.
The original paper's gains are on Atari, not on chains built to defeat undirected exploration. Noise is undirected too: nothing about it seeks the unvisited states, which is what RND (section 4) and optimistic initialisation do.
""")

# ----------------------------------------------------------------------
md(r"""
## 7. A curriculum that forgets

Every curriculum above uses a table, and a table cannot forget: each state has its own entry.
A network can. [`algos/curriculum_failure.py`](../algos/curriculum_failure.py): a 7×7 grid, start in the centre, observation $[row, col, goal_{row}, goal_{col}]$, so **one** network can solve every goal.
Stage k is goal k, the four corners in turn, 30,000 DQN steps each. Every goal is re-evaluated every 1,000 steps: the curve the plan says nobody plots.

| arm | what is in the training batches |
| --- | --- |
| plain | replay of 2,000 transitions: the current goal only |
| rehearsal | the same, but a quarter of episodes go to a goal from an earlier stage |
| big replay | replay of 120,000: every transition of every stage is kept |

5 seeds per arm, from [`experiments/curriculum_failure.py`](../experiments/curriculum_failure.py).
""")

code(r"""
fail = {}
for p in sorted(RESULTS.glob("curriculum_failure__*.json")):
    r = json.loads(p.read_text()); fail.setdefault(r["name"], []).append(r)
arms = ["plain", "rehearsal", "big replay"]
fig, axs = plt.subplots(1, 3, figsize=(15, 3.2), sharey=True)
for ax, arm in zip(axs, arms):
    steps = [e["step"] for e in fail[arm][0]["log"]]
    success = np.mean([[e["success"] for e in r["log"]] for r in fail[arm]], axis=0)   # (evals, goals)
    for g in range(4):
        ax.plot(steps, success[:, g], label=f"goal {g + 1}")
    for k in range(1, 4):
        ax.axvline(k * fail[arm][0]["stage_steps"], color="k", ls=":", lw=1)
    ax.set_title(f"{arm}: share of seeds solving each goal"); ax.set_xlabel("environment steps (dotted = next stage)")
axs[0].set_ylabel("greedy success"); axs[0].legend(fontsize=8); plt.show()
print("seeds solving each goal after the last stage (goal 4 trained last):")
for arm in arms:
    print(f"  {arm:>10}: {np.sum([r['log'][-1]['success'] for r in fail[arm]], axis=0).tolist()} of {len(fail[arm])}")
""")

md(r"""
- **Plain:** after the last stage, goals 1 to 3 are solved on **0 of 5** seeds, though each was solved when its own stage ended. Only the goal trained last survives. A curriculum that evaluates only the current stage would report success throughout.
- **Rehearsal:** a quarter of episodes on earlier goals keeps them: 5, 4, 4 and 5 of 5 seeds at the end.
- **Big replay:** all four goals on 5 of 5. A buffer that keeps old stages is rehearsal by another name, which is one reason ordinary DQN runs rarely show this.
""")

# ----------------------------------------------------------------------
md(r"""
## 8. What breaks, and how you would know

| method | symptom | likely cause |
| --- | --- | --- |
| count bonus | Q above any achievable return, greedy policy wanders | the bonus is in the reward and has not decayed (section 3) |
| count bonus | stops trying an action in a well-visited state | counting states instead of (state, action) pairs |
| optimistic Q | very slow to settle | initial value set far above the best return; set it just above |
| RND | finds the goal, then the greedy policy fails | bonus mixed into the one value head; use two heads (section 4) |
| RND | bonus constant from the start | predictor too large or learning too fast; it fits everything immediately |
| HER | no better than DQN | relabelled copies keep the old reward; it must be recomputed for the new goal |
| HER | relabelled goals from before the transition | "future" means a state reached at or after t |
| reverse curriculum | start radius stops growing | success threshold out of reach at this ε and chunk size |
| size curriculum | a stage takes as long as from scratch | the transfer map is misaligned with the goal |
| any deep agent on a chain | solves it at step ~20, before any exploration | the same action index is "right" everywhere; scramble the actions (section 6) |
| learning-progress teacher | keeps choosing the easiest task | progress measured as the success rate, not its change |
| Noisy DQN | never leaves the start region | the noise rarely changes the greedy action; it is undirected (section 6) |
| any curriculum with a network | early stages fail after later ones | forgetting; evaluate every stage, rehearse, or keep old data in replay (section 7) |
| reverse curriculum or bands | a start counts as success in one step, with no reward | the absorbing goal cell was a start state (`distance_to_goal` now excludes it) |
""")


def build():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = pathlib.Path(__file__).with_name("exploration_01_bonus_curriculum_her.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    build()
