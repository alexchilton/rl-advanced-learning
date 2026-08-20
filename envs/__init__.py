"""Tiny environments with exact ground truth.

Every finite environment here defines its dynamics once, as a transition table
`P`, and `step()` samples from that table. `envs.solvers` reads the same table.
So the "true" Q function cannot drift away from the environment the agent is
actually interacting with, and a disagreement between the two always means the
agent is wrong.

That guarantee is the reason these exist alongside Gymnasium rather than instead
of it. Gymnasium tells you whether an implementation matches the field. These tell
you whether it matches the truth.

The interface is Gymnasium's throughout:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from envs import solvers
from envs.bandit import Bandit
from envs.chain import Baird, Chain
from envs.cliff import CliffWalking
from envs.gridworld import GridWorld
from envs.pointmass import PointMass
from envs.racetrack import Racetrack
from envs.tabular import TabularEnv

__all__ = [
    "Baird",
    "Bandit",
    "Chain",
    "CliffWalking",
    "GridWorld",
    "PointMass",
    "Racetrack",
    "TabularEnv",
    "solvers",
]
