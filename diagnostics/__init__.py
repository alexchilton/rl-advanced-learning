"""Tools for seeing what an agent is actually doing.

`render` draws grids in the terminal: value heatmaps, policy arrows, visit counts,
trajectories, and a learned-vs-exact comparison with the error panel that makes a
systematic bias obvious at a glance.

More to come, in build order: probe environments, the health panel, seed sweeps,
and the overfit-one-batch check.
"""

from diagnostics import render

__all__ = ["render"]
