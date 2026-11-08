"""Pictures of the environments, for notebooks. `render` is the terminal version.

    grid_figure(env, ...)       GridWorld or CliffWalking: walls, cliff, start, goal,
                                optional value heatmap, policy arrows, visit counts, path
    racetrack_figure(env, ...)  the track, start and finish lines, optional per-cell
                                values and a trajectory coloured by speed

Both draw into a matplotlib Axes and return it, so panels can sit side by side.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# (row, col) deltas shared by GridWorld and CliffWalking: up, right, down, left.
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _shape(env):
    if hasattr(env, "size"):
        return env.size, env.size
    return env.rows, env.cols


def grid_figure(env, ax=None, values=None, policy=None, visits=None, path=None, title="",
                cmap="viridis", show_values=False):
    """Draw a GridWorld or CliffWalking.

    values  per-state numbers (or a Q table, reduced by max) shown as a heatmap
    policy  per-state actions, or a Q table (reduced by argmax), shown as arrows
    visits  per-state counts, shown as a heatmap instead of `values`
    path    list of (row, col) cells, drawn as a line
    """
    rows, cols = _shape(env)
    ax = ax or plt.subplots(figsize=(0.55 * cols + 1.5, 0.55 * rows + 0.6))[1]
    blocked = set(getattr(env, "walls", ())) | set(getattr(env, "cliff", ()))

    heat = values if visits is None else visits
    if heat is not None:
        heat = np.asarray(heat, dtype=float)
        if heat.ndim == 2:
            heat = heat.max(axis=1)
        grid = np.full((rows, cols), np.nan)
        for s in range(rows * cols):
            if env.cell(s) not in blocked:
                grid[env.cell(s)] = heat[s]
        image = ax.imshow(grid, cmap=cmap if visits is None else "magma")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if show_values:
            for (r, c), v in np.ndenumerate(grid):
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
    else:
        ax.imshow(np.zeros((rows, cols)), cmap=ListedColormap(["#f4f4f4"]))

    for cell in getattr(env, "walls", ()):
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="#333"))
    for cell in getattr(env, "cliff", ()):
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="#c62828"))
    for cell, label, colour in ((env.start, "S", "#1565c0"), (env.goal, "G", "#2e7d32")):
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, fill=False, lw=2.5, ec=colour))
        ax.text(cell[1], cell[0], label, ha="center", va="center", fontsize=11, weight="bold", color=colour)

    if policy is not None:
        policy = np.asarray(policy)
        if policy.ndim == 2:
            policy = policy.argmax(axis=1)
        for s in range(rows * cols):
            cell = env.cell(s)
            if cell in blocked or cell == env.goal:
                continue
            dr, dc = _DELTAS[int(policy[s])]
            ax.annotate("", xy=(cell[1] + 0.32 * dc, cell[0] + 0.32 * dr),
                        xytext=(cell[1] - 0.32 * dc, cell[0] - 0.32 * dr),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color="white" if heat is not None else "#222"))

    if path is not None:
        ax.plot([c for _, c in path], [r for r, _ in path], "o-", color="#ff9800", lw=2, ms=4)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    return ax


def racetrack_figure(env, ax=None, states=None, cell_values=None, title="", cmap="viridis"):
    """Draw the Racetrack. `states` is a trajectory of state indices; `cell_values`
    is a (rows, cols) array such as `env.cell_values(Q)`."""
    ax = ax or plt.subplots(figsize=(0.35 * env.cols + 1.5, 0.35 * env.rows + 0.6))[1]
    base = np.full((env.rows, env.cols), np.nan)
    for cell in env.drivable:
        base[cell] = 0.0
    if cell_values is not None:
        image = ax.imshow(np.where(np.isnan(base), np.nan, cell_values), cmap=cmap)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(base, cmap=ListedColormap(["#e8e8e8"]))
    ax.set_facecolor("#5d4037")  # off-track
    for cell in env.start_cells:
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="#1565c0", alpha=0.8))
    for cell in env.finish_cells:
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="#2e7d32", alpha=0.8))

    if states is not None:
        points = []
        for s in states:
            if int(s) == env.finished_state:
                continue
            (r, c), v_up, v_right = env.decode(int(s))
            points.append((r, c, max(v_up, v_right)))
        if points:
            rs, cs, speed = zip(*points)
            ax.plot(cs, rs, "-", color="#ff9800", lw=1.5, alpha=0.8)
            dots = ax.scatter(cs, rs, c=speed, cmap="autumn_r", vmin=0, vmax=env.max_speed, s=45,
                              edgecolors="k", zorder=3)
            plt.colorbar(dots, ax=ax, fraction=0.046, pad=0.04, label="speed")

    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    return ax
