"""Colour terminal rendering for grid environments.

Plots are for reports. This is for the debugging loop: no files, no window, no
waiting, and it prints straight into the log next to the numbers that produced it.

Everything here is 24-bit ANSI colour, which every modern terminal supports. Set
`NO_COLOR=1` in the environment and it degrades to plain ASCII rather than
spraying escape codes into a file.

    from diagnostics.render import value_heatmap, policy_overlay, compare_grids
    print(value_heatmap(env, env.true_v(0.95), "exact V*"))
    print(policy_overlay(env, Q_learned, "what the agent thinks"))

The most useful function is `compare_grids`, which puts a learned value function
next to the exact one and colours the difference. Staring at a reward curve tells
you that something is wrong. This tells you where.
"""

from __future__ import annotations

import os

import numpy as np

RESET = "\033[0m"

# Blue (low) through white to red (high). Diverging, because value functions are
# usually signed and a sequential map hides the sign.
COLD = (40, 90, 190)
MID = (245, 245, 245)
HOT = (215, 60, 50)

# Sequential, for quantities with a natural zero like visit counts and errors.
DARK = (20, 20, 45)
BRIGHT = (250, 215, 90)

ARROWS = ("^", ">", "v", "<")


def colour_enabled() -> bool:
    return not os.environ.get("NO_COLOR")


def _blend(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _diverging(t):
    """t in [0, 1], 0.5 is the midpoint."""
    return _blend(COLD, MID, t * 2) if t < 0.5 else _blend(MID, HOT, (t - 0.5) * 2)


def _sequential(t):
    return _blend(DARK, BRIGHT, t)


def _cell(text: str, background, foreground=None) -> str:
    if not colour_enabled():
        return text
    r, g, b = background
    # Pick the text colour by luminance so it stays readable at both ends.
    if foreground is None:
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        foreground = (20, 20, 20) if luminance > 140 else (250, 250, 250)
    fr, fg, fb = foreground
    return f"\033[48;2;{r};{g};{b}m\033[38;2;{fr};{fg};{fb}m{text}{RESET}"


def _normalise(values, symmetric: bool):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values), 0.0, 0.0
    low, high = float(finite.min()), float(finite.max())
    if symmetric:
        extent = max(abs(low), abs(high)) or 1.0
        low, high = -extent, extent
    if high - low < 1e-12:
        return np.full_like(values, 0.5), low, high
    return (values - low) / (high - low), low, high


def _grid_shape(env):
    """(rows, cols) for GridWorld or CliffWalking, which name them differently."""
    if hasattr(env, "rows"):
        return env.rows, env.cols
    return env.size, env.size


def _blocked(env, cell) -> bool:
    return cell in getattr(env, "walls", frozenset()) or cell in getattr(env, "cliff", frozenset())


def _to_state_values(values, n_states):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[0] == n_states:
        return values.max(axis=1)  # a Q table -- show V(s) = max_a Q(s,a)
    return values.reshape(n_states)


# ----------------------------------------------------------------------
def value_heatmap(env, values, title: str = "", width: int = 8, decimals: int = 2,
                  symmetric: bool = False, sequential: bool = False) -> str:
    """A grid of numbers, coloured by magnitude.

    `values` may be V (n_states,) or Q (n_states, n_actions), in which case it
    shows max_a Q. Set `symmetric=True` for signed quantities like TD errors so
    that zero sits at the neutral colour, and `sequential=True` for counts.
    """
    rows, cols = _grid_shape(env)
    values = _to_state_values(values, env.n_states)
    scaled, low, high = _normalise(values, symmetric)
    palette = _sequential if sequential else _diverging

    lines = []
    if title:
        lines.append(f"{title}   [{low:+.3g} .. {high:+.3g}]")
    for r in range(rows):
        row = []
        for c in range(cols):
            if _blocked(env, (r, c)):
                row.append(_cell("#".center(width), (60, 60, 60)))
                continue
            s = env.index((r, c))
            row.append(_cell(f"{values[s]:{width}.{decimals}f}", palette(scaled[s])))
        lines.append("".join(row))
    return "\n".join(lines)


def policy_overlay(env, Q, title: str = "", width: int = 8, decimals: int = 2) -> str:
    """Value in colour, greedy action as an arrow. The two things you want at once.

    A value function that looks right with arrows that point the wrong way means
    the values are fine and the action indexing is not -- a swap of two entries in
    an action table is invisible in any learning curve and obvious here.
    """
    rows, cols = _grid_shape(env)
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2:
        raise ValueError(f"policy_overlay needs a Q table of shape (n_states, n_actions), got {Q.shape}")
    values = Q.max(axis=1)
    actions = Q.argmax(axis=1)
    scaled, low, high = _normalise(values, symmetric=False)

    lines = []
    if title:
        lines.append(f"{title}   [{low:+.3g} .. {high:+.3g}]")
    for r in range(rows):
        row = []
        for c in range(cols):
            if _blocked(env, (r, c)):
                row.append(_cell("#".center(width), (60, 60, 60)))
                continue
            s = env.index((r, c))
            # Trailing space so adjacent cells do not run into each other -- the
            # arrow sits at column 0 and would otherwise touch the number before it.
            label = f"{ARROWS[int(actions[s])]}{values[s]:{width - 2}.{decimals}f} "
            row.append(_cell(label, _diverging(scaled[s])))
        lines.append("".join(row))
    return "\n".join(lines)


def compare_grids(env, learned, truth, gamma_label: str = "", width: int = 8,
                  max_width: int | None = None) -> str:
    """Learned, exact, and the error between them, side by side.

    This is the function that makes the ground-truth guarantee pay off. The third
    panel is signed and symmetric, so a systematic bias shows as one colour across
    the whole grid -- which is what a missing terminal mask or a wrong discount
    looks like, and it is unmistakable once you have seen it once.
    """
    learned = _to_state_values(learned, env.n_states)
    truth = _to_state_values(truth, env.n_states)
    error = learned - truth

    panels = [
        value_heatmap(env, learned, f"learned {gamma_label}".strip(), width=width),
        value_heatmap(env, truth, f"exact {gamma_label}".strip(), width=width),
        value_heatmap(env, error, "error (learned - exact)", width=width, symmetric=True),
    ]
    # Three panels side by side is the readable layout, but only if they fit.
    # A wrapped grid is worse than a stacked one.
    if max_width is None:
        max_width = _terminal_width()
    rows, cols = _grid_shape(env)
    if 3 * cols * width + 6 <= max_width:
        joined = _side_by_side(panels, gap=3)
    else:
        joined = "\n\n".join(panels)

    max_error = float(np.abs(error).max())
    mean_error = float(np.abs(error).mean())
    signed = "all one sign" if (error >= -1e-12).all() or (error <= 1e-12).all() else "mixed signs"
    joined += f"\n\nmax |error| {max_error:.4f}   mean |error| {mean_error:.4f}   ({signed})"
    return joined


def _terminal_width(default: int = 100) -> int:
    try:
        return max(os.get_terminal_size().columns, 40)
    except OSError:
        return default


def visit_heatmap(env, counts, title: str = "state visits") -> str:
    """Where the agent actually went.

    Half of "it is not learning" is "it never went there". Print this before
    tuning anything: a value function is only as good as the states that fed it.
    """
    return value_heatmap(env, counts, title, decimals=0, sequential=True)


def trajectory(env, states, title: str = "trajectory", width: int = 4) -> str:
    """Draw a path through the grid, numbered by visit order."""
    rows, cols = _grid_shape(env)
    order: dict[int, int] = {}
    for step, s in enumerate(states):
        order.setdefault(int(s), step)
    longest = max(len(states) - 1, 1)

    lines = [title] if title else []
    for r in range(rows):
        row = []
        for c in range(cols):
            if _blocked(env, (r, c)):
                row.append(_cell("#".center(width), (60, 60, 60)))
                continue
            s = env.index((r, c))
            if s in order:
                row.append(_cell(f"{order[s]:>{width}}", _sequential(order[s] / longest)))
            else:
                row.append(_cell(".".center(width), (30, 30, 30)))
        lines.append("".join(row))
    return "\n".join(lines)


def sparkline(values, width: int = 60) -> str:
    """A learning curve on one line. Good enough to spot a collapse in a log."""
    blocks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return ""
    if values.size > width:
        edges = np.linspace(0, values.size, width + 1).astype(int)
        values = np.array([values[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    low, high = values.min(), values.max()
    if high - low < 1e-12:
        return blocks[0] * len(values) + f"  (flat at {low:.3g})"
    scaled = (values - low) / (high - low)
    line = "".join(blocks[int(t * (len(blocks) - 1))] for t in scaled)
    return f"{line}  [{low:.3g} .. {high:.3g}]"


def legend() -> str:
    if not colour_enabled():
        return "(colour disabled by NO_COLOR)"
    bar = "".join(_cell(" ", _diverging(t / 20)) for t in range(21))
    seq = "".join(_cell(" ", _sequential(t / 20)) for t in range(21))
    return f"diverging (values, errors)  low {bar} high\nsequential (counts)         low {seq} high"


def _side_by_side(panels, gap: int = 2) -> str:
    """Join multi-line strings horizontally, padding by VISIBLE width."""
    split = [p.split("\n") for p in panels]
    height = max(len(p) for p in split)
    widths = [max((_visible_width(line) for line in p), default=0) for p in split]

    lines = []
    for row in range(height):
        parts = []
        for panel, panel_width in zip(split, widths):
            line = panel[row] if row < len(panel) else ""
            parts.append(line + " " * (panel_width - _visible_width(line)))
        lines.append((" " * gap).join(parts))
    return "\n".join(lines)


def _visible_width(line: str) -> int:
    """Length ignoring ANSI escape sequences, which occupy no columns."""
    width, index = 0, 0
    while index < len(line):
        if line[index] == "\033":
            end = line.find("m", index)
            index = len(line) if end == -1 else end + 1
            continue
        width += 1
        index += 1
    return width
