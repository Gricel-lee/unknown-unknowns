"""Ridgeline plot of sampled environment values, one ridge per context combination."""
import matplotlib.pyplot as plt
import numpy as np

from unknown_unknowns.data_generator.generator import output_value


def _kde(values, grid):
    """Gaussian kernel density estimate (Scott's rule bandwidth)."""
    values = np.asarray(values)
    bw = 1.06 * values.std() * len(values) ** (-1 / 5) or 0.1
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def _group(samples, names, output=None):
    """Split ``[({name: value}, x), ...]`` into ``{combination: [x, ...]}``, sorted by combination.

    With several values per sample, ``output`` picks which one (e.g. ``"path1"``)."""
    groups = {}
    for context, x in samples:
        groups.setdefault(tuple(context[n] for n in names), []).append(output_value(x, output))
    return dict(sorted(groups.items()))


def _new_axes(ax, groups):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 * len(groups) + 1.5))
    return ax


def _style(ax, groups, names, xlim, overlap, xlabel):
    """Row labels, baselines, limits and a recessive grid shared by both plot styles."""
    for y0 in range(len(groups)):
        ax.axhline(y0, color="#d0d0cc", lw=0.8, zorder=-1)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(
        [", ".join(f"{n}={v}" for n, v in zip(names, c)) + f"  (n={len(xs)})"
         for c, xs in reversed(groups.items())])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.2, len(groups) - 1 + overlap + 0.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e5e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return ax


def ridgeline(samples, names, overlap=1.8, color="#7faee6", ax=None, output=None,
              xlabel="environment value"):
    """Plot ``[({name: value}, x), ...]`` as one density ridge per combination of ``names``.

    Rows are ordered by combination, first one at the top. With several values per sample,
    ``output`` picks which one is plotted. Returns the matplotlib Axes.
    """
    groups = _group(samples, names, output)
    xs = np.concatenate(list(groups.values()))
    pad = 0.1 * (xs.max() - xs.min())
    grid = np.linspace(xs.min() - pad, xs.max() + pad, 400)
    densities = [_kde(g, grid) for g in groups.values()]
    scale = overlap / max(d.max() for d in densities)

    ax = _new_axes(ax, groups)
    for row, d in enumerate(densities):
        y0 = len(groups) - 1 - row          # first combination on top
        y = np.where(d > 0.005 * d.max(), y0 + d * scale, np.nan)   # trim flat tails
        ax.fill_between(grid, y0, y, color=color, lw=0, zorder=row)
        ax.plot(grid, y, color="white", lw=1.5, zorder=row)
    return _style(ax, groups, names, (grid[0], grid[-1]), overlap, xlabel)


def ridgeline_hist(samples, names, bins=30, overlap=0.9, color="#7faee6", ax=None,
                   output=None, xlabel="environment value"):
    """Like :func:`ridgeline`, but each ridge is a histogram (bars) instead of a smooth curve.

    All rows share the same ``bins`` bin edges, so bars line up between rows.
    """
    groups = _group(samples, names, output)
    edges = np.histogram_bin_edges(np.concatenate(list(groups.values())), bins=bins)
    heights = [np.histogram(g, bins=edges, density=True)[0] for g in groups.values()]
    scale = overlap / max(h.max() for h in heights)

    ax = _new_axes(ax, groups)
    for row, h in enumerate(heights):
        y0 = len(groups) - 1 - row          # first combination on top
        keep = h > 0
        ax.bar(edges[:-1][keep], h[keep] * scale, width=np.diff(edges)[keep], bottom=y0,
               align="edge", color=color, edgecolor="white", lw=0.8, zorder=row)
    return _style(ax, groups, names, (edges[0], edges[-1]), overlap, xlabel)


def plot_single_distribution(samples, bins=30, color="#7faee6", ax=None, output=None,
                             xlabel="environment value"):
    """Histogram of all environment values in ``[({name: value}, x), ...]``, ignoring the
    context variables (``output`` picks the value, as in :func:`ridgeline`).
    Returns the matplotlib Axes."""
    xs = np.array([output_value(x, output) for _, x in samples])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(xs, bins=bins, color=color, edgecolor="white", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of samples")
    ax.set_title(f"All samples (n={len(xs)})", loc="left")
    ax.grid(axis="y", color="#e5e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax



if __name__ == "__main__":
    # Demo: plot the robot example (examples/robot/context_data/ at the repository root).
    from pathlib import Path

    from unknown_unknowns.data_generator import DataGenerator, Variable

    DATA = Path(__file__).resolve().parents[3] / "examples" / "robot" / "context_data"
    gen = DataGenerator(
        variables=[Variable("light", [0, 1]), Variable("wet", [0, 1]), Variable("busy", [0, 1])],
        prior=DATA / "context_prior.txt",
        distributions=DATA / "context_distributions.txt",
        seed=0,
    )
    samples = gen.sample_many(1000)

    # Each sample has a travel cost per path; plot path 1, split by every variable
    # (gen.names; fewer variables also work), then all samples together.
    ridgeline_hist(samples, gen.names, output="path1", xlabel="cost travel path_1")
    plt.tight_layout()
    ridgeline(samples, gen.names, output="path1", xlabel="cost travel path_1")
    plt.tight_layout()
    plot_single_distribution(samples, output="path1", xlabel="cost travel path_1")
    plt.tight_layout()
    plt.show()
