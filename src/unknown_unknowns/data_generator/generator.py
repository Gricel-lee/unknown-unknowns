"""Generator of (context, environment value) samples.

A sample's value is one number, or a ``{name: number}`` dict (e.g. the travel cost of
``path1`` and ``path2``) when the distributions give one Normal per name.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .prior import load_distributions, load_prior


@dataclass(frozen=True)
class Variable:
    """A context variable and the values it can take, e.g. ``Variable("light", [0, 1])``."""
    name: str
    values: tuple

    def __post_init__(self):
        values = tuple(self.values)
        if not values:
            raise ValueError(f"variable {self.name!r} has no values")
        if len(set(values)) != len(values):
            raise ValueError(f"variable {self.name!r} has repeated values")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Normal:
    """Normal distribution of the environment value for one context combination."""
    mean: float
    std: float

    def __post_init__(self):
        if self.std < 0:
            raise ValueError(f"std must be non-negative, got {self.std}")

    def sample(self, rng, size=None):
        return rng.normal(self.mean, self.std, size)


def _as_normal(dist):
    return dist if isinstance(dist, Normal) else Normal(**dist)


def _as_distribution(dist):
    """A Normal (one value per sample) or ``{name: Normal}`` (one value per name)."""
    if isinstance(dist, dict) and set(dist) != {"mean", "std"}:
        return {name: _as_normal(d) for name, d in dist.items()}
    return _as_normal(dist)


def output_value(value, output=None):
    """The ``output`` value (e.g. ``"path1"``) of one sample's value; the value itself if None."""
    if output is None:
        if isinstance(value, dict):
            raise ValueError(f"samples have several values {list(value)}; pick one with output=")
        return value
    return value[output]


class DataGenerator:
    """Samples a context combination from a prior, then a value from that combination's Normal
    (or one value per name, e.g. per path, from that combination's Normals).

    Parameters
    ----------
    variables : list of Variable
        Context variables, in the order used in the prior file and in combination tuples.
    prior : str or Path or dict
        Path to the txt file with the probability of each combination (see
        ``prior.py`` for the format), or an already-built ``{combination: probability}`` dict.
    distributions : str or Path or dict, optional
        Path to a txt file with the Normal(s) of each combination (see ``prior.py``), or
        ``{combination tuple: Normal or {"mean": .., "std": ..} or {name: Normal, ...}}``.
        More can be added later with :meth:`set_distribution`.
    seed : int or numpy.random.Generator, optional
        Seed for reproducible sampling.
    """

    def __init__(self, variables, prior, distributions=None, seed=None):
        self.variables = list(variables)
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"variable names must be unique, got {names}")
        self.prior = dict(prior) if isinstance(prior, dict) else load_prior(prior, self.variables)
        self.distributions = {}
        if isinstance(distributions, (str, Path)):
            distributions = load_distributions(distributions, self.variables)
        for combo, dist in (distributions or {}).items():
            self.set_distribution(combo, dist)
        self.rng = np.random.default_rng(seed)

    @property
    def names(self):
        return [v.name for v in self.variables]

    @property
    def outputs(self):
        """Names of the values in each sample (e.g. ``["path1", "path2"]``), or None when
        each sample has a single value."""
        kinds = {tuple(d) if isinstance(d, dict) else None for d in self.distributions.values()}
        if len(kinds) > 1:
            raise ValueError(f"combinations have different outputs: {kinds}")
        kind = kinds.pop() if kinds else None
        return list(kind) if kind else None

    def _combo(self, combo):
        """Normalise a combination given as a tuple or as a ``{name: value}`` dict."""
        if isinstance(combo, dict):
            if set(combo) != set(self.names):
                raise ValueError(f"combination must give a value for each of {self.names}")
            combo = tuple(combo[n] for n in self.names)
        combo = tuple(combo)
        if len(combo) != len(self.variables):
            raise ValueError(f"combination {combo} must have {len(self.variables)} values")
        for value, var in zip(combo, self.variables):
            if value not in var.values:
                raise ValueError(f"{value!r} is not a value of {var.name!r}")
        return combo

    def set_distribution(self, combo, dist=None, *, mean=None, std=None):
        """Set the Normal (or one Normal per name) for one combination.

        ``gen.set_distribution((1, 0, 0), mean=5, std=1)``,
        ``gen.set_distribution({"light": 1, "wet": 0, "busy": 0}, {"mean": 5, "std": 1})``,
        ``gen.set_distribution((1, 0, 0), {"path1": {"mean": 5, "std": 1},
        "path2": {"mean": 7, "std": 1}})``.
        """
        dist = Normal(mean, std) if dist is None else _as_distribution(dist)
        self.distributions[self._combo(combo)] = dist

    def load_distributions(self, path):
        """Set the Normals listed in a txt file (see ``prior.py``).

        Combinations in the file replace any already set; others are kept.
        """
        for combo, dist in load_distributions(path, self.variables).items():
            self.set_distribution(combo, dist)

    def check(self):
        """Raise if a combination with non-zero probability has no distribution, or if
        combinations have different outputs."""
        self.outputs
        missing = [c for c, p in self.prior.items() if p > 0 and c not in self.distributions]
        if missing:
            raise ValueError(f"no distribution set for combinations with non-zero "
                             f"probability: {missing}")

    def sample(self):
        """Return one ``({name: value}, environment value)`` pair."""
        return self.sample_many(1)[0]

    def sample_many(self, n):
        """Return a list of ``n`` ``({name: value}, environment value)`` pairs."""
        self.check()
        combos = list(self.prior)
        probs = np.array([self.prior[c] for c in combos])
        idx = self.rng.choice(len(combos), size=n, p=probs / probs.sum())
        out = []
        for i in idx:
            combo = combos[i]
            dist = self.distributions[combo]
            if isinstance(dist, dict):
                value = {name: float(d.sample(self.rng)) for name, d in dist.items()}
            else:
                value = float(dist.sample(self.rng))
            out.append((dict(zip(self.names, combo)), value))
        return out

    def _plot_by(self, plot_by):
        """Variable names to split the plot by: all of them, ``"light"``, ``"light,wet"``
        or ``("light", "wet")``."""
        if plot_by is None:
            return self.names
        if isinstance(plot_by, str):
            plot_by = plot_by.split(",")
        plot_by = [name.strip() for name in plot_by]
        unknown = [name for name in plot_by if name not in self.names]
        if unknown or not plot_by:
            raise ValueError(f"plot_by must name some of {self.names}, got {plot_by}")
        return plot_by

    def plot_ridgeline(self, n=1000, samples=None, plot_by=None, **kwargs):
        """Sample ``n`` points (or use ``samples``) and plot one smooth density ridge per
        combination of the ``plot_by`` variables (default: all of them). With several
        values per sample, choose one with ``output=`` (e.g. ``"path1"``).

        Extra arguments go to :func:`plot_ridgeline.ridgeline`. Returns the Axes.
        """
        from .plot_ridgeline import ridgeline   # matplotlib is only needed for plotting
        samples = self.sample_many(n) if samples is None else samples
        return ridgeline(samples, self._plot_by(plot_by), **kwargs)

    def plot_ridgeline_hist(self, n=1000, samples=None, plot_by=None, **kwargs):
        """Like :meth:`plot_ridgeline`, but with histogram bars instead of smooth curves.

        Extra arguments go to :func:`plot_ridgeline.ridgeline_hist`. Returns the Axes.
        """
        from .plot_ridgeline import ridgeline_hist
        samples = self.sample_many(n) if samples is None else samples
        return ridgeline_hist(samples, self._plot_by(plot_by), **kwargs)

    def plot_single_distribution(self, n=1000, samples=None, **kwargs):
        """Sample ``n`` points (or use ``samples``) and plot one histogram of all values,
        ignoring the context. Extra arguments go to
        :func:`plot_ridgeline.plot_single_distribution`. Returns the Axes."""
        from .plot_ridgeline import plot_single_distribution
        return plot_single_distribution(self.sample_many(n) if samples is None else samples,
                                        **kwargs)

    def __call__(self):
        return self.sample()
