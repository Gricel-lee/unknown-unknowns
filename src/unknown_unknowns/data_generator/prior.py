"""Reading the prior over context combinations, and their Normals, from txt files.

File format: one ``combination = probability`` entry per line. Blank lines and
lines starting with ``#`` are ignored. A combination is written either as one
character per variable (``101``) or comma-separated (``1,0,1``), in the order
the variables were declared. The comma form is needed when a value has more
than one character (e.g. ``10,0,1``). Every combination must appear exactly
once and the probabilities must add up to 1::

    # light wet busy
    000 = 0.1
    001 = 0.1
    ...

The per-combination Normal file (:func:`load_distributions`) uses the same
combination keys with ``mean, std`` on the right::

    {000 = 30, 1.2}
    {001 = 34, 2.0}
"""
import itertools
import math
from pathlib import Path

SUM_TOL = 1e-6


def _read_entries(path, variables):
    """Yield ``(where, combination tuple, right-hand side)`` for each entry in ``path``.

    The right-hand side is the text after ``=``, or a ``{name: value text}`` dict for
    entries in the named form ``env:000, name:value, ...``.

    Checks that every combination is valid and appears at most once.
    """
    by_text = [{str(v): v for v in var.values} for var in variables]
    seen = set()
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line.startswith("{") and line.endswith("}"):
            line = line[1:-1].strip()
        if not line:
            continue
        where = f"{path}:{lineno}"
        if line.startswith("env:"):
            # named form: env:000, name:value, name:value, ...
            rhs = {}
            for item in line.split(","):
                if ":" not in item:
                    raise ValueError(f"{where}: expected 'name:value' items, got {item.strip()!r}")
                name, value = (s.strip() for s in item.split(":", 1))
                if name in rhs:
                    raise ValueError(f"{where}: {name!r} given twice")
                rhs[name] = value
            key = rhs.pop("env")
        elif "=" in line:
            key, rhs = (s.strip() for s in line.split("=", 1))
        else:
            raise ValueError(f"{where}: expected 'combination = ...' or 'env:...', got {raw!r}")
        parts = [p.strip() for p in key.split(",")] if "," in key else list(key)
        if len(parts) != len(variables):
            raise ValueError(f"{where}: {key!r} has {len(parts)} values, "
                             f"expected {len(variables)}")
        combo = []
        for text, var, lookup in zip(parts, variables, by_text):
            if text not in lookup:
                raise ValueError(f"{where}: {text!r} is not a value of {var.name!r} "
                                 f"(allowed: {list(var.values)})")
            combo.append(lookup[text])
        combo = tuple(combo)
        if combo in seen:
            raise ValueError(f"{where}: combination {key!r} listed twice")
        seen.add(combo)
        yield where, combo, rhs


def load_prior(path, variables):
    """Return ``{combination tuple: probability}`` read from ``path``."""
    prior = {}
    for where, combo, prob in _read_entries(path, variables):
        p = float(prob)
        if not 0 <= p <= 1:
            raise ValueError(f"{where}: probability {p} is outside [0, 1]")
        prior[combo] = p

    expected = set(itertools.product(*(var.values for var in variables)))
    missing = expected - prior.keys()
    if missing:
        raise ValueError(f"{path}: missing combinations {sorted(missing)}")
    total = sum(prior.values())
    if not math.isclose(total, 1.0, abs_tol=SUM_TOL):
        raise ValueError(f"{path}: probabilities add up to {total}, not 1")
    return prior


def _normal(where, mean, std):
    mean, std = float(mean), float(std)
    if std < 0:
        raise ValueError(f"{where}: std must be non-negative, got {std}")
    return {"mean": mean, "std": std}


def _named_normals(where, fields):
    """``{"path1_mean": "26", "path1_std": "1", ...}`` -> ``{"path1": {"mean": 26, "std": 1}, ...}``."""
    outputs = []
    for name in fields:
        output, _, stat = name.rpartition("_")
        if stat not in ("mean", "std") or not output:
            raise ValueError(f"{where}: expected '<name>_mean' or '<name>_std', got {name!r}")
        if output not in outputs:
            outputs.append(output)
    missing = [f"{o}_{stat}" for o in outputs for stat in ("mean", "std") if f"{o}_{stat}" not in fields]
    if missing:
        raise ValueError(f"{where}: missing {missing}")
    return {o: _normal(where, fields[f"{o}_mean"], fields[f"{o}_std"]) for o in outputs}


def load_distributions(path, variables):
    """Return the Normal(s) of each combination read from ``path``.

    Same format as the prior file, with entries in one of two forms (braces optional):

    - ``{000 = 30, 1.2}``: one value per sample, ``{combination: {"mean": .., "std": ..}}``;
    - ``{env:000, path1_mean:26, path1_std:1.0, path2_mean:25, path2_std:1.0}``: one value
      per name (here ``path1`` and ``path2``), ``{combination: {name: {"mean": .., "std": ..}}}``.

    Combinations may be left out (set them later with ``DataGenerator.set_distribution``).
    """
    distributions = {}
    for where, combo, rhs in _read_entries(path, variables):
        if isinstance(rhs, dict):
            distributions[combo] = _named_normals(where, rhs)
            continue
        numbers = [s.strip() for s in rhs.split(",")]
        if len(numbers) != 2:
            raise ValueError(f"{where}: expected 'mean, std', got {rhs!r}")
        distributions[combo] = _normal(where, *numbers)
    return distributions
