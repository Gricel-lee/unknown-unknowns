"""Detection of multimodality in the logged costs (Hartigan's dip test).

The costs are split into one bin per combination of the values of the known variables, and each
cost (output) in each bin is tested for unimodality. A multimodal bin means that some variable
the model does not know affects the costs.
"""
import itertools

import diptest
import numpy as np

from unknown_unknowns.data_generator import output_value


def select(samples, where=None, output=None):
    """Return the environment values of ``[({name: value}, x), ...]`` as an array.

    ``where`` keeps only samples whose context matches, e.g. ``{"light": 1, "busy": 0}``
    (variables not named are ignored). ``None`` keeps all samples. With several values
    per sample, ``output`` picks which one (e.g. ``"path1"``).
    """
    where = where or {}
    values = []
    for context, x in samples:
        unknown = set(where) - set(context)
        if unknown:
            raise ValueError(f"unknown variables {sorted(unknown)}, samples have {sorted(context)}")
        if all(context[name] == value for name, value in where.items()):
            values.append(output_value(x, output))
    return np.array(values, dtype=float)


def dip_test(values, alpha=0.05):
    """Hartigan's dip test. Return ``(is_unimodal, p_value)``.

    The null hypothesis is unimodality, so the data counts as unimodal unless
    ``p_value < alpha``.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        raise ValueError(f"the dip test needs at least 4 values, got {len(values)}")
    _, p_value = diptest.diptest(values)
    return bool(p_value >= alpha), float(p_value)


def detect(samples, where=None, test=dip_test, output=None, **kwargs):
    """Select the ``output`` values matching ``where`` (see :func:`select`) and run ``test``
    on them.

    ``test`` is any function ``values -> (is_unimodal, p_value)``; extra arguments
    (e.g. ``alpha``) are passed to it.
    """
    return test(select(samples, where, output), **kwargs)


def check_partition(samples, variables, alpha, outputs, min_bin_size=10):
    """Split the samples into one bin per combination of ``variables``' values and dip-test
    each output (path) in each bin.

    Returns ``[(bin, output, unimodal, p_value), ...]`` with ``bin`` like
    ``{"light": 0, "wet": 1}`` (``{}`` = all samples when no variables are known). Bins with
    fewer than ``min_bin_size`` samples get ``(bin, output, None, None)``. The tests use a
    Bonferroni-corrected ``alpha``, so testing many bins and paths doesn't make false
    alarms likelier.
    """
    names = [v.name for v in variables]
    bins = [dict(zip(names, combo)) for combo in itertools.product(*(v.values for v in variables))]
    corrected = alpha / (len(bins) * len(outputs))
    results = []
    for where in bins:
        for output in outputs:
            if len(select(samples, where, output)) < min_bin_size:
                results.append((where, output, None, None))
            else:
                results.append((where, output,
                                *detect(samples, where=where, output=output, alpha=corrected)))
    return results


def all_unimodal(results):
    return all(unimodal is not False for _, _, unimodal, _ in results)


def all_tested(results):
    """True if every bin had enough samples (``min_bin_size``) to be tested."""
    return all(unimodal is not None for _, _, unimodal, _ in results)
