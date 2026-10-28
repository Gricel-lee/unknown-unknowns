"""Identification by unimodality: try combinations of the unknown variables, one per call, and
accept one when it makes every bin of the costs unimodal and each of its variables has a
significant effect on the costs."""
import itertools
import math

from unknown_unknowns.data_generator import Variable
from unknown_unknowns.detector import all_tested, all_unimodal, check_partition, select
from unknown_unknowns.identifier.base import Identifier


def has_effect(samples, variable, given, alpha, outputs, min_bin_size=10):
    """Whether ``variable`` has a significant effect on the costs, given the variables
    ``given``: in some bin of ``given`` and for some output, the mean cost differs between
    the values of ``variable`` (Welch z-test, Bonferroni-corrected over bins, outputs and
    values). Groups with fewer than ``min_bin_size`` samples are skipped."""
    names = [v.name for v in given]
    p_values = []
    for combo in itertools.product(*(v.values for v in given)):
        where = dict(zip(names, combo))
        for output in outputs:
            groups = [select(samples, {**where, variable.name: value}, output)
                      for value in variable.values]
            base = groups[0]
            for other in groups[1:]:
                if min(len(base), len(other)) < min_bin_size:
                    continue
                se = math.sqrt(base.var(ddof=1) / len(base) + other.var(ddof=1) / len(other))
                z = abs(other.mean() - base.mean()) / se if se > 0 else math.inf
                p_values.append(math.erfc(z / math.sqrt(2)))
    return bool(p_values) and min(p_values) * len(p_values) < alpha


def candidates(unknown):
    """Combinations of the unknown variables to try: single variables first, then pairs, ...
    e.g.: ``[("light",), ("wet",), ("busy",), ("light", "wet"), ...]``."""
    for k in range(1, len(unknown) + 1):
        yield from itertools.combinations(unknown, k)


def identification_step(samples, results, variables, known, unknown, tried, alpha, outputs):
    """From identified unknowns to knows:
    Look for the variables that explain an anomaly (call only after anomaly detection).

    ``results`` are the anomaly-detection results of the current split, i.e.
    ``check_partition(samples, known, ...)``, in which some bin is not unimodal. This step tries
    the next combination of ``unknown`` variables (added to the known ones), one per call. The
    combination is

    - accepted if every bin of the enlarged split has enough samples to be tested and is
      unimodal, and every variable of the combination has a significant effect on the costs
      given the others (:func:`has_effect`; this rejects distractors whose bins only look
      unimodal for lack of data). Then every previously known variable that no longer has an
      effect given the others (e.g. a proxy of the variable just accepted) is pruned;
    - rejected, and recorded in ``tried`` (``{names: number of samples}``), if some bin is still
      not unimodal or some variable has no effect. A rejected combination is tried again once
      the number of samples has doubled (a rejection may be a false alarm of the tests);
    - left for a later call, with more samples, if some bin is too small to test.

    Returns ``(known, unknown, results, event)``: the updated variable lists, the test results
    of the (possibly new) split, and ``("added", names, pruned)``, ``("tried", names)``,
    ``("waiting", names)`` or ``None`` (every combination was rejected recently).
    """
    # for every combination of unknowns (not in known), try it if it hasn't been rejected recently
    for candidate in candidates(unknown):
        trial = [v for v in variables if v in known or v in candidate]
        names = tuple(v.name for v in trial)
        # if the combination was rejected recently, skip it until more samples are available
        if names in tried and len(samples) < 2 * tried[names]:
            continue
        trial_results = check_partition(samples, trial, alpha, outputs)
        if not all_unimodal(trial_results):
            tried[names] = len(samples)
            return known, unknown, results, ("tried", list(names))
        if not all_tested(trial_results):
            return known, unknown, results, ("waiting", list(names))
        if not all(has_effect(samples, v, [u for u in trial if u is not v], alpha, outputs)
                   for v in candidate):
            tried[names] = len(samples)
            return known, unknown, results, ("tried", list(names))
        pruned = [v for v in trial if v not in candidate
                  and not has_effect(samples, v, [u for u in trial if u is not v], alpha, outputs)]
        known = [v for v in trial if v not in pruned]
        unknown = [v for v in variables if v not in known]
        if pruned:
            results = check_partition(samples, known, alpha, outputs)
        else:
            results = trial_results
        return known, unknown, results, ("added", [v.name for v in candidate],
                                         [v.name for v in pruned])
    return known, unknown, results, None


class UnimodalityIdentifier(Identifier):
    """Try combinations of the candidates (single variables first, then pairs, ...) until one
    makes every bin of the costs unimodal and each of its variables has a significant effect
    (see :func:`identification_step`); that combination is added. Rejected combinations are
    remembered across calls and tried again once the logs have doubled in size."""
    name = "unimodality"

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.tried = {}

    def identify(self, samples, known, candidates, outputs):
        names = list(known) + list(candidates)
        variables = [Variable(n, sorted({context[n] for context, _ in samples})) for n in names]
        by_name = {v.name: v for v in variables}
        known_vars = [by_name[n] for n in known]
        unknown = [by_name[n] for n in candidates]
        results = check_partition(samples, known_vars, self.alpha, outputs)
        rows = []
        while True:
            new_known, unknown, results, event = identification_step(
                samples, results, variables, known_vars, unknown, self.tried, self.alpha, outputs)
            if event is None:
                return [], rows
            kind, combination = event[0], event[1]
            added = [n for n in combination if n not in known]
            rows.append({"candidate": " ".join(added), "score": None, "outcome": kind,
                         "accepted": kind == "added"})
            if kind == "added":
                return added, rows
            if kind == "waiting":
                return [], rows
