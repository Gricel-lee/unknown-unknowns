"""Identification by Shapley values (SHAP: SHapley Additive exPlanations).

The cost of a sample i is predicted from a set S of context variables (together with the known
variables K) by the mean cost of the logged samples that match it on K and S:

    v_i(S) = mean cost of the samples with the same values as i on K and S

For the candidate variables C (the players), the Shapley value of candidate j for sample i is
its average marginal contribution over all orders in which the candidates can be added:

    phi_j(i) = sum_{S subset of C without j} |S|! (m - |S| - 1)! / m! * (v_i(S + j) - v_i(S))

with m = |C|. The values add up to v_i(C) - v_i({}), the part of the cost explained by the
candidates beyond K, and the credit of variables that act together is shared fairly. With
binary variables and a few candidates, all subsets can be enumerated, so the values are exact
(no sampling and no fitted model: the logs are the model).

A candidate's importance is the mean |phi_j(i)| over the samples, in the units of the cost. It
is biased upward (the means of small groups follow noise), so its significance is tested with
a permutation test: shuffling the candidate's values within the bins of K, which removes its
association with the cost.
"""
import itertools
import math

import numpy as np

from unknown_unknowns.data_generator import output_value
from unknown_unknowns.identifier.base import Identifier


def _columns(samples, names):
    return {n: np.array([context[n] for context, _ in samples]) for n in names}


def _group_means(costs, columns):
    """For each sample, the mean cost of the samples with the same values in ``columns``."""
    if not columns:
        return np.full(len(costs), costs.mean())
    keys = np.stack(columns, axis=1)
    _, codes = np.unique(keys, axis=0, return_inverse=True)
    codes = codes.ravel()
    sums = np.bincount(codes, weights=costs)
    counts = np.bincount(codes)
    return (sums / counts)[codes]


def shapley_values(costs, known_columns, candidate_columns):
    """Exact Shapley values ``{candidate: array over samples}`` (see the module docstring)."""
    names = list(candidate_columns)
    m = len(names)
    value = {}
    for size in range(m + 1):
        for subset in itertools.combinations(names, size):
            value[subset] = _group_means(
                costs, list(known_columns) + [candidate_columns[n] for n in subset])
    phi = {}
    for j in names:
        others = [n for n in names if n != j]
        total = np.zeros(len(costs))
        for size in range(m):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for subset in itertools.combinations(others, size):
                with_j = tuple(n for n in names if n in subset or n == j)
                total += weight * (value[with_j] - value[subset])
        phi[j] = total
    return phi


def _shuffle_within(values, strata, rng):
    by_stratum = np.argsort(strata, kind="stable")
    shuffled = np.lexsort((rng.random(len(strata)), strata))
    out = values.copy()
    out[by_stratum] = values[shuffled]
    return out


def _strata(known_columns, n):
    if not known_columns:
        return np.zeros(n, dtype=int)
    _, codes = np.unique(np.stack(known_columns, axis=1), axis=0, return_inverse=True)
    return codes.ravel()


def importance(samples, known, candidates, outputs=(None,), permutations=200, seed=0):
    """For each candidate: its importance (mean |Shapley value|, the largest over the outputs),
    the output it comes from, the importance per output, and its permutation p-value.

    Returns ``{candidate: {"importance", "best_output", "per_output", "p_value",
    "residual"}}``, where ``residual`` is the mean absolute deviation of the best output's cost
    from its mean within the bins of the known variables (the variation left to explain).
    """
    rng = np.random.default_rng(seed)
    known_cols = _columns(samples, known)
    cand_cols = _columns(samples, candidates)
    costs = {o: np.array([output_value(v, o) for _, v in samples], dtype=float) for o in outputs}
    strata = _strata(list(known_cols.values()), len(samples))

    def per_output(columns):
        return {o: {j: float(np.mean(np.abs(phi)))
                    for j, phi in shapley_values(c, list(known_cols.values()), columns).items()}
                for o, c in costs.items()}

    observed = per_output(cand_cols)
    result = {}
    for j in candidates:
        scores = {o: observed[o][j] for o in outputs}
        best = max(scores, key=scores.get)
        null = []
        for _ in range(permutations):
            shuffled = dict(cand_cols, **{j: _shuffle_within(cand_cols[j], strata, rng)})
            null.append(max(per_output(shuffled)[o][j] for o in outputs))
        p_value = (1 + sum(x >= scores[best] for x in null)) / (1 + permutations)
        c = costs[best]
        residual = float(np.mean(np.abs(c - _group_means(c, list(known_cols.values())))))
        result[j] = {"importance": scores[best], "best_output": best, "per_output": scores,
                     "p_value": p_value, "residual": residual}
    return result


class ShapIdentifier(Identifier):
    """Add the candidate with the largest Shapley importance, among those whose importance is
    significant at level ``alpha / |C|`` and accounts for at least a fraction ``epsilon`` of the
    variation left within the bins of the known variables (importance / residual)."""
    name = "shap"

    def __init__(self, alpha=0.05, epsilon=0.05, permutations=200, seed=0):
        self.alpha, self.epsilon = alpha, epsilon
        self.permutations, self.seed = permutations, seed

    def identify(self, samples, known, candidates, outputs):
        scores = importance(samples, known, candidates, outputs, self.permutations, self.seed)
        ranking = []
        for j, s in scores.items():
            ratio = s["importance"] / s["residual"] if s["residual"] > 0 else 0.0
            significant = s["p_value"] < self.alpha / len(candidates)
            relevant = ratio >= self.epsilon
            ranking.append({"candidate": j, "score": s["importance"],
                            "best_output": s["best_output"], "p_value": s["p_value"],
                            "residual": s["residual"], "ratio": ratio,
                            "significant": significant, "relevant": relevant,
                            "accepted": significant and relevant})
        ranking.sort(key=lambda r: -r["score"])
        accepted = [r for r in ranking if r["accepted"]]
        return ([accepted[0]["candidate"]] if accepted else []), ranking
