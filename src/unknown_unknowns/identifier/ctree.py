"""Conditional inference trees (Hothorn, Hornik & Zeileis, 2006, "Unbiased recursive
partitioning: A conditional inference framework", JCGS) to identify relevant context variables.

The tree is grown recursively. In each node:

1. for every context variable X_j, test the null hypothesis that the costs (all outputs
   together, as a vector) are independent of X_j, with the conditional (permutation)
   framework of Strasser & Weber (1999): the linear statistic T_j = sum_i g(x_ij) h(y_i)^T,
   with g the one-hot coding of X_j and h(y) the vector of costs, standardised by its exact
   mean and covariance under permutation, gives a quadratic statistic c_j that is
   asymptotically chi-squared;
2. adjust the p-values for the number of variables tested (Bonferroni, 1 - (1 - p)^m);
3. stop if no adjusted p-value is below alpha (this is what keeps the tree from splitting on
   noise); otherwise split the node on the variable with the smallest p-value, one child per
   value, and repeat in every child.

Testing at alpha in every node, as in the original method, lets deep nodes split on noise once
the tree has many nodes; that is harmless for prediction but makes irrelevant variables look
relevant. By default (``adjust_depth=True``) alpha is therefore divided by the number of nodes
that can be tested at the node's depth (the product of the numbers of children of its
ancestors, 2^depth for binary variables).

The variables used anywhere in the tree are the relevant ones. A variable can be used in some
branches only (e.g. ``busy`` may matter only when it is wet).
"""
import math

import numpy as np

from unknown_unknowns.data_generator import output_value
from unknown_unknowns.identifier.base import Identifier


# --- chi-squared survival function (numpy has none) -----------------------------------------

def _gammaincc(a, x):
    """Regularised upper incomplete gamma function Q(a, x) (Numerical Recipes, gammq)."""
    if x <= 0:
        return 1.0
    if x < a + 1:   # series for P(a, x), then Q = 1 - P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    # continued fraction for Q(a, x) (modified Lentz)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def chi2_sf(x, df):
    """P(chi^2_df > x)."""
    return _gammaincc(df / 2.0, x / 2.0) if df > 0 else 1.0


# --- the independence test ---------------------------------------------------------------------

def independence_test(x, y):
    """Conditional inference test of independence between a categorical ``x`` (n values) and
    a numeric response ``y`` (n x q). Returns ``(statistic, p_value)`` (quadratic form,
    chi-squared with the rank of the covariance as degrees of freedom)."""
    y = np.asarray(y, dtype=float).reshape(len(x), -1)
    n = len(x)
    levels, codes = np.unique(np.asarray(x), return_inverse=True)
    if len(levels) < 2 or n < 2:
        return 0.0, 1.0
    g = np.eye(len(levels))[codes]                       # n x k one-hot coding
    t = (g.T @ y).ravel(order="F")                       # vec of the k x q linear statistic
    s = g.sum(axis=0)                                    # count per level
    mu_h = y.mean(axis=0)
    sigma_h = np.cov(y, rowvar=False, bias=True).reshape(y.shape[1], y.shape[1])
    mean = np.kron(mu_h, s)
    cov = (n / (n - 1)) * np.kron(sigma_h, np.diag(s)) - (1 / (n - 1)) * np.kron(sigma_h, np.outer(s, s))
    rank = np.linalg.matrix_rank(cov, tol=1e-10 * max(1.0, np.abs(cov).max()))
    if rank == 0:
        return 0.0, 1.0
    diff = t - mean
    statistic = float(diff @ np.linalg.pinv(cov, rcond=1e-10) @ diff)
    return statistic, chi2_sf(statistic, rank)


# --- the tree ------------------------------------------------------------------------------------

def _response(samples, outputs):
    return np.array([[output_value(value, o) for o in outputs] for _, value in samples])


def ctree(samples, variables, outputs=(None,), alpha=0.05, min_split=20, min_bucket=7,
          max_depth=None, adjust_depth=True, _depth=0, _width=1):
    """Grow a conditional inference tree of the costs on the context ``variables`` (names).

    A node is split only if it has at least ``min_split`` samples, some variable's adjusted
    p-value is below ``alpha`` (divided by the number of nodes at its depth if
    ``adjust_depth``), and every child would have at least ``min_bucket`` samples.
    Returns the root node: ``{"n", "mean": {output: mean}, "tests": {variable: (statistic,
    p_value)}, "split": variable or None, "p_value", "children": {value: node}}``.
    """
    y = _response(samples, outputs)
    node = {"n": len(samples), "mean": dict(zip(outputs, y.mean(axis=0))) if len(samples) else {},
            "tests": {}, "split": None, "p_value": None, "children": {}}
    if len(samples) < min_split or not variables or (max_depth is not None and _depth >= max_depth):
        return node
    for v in variables:
        node["tests"][v] = independence_test([context[v] for context, _ in samples], y)
    m = len(variables)
    adjusted = {v: 1 - (1 - p) ** m for v, (_, p) in node["tests"].items()}
    best = min(adjusted, key=adjusted.get)
    if adjusted[best] >= (alpha / _width if adjust_depth else alpha):
        return node
    groups = {}
    for context, value in samples:
        groups.setdefault(context[best], []).append((context, value))
    if len(groups) < 2 or min(len(g) for g in groups.values()) < min_bucket:
        return node
    node["split"], node["p_value"] = best, adjusted[best]
    rest = [v for v in variables if v != best]
    node["children"] = {value: ctree(group, rest, outputs, alpha, min_split, min_bucket,
                                     max_depth, adjust_depth, _depth + 1, _width * len(groups))
                        for value, group in sorted(groups.items())}
    return node


def variables_used(tree):
    """The variables the tree splits on, in breadth-first order (most important first)."""
    used, queue = [], [tree]
    while queue:
        node = queue.pop(0)
        if node["split"] is not None:
            if node["split"] not in used:
                used.append(node["split"])
            queue += list(node["children"].values())
    return used


def format_tree(tree, indent=""):
    """A text drawing of the tree, e.g. for printing or logging."""
    means = ", ".join(f"{o}={m:.2f}" for o, m in tree["mean"].items())
    if tree["split"] is None:
        return f"{indent}leaf: n={tree['n']}, {means}\n"
    text = f"{indent}split on {tree['split']} (p={tree['p_value']:.2g}), n={tree['n']}\n"
    for value, child in tree["children"].items():
        text += f"{indent}  {tree['split']}={value}:\n" + format_tree(child, indent + "    ")
    return text


class CtreeIdentifier(Identifier):
    """Grow a conditional inference tree of the costs on the known and candidate variables;
    the candidates it splits on are added, in breadth-first order (most important first)."""
    name = "ctree"

    def __init__(self, alpha=0.01, min_split=20, min_bucket=7, adjust_depth=True):
        self.alpha, self.min_split, self.min_bucket = alpha, min_split, min_bucket
        self.adjust_depth = adjust_depth
        self.tree = None   # the last tree grown

    def identify(self, samples, known, candidates, outputs):
        self.tree = ctree(samples, list(known) + list(candidates), outputs, self.alpha,
                          self.min_split, self.min_bucket, adjust_depth=self.adjust_depth)
        used = variables_used(self.tree)
        selected = [v for v in used if v in candidates]
        rows = [{"candidate": v, "score": stat, "p_value": p, "accepted": v in selected}
                for v, (stat, p) in self.tree["tests"].items() if v in candidates]
        rows.sort(key=lambda r: r["p_value"])
        return selected, rows
