"""Identification by mutual information between context variables and the costs.

For a combination X of context variables (treated as one variable whose values are the joint
values, e.g. (light, wet) -> 00, 01, 10, 11) and a cost C:

    I(X; C) = sum_x p(x) * D_KL( P(C | X=x) || P(C) )

i.e. how far the cost distribution of each partition is from the overall one, weighted by how
often the partition occurs. The costs are continuous, so their distributions are estimated
with histograms over equal-frequency bins.

Raw MI cannot be used on its own: adding variables never lowers it, and the histogram estimate
is biased upward (a variable with no effect still gets MI > 0). Candidates are therefore scored
by their *conditional* MI given the known variables, I(new; C | known) (what they add beyond
them; a proxy of a known variable adds ~0), with a permutation-test p-value.

- :func:`rank_gains` and :func:`identify`: one candidate variable at a time, accepted if its
  gain is significant and removes at least a fraction epsilon of the remaining uncertainty
  H(C | known); the accepted candidate with the highest gain is added.
- :func:`rank_candidates` and :func:`select_variables`: combinations of candidates, greedy
  forward selection of the smallest significant combination.
"""
import itertools

import numpy as np

from unknown_unknowns.data_generator import output_value
from unknown_unknowns.identifier.base import Identifier


def _codes(samples, names):
    """Joint value of the variables ``names`` in each sample, as integers 0..K-1 (0 if none)."""
    if not names:
        return np.zeros(len(samples), dtype=int), 1
    columns = [[context[n] for context, _ in samples] for n in names]
    joint = list(zip(*columns))
    values = {v: i for i, v in enumerate(sorted(set(joint)))}
    return np.array([values[v] for v in joint]), len(values)


def _cost_bins(samples, output, bins):
    """Cost of each sample, replaced by the index of its equal-frequency bin (0..bins-1)."""
    costs = np.array([output_value(x, output) for _, x in samples], dtype=float)
    edges = np.quantile(costs, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, costs, side="right")


def _mi(x, k, c, bins):
    """I(X; C) in nats, for codes x (0..k-1) and binned costs c (0..bins-1):
    sum_x p(x) * KL(P(C | x) || P(C))."""
    joint = np.bincount(x * bins + c, minlength=k * bins).reshape(k, bins).astype(float)
    n = joint.sum()
    p_x = joint.sum(axis=1) / n
    p_c = joint.sum(axis=0) / n
    mi = 0.0
    for i in range(k):
        if p_x[i] == 0:
            continue
        p_c_given_x = joint[i] / joint[i].sum()
        nz = p_c_given_x > 0
        mi += p_x[i] * np.sum(p_c_given_x[nz] * np.log(p_c_given_x[nz] / p_c[nz]))
    return float(mi)


def mutual_information(samples, variables, output=None, bins=10):
    """I(variables; cost) in nats, for a list of variable names (one combination)."""
    x, k = _codes(samples, list(variables))
    return _mi(x, k, _cost_bins(samples, output, bins), bins)


def conditional_mutual_information(samples, new, given=(), output=None, bins=10):
    """I(new; cost | given) = I(given + new; cost) - I(given; cost), in nats."""
    return (mutual_information(samples, list(given) + list(new), output, bins)
            - mutual_information(samples, list(given), output, bins))


def _shuffle_within(values, strata, rng):
    """``values`` with their order shuffled inside each stratum (keeps the strata intact)."""
    by_stratum = np.argsort(strata, kind="stable")
    shuffled = np.lexsort((rng.random(len(strata)), strata))
    out = values.copy()
    out[by_stratum] = values[shuffled]
    return out


def score(samples, new, given=(), outputs=(None,), bins=10, permutations=200, seed=0):
    """Conditional MI of the combination ``new`` given ``given``, computed for each output
    separately; the score is the best (largest) of them, with its permutation p-value. Returns
    ``(cmi, p_value, {output: cmi})``, where ``cmi`` is the maximum over the outputs.

    The null distribution comes from shuffling the joint value of ``new`` inside each bin of
    ``given`` (which keeps ``given``'s own information), ``permutations`` times, and taking the
    same maximum over the outputs for each shuffle; so choosing the best output does not make
    chance results likelier to pass."""
    rng = np.random.default_rng(seed)
    g, kg = _codes(samples, list(given))
    nw, kn = _codes(samples, list(new))
    costs = {o: _cost_bins(samples, o, bins) for o in outputs}

    def total(new_codes):
        joint = g * kn + new_codes
        per = {o: _mi(joint, kg * kn, c, bins) - _mi(g, kg, c, bins) for o, c in costs.items()}
        return max(per.values()), per

    observed, per_output = total(nw)
    null = [total(_shuffle_within(nw, g, rng))[0] for _ in range(permutations)]
    p_value = (1 + sum(v >= observed for v in null)) / (1 + permutations)
    return observed, p_value, per_output


def rank_candidates(samples, known, unknown, outputs=(None,), bins=10, permutations=200,
                    max_size=None, seed=0):
    """Score every combination of the ``unknown`` variables (names; single variables, pairs,
    ... up to ``max_size``) by its conditional MI given the ``known`` ones.

    Returns a list of dicts ``{"variables", "cmi", "p_value", "per_output", "best_output"}``,
    sorted by decreasing conditional MI (``cmi`` is the best output's, see :func:`score`). Keep the combinations with ``p_value < alpha``; among those,
    prefer the smallest, since a larger set only scores higher if its extra variables add
    information (check that with :func:`score` given the smaller set). Many combinations are
    tested, so correct ``alpha`` for their number (as :func:`select_variables` does).
    """
    known, unknown = list(known), list(unknown)
    ranking = []
    for size in range(1, (max_size or len(unknown)) + 1):
        for combo in itertools.combinations(unknown, size):
            cmi, p, per = score(samples, combo, known, outputs, bins, permutations, seed)
            ranking.append({"variables": list(combo), "cmi": cmi, "p_value": p,
                            "per_output": per, "best_output": max(per, key=per.get)})
    return sorted(ranking, key=lambda r: -r["cmi"])


def select_variables(samples, known, unknown, outputs=(None,), alpha=0.05, bins=10,
                     permutations=200, max_size=None, seed=0):
    """Greedy forward selection with :func:`rank_candidates`.

    Repeatedly add the smallest combination of unknown variables whose conditional MI given the
    variables selected so far is significant (the highest MI among combinations of that size),
    until no combination is significant. Many combinations are tested at once, so ``alpha`` is
    Bonferroni-corrected by their number; ``permutations`` must be large enough for the
    p-values to go below it (the smallest p-value is ``1 / (permutations + 1)``). Conditioning on what is
    already selected means that proxies of selected variables, and supersets that only add
    noise, are not significant. Returns ``(selected names, steps)``, where each step is a dict
    ``{"given": names conditioned on, "ranking": rank_candidates(...), "chosen": names or None}``.
    """
    known, unknown = list(known), list(unknown)
    selected, steps = [], []
    while unknown:
        given = known + selected
        ranking = rank_candidates(samples, given, unknown, outputs, bins, permutations, max_size,
                                  seed)
        significant = [r for r in ranking if r["p_value"] < alpha / len(ranking)]
        best = min(significant, key=lambda r: (len(r["variables"]), -r["cmi"]), default=None)
        steps.append({"given": given, "ranking": ranking,
                      "chosen": best["variables"] if best else None})
        if best is None:
            break
        selected += best["variables"]
        unknown = [v for v in unknown if v not in best["variables"]]
    return selected, steps


def conditional_entropy(samples, given=(), output=None, bins=10):
    """H(O | K) in nats: sum_k p(k) * H(P^k), the entropy of the (binned) cost left within
    each bin of the variables ``given``, with the cost binned as in :func:`mutual_information`."""
    k_codes, k = _codes(samples, list(given))
    c = _cost_bins(samples, output, bins)
    joint = np.bincount(k_codes * bins + c, minlength=k * bins).reshape(k, bins).astype(float)
    n = joint.sum()
    entropy = 0.0
    for row in joint:
        m = row.sum()
        if m == 0:
            continue
        p = row[row > 0] / m
        entropy -= (m / n) * float(np.sum(p * np.log(p)))
    return entropy


def rank_gains(samples, known, candidates, outputs=(None,), alpha=0.05, epsilon=0.05, bins=10,
               permutations=200, seed=0):
    """Rank the candidate variables ``candidates`` (names) by the information they add about
    the costs O given the known variables K:

        GAIN(D, W, K) = MI(O; W | K) = sum_k p(k) sum_w p(w | k) D_KL(P^{k,w} || P^k)

    computed for each output and keeping the largest (see :func:`score`). A candidate is
    accepted if (i) its gain is significant at level ``alpha / |C|`` (permutation test, |C| the
    number of candidates) and (ii) ``GAIN / H(O | K) >= epsilon``, i.e. it removes at least a
    fraction ``epsilon`` of the uncertainty left about the cost (for the output with the
    largest gain).

    Returns a list of dicts ``{"variable", "gain", "best_output", "p_value", "entropy",
    "ratio", "significant", "relevant", "accepted", "per_output"}``, by decreasing gain.
    """
    known, candidates = list(known), list(candidates)
    ranking = []
    for w in candidates:
        gain, p_value, per_output = score(samples, [w], known, outputs, bins, permutations, seed)
        best_output = max(per_output, key=per_output.get)
        entropy = conditional_entropy(samples, known, best_output, bins)
        ratio = gain / entropy if entropy > 0 else 0.0
        significant = p_value < alpha / len(candidates)
        relevant = ratio >= epsilon
        ranking.append({"variable": w, "gain": gain, "best_output": best_output,
                        "p_value": p_value, "entropy": entropy, "ratio": ratio,
                        "significant": significant, "relevant": relevant,
                        "accepted": significant and relevant, "per_output": per_output})
    return sorted(ranking, key=lambda r: -r["gain"])


def identify(samples, known, candidates, outputs=(None,), alpha=0.05, epsilon=0.05, bins=10,
             permutations=200, seed=0):
    """The accepted candidate with the highest gain (see :func:`rank_gains`), or None if no
    candidate is accepted. Returns ``(variable or None, ranking)``."""
    ranking = rank_gains(samples, known, candidates, outputs, alpha, epsilon, bins, permutations,
                         seed)
    accepted = {r["variable"]: r["gain"] for r in ranking if r["accepted"]}
    best = max(accepted, key=accepted.get) if accepted else None
    return best, ranking


class MutualInfoIdentifier(Identifier):
    """Add the candidate with the highest GAIN = MI(O; W | K), among those whose gain is
    significant at level ``alpha / |C|`` and removes at least a fraction ``epsilon`` of the
    remaining uncertainty, GAIN / H(O | K) >= epsilon (see :func:`rank_gains`)."""
    name = "mi"

    def __init__(self, alpha=0.05, epsilon=0.05, bins=10, permutations=200, seed=0):
        self.alpha, self.epsilon, self.bins = alpha, epsilon, bins
        self.permutations, self.seed = permutations, seed

    def identify(self, samples, known, candidates, outputs):
        best, ranking = identify(samples, known, candidates, outputs, self.alpha, self.epsilon,
                                 self.bins, self.permutations, self.seed)
        rows = [{"candidate": r["variable"], "score": r["gain"], "best_output": r["best_output"],
                 "p_value": r["p_value"], "entropy": r["entropy"], "ratio": r["ratio"],
                 "significant": r["significant"], "relevant": r["relevant"],
                 "accepted": r["accepted"]} for r in ranking]
        return ([best] if best else []), rows
