"""Checks on unknown_unknowns.identifier.mutual_info."""
import itertools

import numpy as np
import pytest

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.identifier import (conditional_mutual_information, mutual_information,
                                       rank_candidates, select_variables)

VARS = [Variable("a", [0, 1]), Variable("b", [0, 1]), Variable("c", [0, 1])]
COMBOS = list(itertools.product((0, 1), repeat=3))
UNIFORM = {c: 1 / 8 for c in COMBOS}


def samples(mean, prior=UNIFORM, n=1000, seed=0):
    dists = {c: {"x": {"mean": mean(*c), "std": 1.0}} for c in COMBOS}
    return DataGenerator(VARS, prior, distributions=dists, seed=seed).sample_many(n)


def test_mi_matches_the_kl_formula():
    # two equally likely partitions with disjoint costs: MI = log 2 (one bit), estimated with
    # 2 equal-frequency bins
    data = [({"a": 0, "b": 0, "c": 0}, {"x": 0.0})] * 50 + [({"a": 1, "b": 0, "c": 0}, {"x": 9.0})] * 50
    assert mutual_information(data, ["a"], "x", bins=2) == pytest.approx(np.log(2))
    assert mutual_information(data, ["b"], "x", bins=2) == pytest.approx(0.0)


def test_mi_never_decreases_with_more_variables():
    data = samples(lambda a, b, c: 10 + 6 * b)
    assert mutual_information(data, ["a", "b"], "x") >= mutual_information(data, ["b"], "x")


def test_conditional_mi_of_a_proxy_is_small():
    prior = {c: (0.2 if c[0] == c[1] else 0.05) for c in COMBOS}   # a correlated with b
    data = samples(lambda a, b, c: 10 + 6 * b, prior)
    assert mutual_information(data, ["a"], "x") > 0.05                        # a looks informative
    assert conditional_mutual_information(data, ["a"], ["b"], "x") < 0.02     # ... but not given b


def test_rank_finds_the_relevant_variable():
    ranking = rank_candidates(samples(lambda a, b, c: 10 + 6 * b), [], ["a", "b", "c"], ["x"])
    corrected = 0.05 / len(ranking)       # 7 combinations tested
    singles = [r for r in ranking if len(r["variables"]) == 1]
    assert singles[0]["variables"] == ["b"] and singles[0]["p_value"] < corrected
    assert all(r["p_value"] > corrected for r in singles[1:])


def test_select_variables():
    selected, _ = select_variables(samples(lambda a, b, c: 10 + 6 * a + 6 * c), [], ["a", "b", "c"], ["x"])
    assert sorted(selected) == ["a", "c"]


def test_no_selection_without_effect():
    selected, _ = select_variables(samples(lambda a, b, c: 10.0), [], ["a", "b", "c"], ["x"])
    assert selected == []
