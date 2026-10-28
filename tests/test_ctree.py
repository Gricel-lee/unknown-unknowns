"""Checks on unknown_unknowns.identifier.ctree."""
import itertools

import numpy as np
import pytest

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.identifier import ctree, independence_test, variables_used
from unknown_unknowns.identifier.ctree import chi2_sf

VARS = [Variable("a", [0, 1]), Variable("b", [0, 1]), Variable("c", [0, 1])]
COMBOS = list(itertools.product((0, 1), repeat=3))


def samples(mean, n=1000, seed=0):
    dists = {c: {"x": {"mean": mean(*c), "std": 1.0}, "y": {"mean": 5.0, "std": 1.0}} for c in COMBOS}
    return DataGenerator(VARS, {c: 1 / 8 for c in COMBOS}, distributions=dists,
                         seed=seed).sample_many(n)


@pytest.mark.parametrize("x, df, expected", [(3.841459, 1, 0.05), (5.991465, 2, 0.05),
                                             (9.21034, 2, 0.01), (0.5, 3, 0.918891)])
def test_chi2_sf(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, rel=1e-4)


def test_independence_test_is_calibrated():
    rng = np.random.default_rng(0)
    ps = [independence_test(rng.integers(0, 2, 200), rng.normal(size=(200, 2)))[1] for _ in range(1000)]
    assert 0.03 < np.mean(np.array(ps) < 0.05) < 0.07


def test_tree_finds_the_relevant_variables():
    tree = ctree(samples(lambda a, b, c: 10 + 3 * a + 3 * c), ["a", "b", "c"], ["x", "y"])
    assert sorted(variables_used(tree)) == ["a", "c"]


def test_local_split():
    # c matters only when a=1: the tree splits on c in that branch only
    tree = ctree(samples(lambda a, b, c: 10 + 3 * a + 3 * a * c, n=2000), ["a", "b", "c"], ["x"])
    assert tree["split"] == "a"
    assert tree["children"][1]["split"] == "c" and tree["children"][0]["split"] is None


def test_no_split_without_effect():
    tree = ctree(samples(lambda a, b, c: 10.0), ["a", "b", "c"], ["x", "y"])
    assert tree["split"] is None and variables_used(tree) == []
