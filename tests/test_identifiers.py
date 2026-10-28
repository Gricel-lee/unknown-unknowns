"""Checks on the Identifier interface, the SHAP identifier and the other identifiers."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.identifier import IDENTIFIERS, Identifier, ShapIdentifier, shapley_values

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from run import discover  # noqa: E402

VARS = [Variable("a", [0, 1]), Variable("b", [0, 1]), Variable("c", [0, 1])]
COMBOS = list(itertools.product((0, 1), repeat=3))


def samples(mean, n=1000, seed=0):
    dists = {c: {"x": {"mean": mean(*c), "std": 1.0}, "y": {"mean": 5.0, "std": 1.0}} for c in COMBOS}
    return DataGenerator(VARS, {c: 1 / 8 for c in COMBOS}, distributions=dists,
                         seed=seed).sample_many(n)


def test_identifier_is_abstract():
    with pytest.raises(TypeError):
        Identifier()


def test_shapley_values_are_exact():
    # additive effects with balanced binary variables: phi_j = effect_j * (x_j - 1/2), exactly
    # in expectation; the values always add up to v(C) - v({}) (efficiency)
    data = samples(lambda a, b, c: 10 + 4 * a + 2 * b)
    cost = np.array([v["x"] for _, v in data])
    cols = {n: np.array([ctx[n] for ctx, _ in data]) for n in "abc"}
    phi = shapley_values(cost, [], cols)
    assert np.mean(np.abs(phi["a"])) == pytest.approx(2.0, abs=0.15)
    assert np.mean(np.abs(phi["b"])) == pytest.approx(1.0, abs=0.15)
    assert np.mean(np.abs(phi["c"])) < 0.15
    total = sum(phi.values())
    assert math.isclose(float(np.mean(total)), 0.0, abs_tol=1e-9)


@pytest.mark.parametrize("name", sorted(IDENTIFIERS))
def test_every_identifier_finds_the_relevant_variable(name):
    selected, ranking = IDENTIFIERS[name]().identify(samples(lambda a, b, c: 10 + 6 * b), [],
                                                     ["a", "b", "c"], ["x", "y"])
    assert selected == ["b"]
    assert all({"candidate", "score", "accepted"} <= set(row) for row in ranking)


@pytest.mark.parametrize("name", ["mi", "shap", "ctree"])
def test_no_identifier_selects_without_effect(name):
    # these identifiers correct for the number of candidates tested; the unimodality search does
    # not (each candidate's effect test is at level alpha), and relies on the detection step to
    # be called only when some bin is multimodal (see the next test)
    selected, _ = IDENTIFIERS[name]().identify(samples(lambda a, b, c: 10.0), [], ["a", "b", "c"],
                                               ["x", "y"])
    assert selected == []


@pytest.mark.parametrize("name", sorted(IDENTIFIERS))
def test_no_variable_is_discovered_without_effect(name):
    known, reason, _, ident = discover(samples(lambda a, b, c: 10.0), VARS, ["x", "y"],
                                       IDENTIFIERS[name]())
    assert known == [] and ident == [] and "unimodal" in reason


def test_shap_epsilon_rejects_small_effects():
    data = samples(lambda a, b, c: 10 + 0.6 * b, n=4000)
    _, ranking = ShapIdentifier(epsilon=0.0).identify(data, [], ["a", "b", "c"], ["x"])
    b = next(r for r in ranking if r["candidate"] == "b")
    assert b["significant"] and 0 < b["ratio"] < 0.5
    assert ShapIdentifier(epsilon=0.5).identify(data, [], ["a", "b", "c"], ["x"])[0] == []
