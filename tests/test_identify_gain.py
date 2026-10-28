"""Checks on the gain-based identification (identifier.identify) and on the discovery loop of
experiments/run.py."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.identifier import (MutualInfoIdentifier, conditional_entropy, identify,
                                         rank_gains)

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from run import discover  # noqa: E402

VARS = [Variable("a", [0, 1]), Variable("b", [0, 1]), Variable("c", [0, 1])]
COMBOS = list(itertools.product((0, 1), repeat=3))


def samples(mean, n=1000, seed=0):
    dists = {c: {"x": {"mean": mean(*c), "std": 1.0}, "y": {"mean": 5.0, "std": 1.0}} for c in COMBOS}
    return DataGenerator(VARS, {c: 1 / 8 for c in COMBOS}, distributions=dists,
                         seed=seed).sample_many(n)


def test_conditional_entropy():
    data = samples(lambda a, b, c: 10 + 6 * b)
    # with equal-frequency bins, the entropy of the cost alone is log(bins)
    assert conditional_entropy(data, [], "x", bins=10) == pytest.approx(np.log(10), abs=1e-6)
    # knowing b leaves less uncertainty about the cost
    assert conditional_entropy(data, ["b"], "x") < conditional_entropy(data, [], "x")


def test_rank_gains_accepts_only_the_relevant_variable():
    ranking = rank_gains(samples(lambda a, b, c: 10 + 6 * b), [], ["a", "b", "c"], ["x", "y"])
    assert ranking[0]["variable"] == "b" and ranking[0]["accepted"]
    assert ranking[0]["best_output"] == "x"
    assert not any(r["accepted"] for r in ranking[1:])


def test_epsilon_rejects_small_gains():
    # b's effect is significant but removes little uncertainty: rejected by a high epsilon
    data = samples(lambda a, b, c: 10 + 0.6 * b, n=4000)
    _, ranking = identify(data, [], ["a", "b", "c"], ["x"], epsilon=0.0)
    b = next(r for r in ranking if r["variable"] == "b")
    assert b["significant"] and 0 < b["ratio"] < 0.2
    assert identify(data, [], ["a", "b", "c"], ["x"], epsilon=0.2)[0] is None


def test_discover_adds_relevant_variables_in_order():
    known, reason, _, ident = discover(samples(lambda a, b, c: 10 + 8 * a + 4 * c), VARS, ["x", "y"],
                                       MutualInfoIdentifier())
    assert [v.name for v in known] == ["a", "c"]
    assert "unimodal" in reason
    assert [r["candidate"] for r in ident if r["selected"]] == ["a", "c"]


def test_discover_without_effect():
    known, reason, detection, ident = discover(samples(lambda a, b, c: 10.0), VARS, ["x", "y"],
                                               MutualInfoIdentifier())
    assert known == [] and ident == [] and "unimodal" in reason


def test_discover_respects_m_max():
    known, reason, _, _ = discover(samples(lambda a, b, c: 10 + 20 * a + 10 * b + 5 * c), VARS,
                                   ["x", "y"], MutualInfoIdentifier(), m_max=1)
    assert len(known) == 1 and "m_max" in reason
