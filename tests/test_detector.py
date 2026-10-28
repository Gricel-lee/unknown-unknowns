"""Checks on unknown_unknowns.detector."""
from pathlib import Path

import numpy as np
import pytest

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.detector import detect, dip_test, select

EXAMPLES = Path(__file__).parent.parent / "examples/robot/context_data"
VARS = [Variable("light", [0, 1]), Variable("wet", [0, 1]), Variable("busy", [0, 1])]


def test_dip_test_unimodal():
    values = np.random.default_rng(0).normal(0, 1, 1000)
    is_unimodal, p = dip_test(values)
    assert is_unimodal and p > 0.05


def test_dip_test_bimodal():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 500), rng.normal(8, 1, 500)])
    is_unimodal, p = dip_test(values)
    assert not is_unimodal and p < 0.05


def test_dip_test_too_few_values():
    with pytest.raises(ValueError, match="at least 4"):
        dip_test([1.0, 2.0])


def test_select():
    samples = [({"light": 1, "busy": 0}, 1.0), ({"light": 1, "busy": 1}, 2.0),
               ({"light": 0, "busy": 0}, 3.0)]
    assert list(select(samples)) == [1.0, 2.0, 3.0]
    assert list(select(samples, {"light": 1})) == [1.0, 2.0]
    assert list(select(samples, {"light": 1, "busy": 0})) == [1.0]
    with pytest.raises(ValueError, match="unknown variables"):
        select(samples, {"colour": 1})


def test_detect_on_generated_data():
    gen = DataGenerator(VARS, EXAMPLES / "context_prior.txt",
                        distributions=EXAMPLES / "context_distributions.txt", seed=0)
    samples = gen.sample_many(2000)
    # One bin is a single Normal, so its path-1 costs should look unimodal.
    assert detect(samples, where={"light": 1, "wet": 0, "busy": 0}, output="path1")[0]
    # All bins mixed together (path-1 means from 19 to 44.5) should not.
    assert not detect(samples, output="path1")[0]


def test_select_output():
    samples = [({"light": 1}, {"path1": 1.0, "path2": 5.0}), ({"light": 0}, {"path1": 2.0, "path2": 6.0})]
    assert list(select(samples, {"light": 1}, output="path2")) == [5.0]
    assert list(select(samples, output="path1")) == [1.0, 2.0]
    with pytest.raises(ValueError, match="pick one with output"):
        select(samples)
