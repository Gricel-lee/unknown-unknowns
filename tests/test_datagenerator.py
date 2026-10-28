"""Checks on unknown_unknowns.data_generator."""
from pathlib import Path

import numpy as np
import pytest

from unknown_unknowns.data_generator import (DataGenerator, Normal, Variable,
                                            load_distributions, load_prior, output_value)

VARS = [Variable("light", [0, 1]), Variable("wet", [0, 1]), Variable("busy", [0, 1])]


def write(tmp_path, text):
    path = tmp_path / "prior.txt"
    path.write_text(text)
    return path


def full_prior(tmp_path, probs):
    lines = [f"{i:03b} = {p}" for i, p in enumerate(probs)]
    return write(tmp_path, "\n".join(lines))


def test_load_prior(tmp_path):
    path = full_prior(tmp_path, [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1])
    prior = load_prior(path, VARS)
    assert prior[(1, 0, 1)] == 0.2
    assert len(prior) == 8


def test_load_prior_comma_form(tmp_path):
    variables = [Variable("level", [0, 5, 10]), Variable("wet", [0, 1])]
    text = "\n".join(f"{a},{b} = {1 / 6}" for a in (0, 5, 10) for b in (0, 1))
    assert load_prior(write(tmp_path, text), variables)[(10, 1)] == pytest.approx(1 / 6)


@pytest.mark.parametrize("probs, msg", [
    ([0.1] * 8, "add up to"),
    ([0.5, 0.5] + [0] * 5, "missing"),
])
def test_load_prior_errors(tmp_path, probs, msg):
    with pytest.raises(ValueError, match=msg):
        load_prior(full_prior(tmp_path, probs), VARS)


def test_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="not a value"):
        load_prior(write(tmp_path, "200 = 1"), VARS)


def test_sample(tmp_path):
    # All mass on (1, 0, 1).
    probs = [0] * 8
    probs[0b101] = 1
    gen = DataGenerator(VARS, full_prior(tmp_path, probs), seed=0)
    gen.set_distribution({"light": 1, "wet": 0, "busy": 1}, {"mean": 5, "std": 1})
    context, value = gen()
    assert context == {"light": 1, "wet": 0, "busy": 1}
    values = [v for _, v in gen.sample_many(5000)]
    assert np.mean(values) == pytest.approx(5, abs=0.1)
    assert np.std(values) == pytest.approx(1, abs=0.1)


def test_sample_frequencies():
    prior = {(0,): 0.25, (1,): 0.75}
    gen = DataGenerator([Variable("light", [0, 1])], prior,
                        distributions={(0,): Normal(0, 0), (1,): Normal(10, 0)}, seed=1)
    samples = gen.sample_many(10000)
    assert np.mean([c["light"] for c, _ in samples]) == pytest.approx(0.75, abs=0.02)
    assert all(v == 10 * c["light"] for c, v in samples)


def test_missing_distribution(tmp_path):
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8))
    gen.set_distribution((0, 0, 0), mean=5, std=1)
    with pytest.raises(ValueError, match="no distribution"):
        gen.sample()

def test_load_distributions(tmp_path):
    path = write(tmp_path, "# light wet busy\n{000 = 30, 1.2}\n101 = 42, 1.8\n")
    dists = load_distributions(path, VARS)
    assert dists == {(0, 0, 0): {"mean": 30, "std": 1.2}, (1, 0, 1): {"mean": 42, "std": 1.8}}


@pytest.mark.parametrize("text, msg", [
    ("000 = 30", "mean, std"),
    ("000 = 30, -1", "non-negative"),
    ("000 = 30, 1\n000 = 31, 1", "listed twice"),
])
def test_load_distributions_errors(tmp_path, text, msg):
    with pytest.raises(ValueError, match=msg):
        load_distributions(write(tmp_path, text), VARS)


def test_distributions_from_file():
    examples = Path(__file__).parent.parent / "examples/robot/context_data"
    gen = DataGenerator(VARS, examples / "context_prior.txt",
                        distributions=examples / "context_distributions.txt", seed=0)
    assert gen.distributions[(1, 0, 1)] == {"path1": Normal(44.5, 0.8), "path2": Normal(31, 1.2)}
    assert gen.outputs == ["path1", "path2"]
    gen.sample_many(10)


def test_load_distributions_method():
    examples = Path(__file__).parent.parent / "examples/robot/context_data"
    gen = DataGenerator(VARS, examples / "context_prior.txt")
    gen.set_distribution((0, 0, 0), {"path1": {"mean": 0, "std": 1}, "path2": {"mean": 0, "std": 1}})
    gen.load_distributions(examples / "context_distributions.txt")   # replaces (0, 0, 0)
    assert gen.distributions[(0, 0, 0)]["path1"] == Normal(26, 1.0)


def __main__():
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    pytest.main([__file__, "-v", "-s"])

    

@pytest.mark.parametrize("method", ["plot_ridgeline", "plot_ridgeline_hist"])
def test_plot_methods(tmp_path, method):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8), seed=0)
    for i in range(8):
        gen.set_distribution(tuple(int(b) for b in f"{i:03b}"), mean=i, std=1)
    ax = getattr(gen, method)(n=200)
    assert len(ax.get_yticklabels()) == 8


@pytest.mark.parametrize("plot_by, rows", [
    ("light", 2), ("light,wet", 4), (("wet", "busy"), 4), (None, 8),
])
def test_plot_by(tmp_path, plot_by, rows):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8), seed=0)
    for i in range(8):
        gen.set_distribution(tuple(int(b) for b in f"{i:03b}"), mean=i, std=1)
    assert len(gen.plot_ridgeline_hist(n=200, plot_by=plot_by).get_yticklabels()) == rows


def test_plot_by_unknown(tmp_path):
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8))
    with pytest.raises(ValueError, match="plot_by"):
        gen.plot_ridgeline(samples=[], plot_by="colour")


def test_plot_single_distribution(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8), seed=0)
    for i in range(8):
        gen.set_distribution(tuple(int(b) for b in f"{i:03b}"), mean=i, std=1)
    ax = gen.plot_single_distribution(n=300, bins=20)
    assert len(ax.patches) == 20
    assert sum(p.get_height() for p in ax.patches) == 300


PATHS = """# two paths
{env:000, path1_mean:26, path1_std:1.0, path2_mean:25, path2_std:0}
{env:101, path1_mean:44.5, path1_std:0.8, path2_mean:31, path2_std:1.2}
"""


def test_load_distributions_named(tmp_path):
    dists = load_distributions(write(tmp_path, PATHS), VARS)
    assert dists[(0, 0, 0)] == {"path1": {"mean": 26, "std": 1.0}, "path2": {"mean": 25, "std": 0}}
    assert dists[(1, 0, 1)]["path2"] == {"mean": 31, "std": 1.2}


@pytest.mark.parametrize("line, msg", [
    ("{env:000, path1_mean:26, path2_std:1.0, path2_mean:25, path2_std:1.0}", "given twice"),
    ("{env:000, path1_mean:26, path2_mean:25, path2_std:1.0}", "missing"),
    ("{env:000, path1_avg:26}", "_mean"),
])
def test_load_distributions_named_errors(tmp_path, line, msg):
    with pytest.raises(ValueError, match=msg):
        load_distributions(write(tmp_path, line), VARS)


def test_sample_two_paths(tmp_path):
    probs = [0] * 8
    probs[0] = 1
    gen = DataGenerator(VARS, full_prior(tmp_path, probs), seed=0)
    gen.load_distributions(write(tmp_path, PATHS))
    assert gen.outputs == ["path1", "path2"]
    context, value = gen()
    assert value["path2"] == 25 and set(value) == {"path1", "path2"}
    assert output_value(value, "path2") == 25
    with pytest.raises(ValueError, match="pick one with output"):
        output_value(value)


def test_mixed_outputs(tmp_path):
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8))
    gen.set_distribution((0, 0, 0), mean=5, std=1)
    gen.set_distribution((0, 0, 1), {"path1": {"mean": 5, "std": 1}})
    with pytest.raises(ValueError, match="different outputs"):
        gen.outputs


def test_plot_one_path(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    gen = DataGenerator(VARS, full_prior(tmp_path, [0.125] * 8), seed=0)
    for i in range(8):
        gen.set_distribution(tuple(int(b) for b in f"{i:03b}"),
                             {"path1": {"mean": i, "std": 1}, "path2": {"mean": -i, "std": 1}})
    ax = gen.plot_ridgeline_hist(n=200, output="path2", xlabel="cost travel path_2")
    assert ax.get_xlabel() == "cost travel path_2" and ax.get_xlim()[1] < 3
    assert gen.plot_single_distribution(n=200, output="path1").get_xlim()[1] > 5
