"""Checks on the dip-test detector and the unimodality identifier (check_partition,
identification_step, has_effect)."""
import itertools

from unknown_unknowns.data_generator import DataGenerator, Variable
from unknown_unknowns.detector import all_unimodal, check_partition
from unknown_unknowns.identifier import has_effect, identification_step

VARS = [Variable("a", [0, 1]), Variable("b", [0, 1]), Variable("c", [0, 1])]


def generator(prior, mean, seed=0):
    """Costs x (one output) with mean `mean(a, b, c)` and std 1."""
    combos = list(itertools.product((0, 1), repeat=3))
    dists = {c: {"x": {"mean": mean(*c), "std": 1.0}} for c in combos}
    return DataGenerator(VARS, prior, distributions=dists, seed=seed)


UNIFORM = {c: 1 / 8 for c in itertools.product((0, 1), repeat=3)}


def run_detector(gen, n=2000, step=200):
    known, unknown, tried = [], list(VARS), {}
    samples, events = gen.sample_many(200), []
    while len(samples) <= n:
        results = check_partition(samples, known, 0.05, ["x"])
        if not all_unimodal(results):    # anomaly -> identification
            known, unknown, _, event = identification_step(samples, results, VARS, known,
                                                               unknown, tried, 0.05, ["x"])
            if event:
                events.append(event)
        samples += gen.sample_many(step)
    return [v.name for v in known], events


def test_has_effect():
    samples = generator(UNIFORM, lambda a, b, c: 10 + 6 * b).sample_many(1000)
    assert has_effect(samples, VARS[1], [], 0.05, ["x"])        # b shifts the cost
    assert not has_effect(samples, VARS[0], [], 0.05, ["x"])    # a does not


def test_finds_the_relevant_variable_only():
    known, events = run_detector(generator(UNIFORM, lambda a, b, c: 10 + 6 * b))
    assert known == ["b"]
    assert ("tried", ["a"]) in events     # the distractor before it was rejected


def test_no_alarm_without_effect():
    known, events = run_detector(generator(UNIFORM, lambda a, b, c: 10.0))
    assert known == [] and events == []


def test_proxy_is_pruned():
    # a is correlated with b (a proxy of it); only b has an effect
    prior = {c: (0.2 if c[0] == c[1] else 0.05) for c in itertools.product((0, 1), repeat=3)}
    known, events = run_detector(generator(prior, lambda a, b, c: 10 + 6 * b), n=4000)
    assert known == ["b"]


def test_waits_for_small_bins():
    samples = generator(UNIFORM, lambda a, b, c: 10 + 6 * a + 6 * b + 6 * c).sample_many(60)
    results = check_partition(samples, [], 0.05, ["x"])
    assert not all_unimodal(results)           # the anomaly is detected
    _, _, _, event = identification_step(samples, results, VARS, [], list(VARS),
                                        {("a",): 60, ("b",): 60, ("c",): 60,   # rejected,
                                         ("a", "b"): 60, ("a", "c"): 60, ("b", "c"): 60},  # not retried yet
                                        0.05, ["x"])
    assert event == ("waiting", ["a", "b", "c"])    # 8 bins of ~7 samples: too small to test


def test_rejected_candidates_are_retried_with_more_data():
    tried = {("b",): 100}
    samples = generator(UNIFORM, lambda a, b, c: 10 + 6 * b).sample_many(150)
    results = check_partition(samples, [], 0.05, ["x"])
    known, _, _, event = identification_step(samples, results, VARS, [], [VARS[1]], tried,
                                                 0.05, ["x"])
    assert event is None and known == []           # 150 < 2 * 100: not retried yet
    samples += generator(UNIFORM, lambda a, b, c: 10 + 6 * b, seed=1).sample_many(100)
    results = check_partition(samples, [], 0.05, ["x"])
    known, _, _, event = identification_step(samples, results, VARS, [], [VARS[1]], tried,
                                                 0.05, ["x"])
    assert event[0] == "added" and [v.name for v in known] == ["b"]
