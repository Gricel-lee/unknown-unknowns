"""Shared definitions for the paper experiments (RQ1-RQ3).

The four case studies of ``examples/`` and the monitoring loop of the approach without PRISM:
Algorithm 1 (``discover`` in experiments/run.py: dip-test detection and identification by
conditional mutual information) is called on all the logs every time new observations have
been collected, so that many runs can be made quickly.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repository root

sys.path.insert(0, str(ROOT / "experiments"))
from run import discover  # noqa: E402
from unknown_unknowns.data_generator import DataGenerator, Variable  # noqa: E402
from unknown_unknowns.detector import all_unimodal, check_partition  # noqa: E402
from unknown_unknowns.identifier import MutualInfoIdentifier  # noqa: E402

EXAMPLES = ROOT / "examples"
OUTPUT = Path(__file__).resolve().parent   # tables/ and figures/ for the paper

# Plot colours (one per case study) and text colours.
COLORS = {"robot": "#2a78d6", "consensus": "#eb6834", "fx": "#1baf7a", "uav": "#8f5bd6"}
LABELS = {"robot": "Robot", "consensus": "Consensus", "fx": "FX", "uav": "UAV"}

# For each case study: its three context variables (in order), its two outputs (the costs
# c1, c2 of its two actions), the baseline mean and standard deviation of each output, and the
# direction in which each variable shifts each output (+1, -1 or 0). The directions are those
# of the case study's example data in examples/<case>/context_data/.
CASES = {
    "robot": {
        "variables": ["light", "wet", "busy"],
        "outputs": ["path1", "path2"],
        "base": {"path1": (26.0, 1.0), "path2": (25.0, 1.0)},
        "effect": {"light": {"path1": +1, "path2": +1},
                   "wet": {"path1": -1, "path2": +1},
                   "busy": {"path1": +1, "path2": +1}},
    },
    "consensus": {
        "variables": ["busy", "remote", "congested"],
        "outputs": ["server1", "server2"],
        "base": {"server1": (10.0, 2.0), "server2": (12.0, 2.0)},
        "effect": {"busy": {"server1": +1, "server2": 0},
                   "remote": {"server1": 0, "server2": +1},
                   "congested": {"server1": +1, "server2": +1}},
    },
    "fx": {
        "variables": ["volatile", "news", "disk_busy"],
        "outputs": ["technical", "fundamental"],
        "base": {"technical": (7.92, 0.63), "fundamental": (13.2, 1.06)},
        "effect": {"volatile": {"technical": +1, "fundamental": 0},
                   "news": {"technical": 0, "fundamental": +1},
                   "disk_busy": {"technical": +1, "fundamental": +1}},
    },
    "uav": {
        "variables": ["operator_busy", "signal_noise", "low_light"],
        "outputs": ["operator", "onboard"],
        "base": {"operator": (8.0, 0.8), "onboard": (14.0, 1.4)},
        "effect": {"operator_busy": {"operator": +1, "onboard": 0},
                   "signal_noise": {"operator": +1, "onboard": 0},
                   "low_light": {"operator": 0, "onboard": +1}},
    },
}

# Logged variables with no effect on any case study, added to every variant (RQ1.3): binary,
# uniform and independent of everything else.
IRRELEVANT = ["z1", "z2"]

# Metaparameters of Algorithm 1 in the experiments (see examples/<case>/config.json); m_max is
# the number of logged candidate variables (3 + 2), so the loop never stops early because of it.
SETTINGS = {"n_min": 10, "alpha": 0.05, "m_max": 5, "epsilon": 0.05, "permutations": 200,
            "bins": 10}

# Number of relevant variables -> which of the three variables are relevant (by position).
# u1 uses the middle variable and u2 the outer ones.
RELEVANT = {0: [], 1: [1], 2: [0, 2], 3: [0, 1, 2]}

# Difficulty -> shift of each relevant variable, in standard deviations of the output.
SEPARATION = {1: 10.0, 2: 7.5, 3: 5.0, 4: 3.75, 5: 3.5, 6: 3.0, 7: 2.75, 8: 2.5}
DIFFICULTY_NAMES = {d: f"{s:g} std" for d, s in SEPARATION.items()}
# Difficulties shown in the figure of the cost distributions.
FIGURE_DIFFICULTIES = (5, 6, 8)   # 3.5, 3 and 2.5 std


def relevant_variables(case, n_unknowns):
    names = CASES[case]["variables"]
    return [names[i] for i in RELEVANT[n_unknowns]]


def variant_name(n_unknowns, difficulty):
    return "u0" if n_unknowns == 0 else f"u{n_unknowns}_d{difficulty}"


def variants():
    """All (n_unknowns, difficulty) pairs: u0 (no difficulty) and u1-u3 x every difficulty."""
    yield 0, None
    for n in (1, 2, 3):
        for d in SEPARATION:
            yield n, d


def logged_variables(case):
    """The case study's three context variables followed by the irrelevant ones."""
    return CASES[case]["variables"] + IRRELEVANT


def generator(case, data_folder, seed):
    """A DataGenerator for ``case`` (its variables and the irrelevant ones) reading
    ``data_folder``/context_*.txt."""
    folder = Path(data_folder)
    return DataGenerator(
        variables=[Variable(v, [0, 1]) for v in logged_variables(case)],
        prior=folder / "context_prior.txt",
        distributions=folder / "context_distributions.txt",
        seed=seed,
    )


def checkpoints(start=100, step=50, growth=0.05, max_samples=10000):
    """Numbers of samples at which Algorithm 1 is called: every ``step`` samples from ``start``,
    then every ``growth`` x samples once that is larger, up to ``max_samples``."""
    n = start
    while n < max_samples:
        yield n
        n += max(step, int(growth * n))
    yield max_samples


def monitor(gen, settings=SETTINGS, **schedule):
    """Simulate the loop of the approach: collect observations and call Algorithm 1 on all of
    them at every checkpoint (see :func:`checkpoints`), keeping the known variables between calls.

    Returns a dict with the known variables at the end (in the order they were added), the
    samples at the first alarm (the first call whose bins are not all unimodal; None if never)
    and when each variable was added, the gain GAIN(D, W, K) (nats) and relative gain
    GAIN / H(O | K) of each variable when it was added, the gain of every candidate at every
    identification step (``rankings``, one ``{candidate: {"gain", "ratio", "significant",
    "relevant"}}`` per step), the stop reason of
    the last call, and the mean time per call (seconds).
    """
    outputs = gen.outputs or [None]
    identifier = MutualInfoIdentifier(settings["alpha"], settings["epsilon"], settings["bins"],
                                      settings["permutations"], seed=0)
    known, samples, first_alarm, added_at, gains, rankings, times = [], [], None, {}, [], [], []
    for n in checkpoints(**schedule):
        samples += gen.sample_many(n - len(samples))
        t0 = time.perf_counter()
        new_known, reason, detection, identification = discover(
            samples, gen.variables, outputs, identifier, settings["alpha"], settings["m_max"],
            settings["n_min"], known)
        times.append(time.perf_counter() - t0)
        if first_alarm is None and any(row["unimodal"] is False for row in detection):
            first_alarm = n
        for v in new_known[len(known):]:
            added_at[v.name] = n
        gains += [{"variable": row["candidate"], "gain": row["score"], "ratio": row["ratio"]}
                  for row in identification if row["selected"]]
        for step in sorted({row["iteration"] for row in identification}):
            rankings.append({row["candidate"]: {"gain": row["score"], "ratio": row["ratio"],
                                                "significant": bool(row["significant"]),
                                                "relevant": bool(row["relevant"])}
                             for row in identification if row["iteration"] == step})
        known = new_known
    return {"accepted": [v.name for v in known], "first_alarm": first_alarm,
            "accepted_at": added_at, "gains": gains, "rankings": rankings,
            "stop_reason": reason,
            "time_per_check": sum(times) / len(times), "samples": len(samples)}
