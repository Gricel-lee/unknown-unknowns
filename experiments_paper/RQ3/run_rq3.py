"""RQ3 (scalability): what are the computational overheads, and how do they grow?

Two measurements, written to ``results/`` and to the paper:

1. Controller synthesis per case study and number of known unknowns k = 0..3: size of the
   PRISM model (states, transitions) and time to build, verify and read the policy
   (``model_size.csv``, ``experiments_paper/tables/rq3.tex``). The models range from 4 to 400,161 states.
2. Analysis: time of a dip-test check with every candidate known, and of Algorithm 1 from
   scratch (detection and identification), as the number of candidates |C| (column ``m``)
   and of samples n grow (``detector_scaling.csv``).

The time of Algorithm 1 is plotted in ``experiments_paper/figures/rq3_scaling.pdf``.

Run: python experiments_paper/RQ3/run_rq3.py   (--plot-only: redraw the figure from results/)
"""
import csv
import itertools
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CASES, COLORS, EXAMPLES, LABELS, OUTPUT, SETTINGS, discover  # noqa: E402


from unknown_unknowns.controller_generator import PrismController  # noqa: E402
from unknown_unknowns.controller_generator.prism import run_prism  # noqa: E402
from unknown_unknowns.data_generator import DataGenerator, Variable  # noqa: E402
from unknown_unknowns.detector import check_partition  # noqa: E402
from unknown_unknowns.identifier import MutualInfoIdentifier  # noqa: E402

RESULTS = HERE / "results"
REPEATS = 3


def prism_size(model, prop):
    """(states, transitions, PRISM's model-construction + model-checking time in s)."""
    out = run_prism([model, prop, "-prop", "1", "-epsilon", "1e-10"])
    states = int(re.search(r"^States:\s+(\d+)", out, re.M)[1])
    transitions = int(re.search(r"^Transitions:\s+(\d+)", out, re.M)[1])
    build = float(re.search(r"Time for model construction:\s*([\d.]+)", out)[1])
    check = float(re.search(r"Time for model checking:\s*([\d.]+)", out)[1])
    return states, transitions, build + check


def synthesis(case, templates, k, samples, folder):
    """Build and verify the controller with the first k variables known; returns a row."""
    controller = PrismController(folder, templates, data=samples,
                                 unknowns=CASES[case]["variables"][:k], name=f"{case}_{k}")
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        controller.get_controller()
        times.append(time.perf_counter() - t0)
    states, transitions, prism_time = prism_size(controller.prism_file, controller.property_file)
    return {"case": case, "k": k, "states": states, "transitions": transitions,
            "prism_time": round(prism_time, 3), "synthesis_time": round(statistics.median(times), 3)}


def model_size_rows(folder):
    rows = []
    for case in CASES:
        folder = EXAMPLES / case / "context_data"
        samples = DataGenerator([Variable(v, [0, 1]) for v in CASES[case]["variables"]],
                                folder / "context_prior.txt",
                                distributions=folder / "context_distributions.txt",
                                seed=0).sample_many(1000)
        for k in range(4):
            rows.append(synthesis(case, EXAMPLES / case / "prism_templates", k, samples, folder))
            print("  ", rows[-1])
    return rows


def synthetic_generator(m, seed, shift=6.0):
    """m binary candidates, uniform prior; the first two shift output a (resp. b) by `shift`."""
    variables = [Variable(f"x{i}", [0, 1]) for i in range(m)]
    combos = list(itertools.product((0, 1), repeat=m))
    prior = {c: 1 / len(combos) for c in combos}
    dists = {c: {"a": {"mean": 10 + shift * c[0], "std": 1.0},
                 "b": {"mean": 10 + shift * c[1], "std": 1.0}} for c in combos}
    return DataGenerator(variables, prior, distributions=dists, seed=seed)


def detector_scaling_rows(ms=(3, 4, 5, 6, 7, 8), ns=(1000, 5000, 20000)):
    """Time of one check with every candidate known (2^m bins x 2 outputs) and of Algorithm 1
    from scratch (m_max = m), for m = |C| candidates and n samples."""
    rows = []
    for m in ms:
        gen = synthetic_generator(m, seed=0)
        for n in ns:
            samples = gen.sample_many(n)
            t0 = time.perf_counter()
            check_partition(samples, gen.variables, 0.05, gen.outputs)
            all_known = time.perf_counter() - t0
            identifier = MutualInfoIdentifier(SETTINGS["alpha"], SETTINGS["epsilon"],
                                              SETTINGS["bins"], SETTINGS["permutations"])
            t0 = time.perf_counter()
            known, _, _, identification = discover(samples, gen.variables, gen.outputs,
                                                   identifier, SETTINGS["alpha"], m,
                                                   SETTINGS["n_min"])
            search = time.perf_counter() - t0
            rows.append({"m": m, "n": n, "bins": 2 ** m, "check_all_known": round(all_known, 4),
                         "iterations": len({r["iteration"] for r in identification}),
                         "search_time": round(search, 4),
                         "accepted": " ".join(v.name for v in known)})
            print("  ", rows[-1])
    return rows


def latex_table(rows):
    """One row per case study and number of known unknowns k: states, transitions and the
    time (s) of the full policy synthesis."""
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Case study & $k$ & States & Transitions & Synthesis (s)\\", r"\midrule"]
    for case in CASES:
        for r in (x for x in rows if x["case"] == case):
            name = LABELS[case] if int(r["k"]) == 0 else ""
            lines.append(f"{name} & {int(r['k'])} & {int(r['states']):,} & {int(r['transitions']):,} & "
                         f"{float(r['synthesis_time']):.2f}\\\\")
        lines.append(r"\midrule" if case != list(CASES)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines).replace(",", "\\,") + "\n"


def figure(size_rows, detector_rows, path):
    """Time of Algorithm 1 from scratch against the number of candidates |C|, for each number
    of samples. (``size_rows`` is not
    plotted; the synthesis times are in the table.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    shades = {1000: "#1baf7a", 5000: "#eb6834", 20000: "#8f5bd6"}
    for n in sorted({r["n"] for r in detector_rows}):
        rs = [r for r in detector_rows if r["n"] == n]
        ax.plot([r["m"] for r in rs], [r["search_time"] for r in rs], "o-", lw=1.5, ms=4,
                color=shades.get(n, "#2a78d6"), label=f"n = {int(n):,}")
    ax.set_yscale("log")
    ax.set_xlabel("candidates $|C|$", fontsize=8)
    ax.set_ylabel("time (s)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e5e5e1", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    return [{k: (float(v) if re.fullmatch(r"[\d.]+", v) else v) for k, v in row.items()}
            for row in csv.DictReader(open(path))]


if __name__ == "__main__":
    (OUTPUT / "tables").mkdir(exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    if "--plot-only" in sys.argv:   # redraw the figure from the saved results
        (OUTPUT / "tables" / "rq3.tex").write_text(latex_table(read_csv(RESULTS / "model_size.csv")))
        figure(read_csv(RESULTS / "model_size.csv"), read_csv(RESULTS / "detector_scaling.csv"),
               OUTPUT / "figures" / "rq3_scaling.pdf")
        sys.exit()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print("1. controller synthesis per case study and k")
        sizes = model_size_rows(tmp)
    print("2. detector")
    detector = detector_scaling_rows()
    write_csv(RESULTS / "model_size.csv", sizes)
    write_csv(RESULTS / "detector_scaling.csv", detector)
    (OUTPUT / "tables" / "rq3.tex").write_text(latex_table(sizes))
    figure(sizes, detector, OUTPUT / "figures" / "rq3_scaling.pdf")
