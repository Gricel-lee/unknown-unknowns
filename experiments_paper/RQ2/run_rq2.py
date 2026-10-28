"""RQ2 (latency): how many observations does the approach need to find the unknown unknowns?

Uses the runs of RQ1 (``../RQ1/results/runs.csv``; run RQ1 first). For every case study,
number of relevant variables and difficulty it reports the number of samples until

- the first alarm (some bin of the known variables is not unimodal), and
- detection (the last relevant variable is accepted, in runs that end correct),

and writes ``results/latency.csv``, ``experiments_paper/tables/rq2.tex`` (median samples to detection)
and ``experiments_paper/figures/rq2_latency.pdf``.

Run: python experiments_paper/RQ2/run_rq2.py
"""
import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CASES, COLORS, LABELS, OUTPUT, SEPARATION  # noqa: E402

RUNS = HERE.parent / "RQ1" / "results" / "runs.csv"
RESULTS = HERE / "results"


def quantiles(xs):
    if not xs:
        return None, None, None
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
    return q[0], statistics.median(xs), q[2]


def latency_rows(runs):
    rows = []
    for case in CASES:
        for n in (1, 2, 3):
            for d in SEPARATION:
                group = [r for r in runs if r["case"] == case and r["n_unknowns"] == str(n)
                         and r["difficulty"] == str(d)]
                alarms = [int(r["first_alarm"]) for r in group if r["first_alarm"]]
                detections = [int(r["detected_at"]) for r in group if r["detected_at"]]
                a_lo, a_med, a_hi = quantiles(alarms)
                d_lo, d_med, d_hi = quantiles(detections)
                rows.append({"case": case, "n_unknowns": n, "difficulty": d, "runs": len(group),
                             "detected": len(detections),
                             "alarm_median": a_med, "alarm_q1": a_lo, "alarm_q3": a_hi,
                             "detection_median": d_med, "detection_q1": d_lo,
                             "detection_q3": d_hi})
    return rows


def fmt(x):
    if x is None:
        return "---"
    return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def latex_table(rows):
    """Median samples to detection; rows: unknowns x difficulty, columns: case studies."""
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             "Unknowns & Difficulty & " + " & ".join(LABELS[c] for c in CASES) + r"\\",
             r"\midrule"]
    for n in (1, 2, 3):
        for d in SEPARATION:
            cells = []
            for case in CASES:
                r = next(x for x in rows if x["case"] == case and x["n_unknowns"] == n
                         and x["difficulty"] == d)
                cell = fmt(r["detection_median"])
                if r["detected"] < r["runs"]:
                    cell += f"$^{{{r['detected']}}}$"
                cells.append(cell)
            lines.append(f"{n} & {SEPARATION[d]:g}$\\sigma$ & "
                         + " & ".join(cells) + r"\\")
        lines.append(r"\midrule" if n < 3 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def figure(rows, path):
    """Median (and quartiles) of the samples to detection, per difficulty and case study."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3), sharey=True)
    offsets = {c: (i - 1.5) * 0.12 for i, c in enumerate(CASES)}
    for ax, n in zip(axes, (1, 2, 3)):
        for case in CASES:
            xs, meds, los, his = [], [], [], []
            for d in SEPARATION:
                r = next(x for x in rows if x["case"] == case and x["n_unknowns"] == n
                         and x["difficulty"] == d)
                if r["detection_median"] is None:
                    continue
                xs.append(d + offsets[case])
                meds.append(r["detection_median"])
                los.append(r["detection_median"] - r["detection_q1"])
                his.append(r["detection_q3"] - r["detection_median"])
            ax.errorbar(xs, meds, yerr=[los, his], fmt="o-", ms=4, lw=1.5, capsize=2,
                        color=COLORS[case], label=LABELS[case])
        ax.set_yscale("log")
        ax.set_xticks(list(SEPARATION), [f"{s:g}σ" for s in SEPARATION.values()], fontsize=8)
        ax.set_title(f"{n} unknown unknown{'s' if n > 1 else ''}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e5e5e1", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("samples to detection", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    (OUTPUT / "tables").mkdir(exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    runs = list(csv.DictReader(open(RUNS)))
    rows = latency_rows(runs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "latency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "tables" / "rq2.tex").write_text(latex_table(rows))
    figure(rows, OUTPUT / "figures" / "rq2_latency.pdf")
    for r in rows:
        print(f"  {r['case']:9} u{r['n_unknowns']} d{r['difficulty']}: detected {r['detected']}/"
              f"{r['runs']}, first alarm {fmt(r['alarm_median'])}, detection "
              f"{fmt(r['detection_median'])} [{fmt(r['detection_q1'])}, {fmt(r['detection_q3'])}]")
