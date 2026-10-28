"""RQ1 (effectiveness): does the approach find exactly the unknown unknowns that matter?

Runs the monitoring loop of the approach (``common.monitor``: Algorithm 1 called on all the
logs at every checkpoint) on every variant of ``examples/`` (see ``generate_variants.py``)
with ``--seeds`` random seeds, and writes to ``results/``:

- ``runs.csv``: one row per run (accepted variables, correct or not, false positives, of
  which irrelevant z's, missed variables, samples at the first alarm and when each variable
  was accepted);
- ``summary.csv`` and ``experiments_paper/tables/rq1.tex``: the share of correct runs, of runs with a
  false positive and of runs that missed a variable, per case study and variant;
- ``experiments_paper/figures/rq1_difficulty.pdf``: what the three difficulties look like (robot, u2).

The runs are also the data of RQ2 (latency). Run:
    python experiments_paper/RQ1/generate_variants.py
    python experiments_paper/RQ1/run_rq1.py --seeds 30
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (CASES, COLORS, FIGURE_DIFFICULTIES, IRRELEVANT, LABELS,  # noqa: E402
                    OUTPUT, SEPARATION, generator, monitor, relevant_variables, variant_name,
                    variants)

RESULTS = HERE / "results"
FIELDS = ["case", "n_unknowns", "difficulty", "seed", "relevant", "accepted", "correct",
          "false_positives", "irrelevant_accepted", "missed", "first_alarm", "detected_at",
          "stop_reason", "time_per_check", "accepted_at", "gains", "irrelevant_gains",
          "irrelevant_tests"]


def one_run(args):
    case, n, d, seed, start, step, max_samples, growth = args
    folder = HERE / "examples" / case / variant_name(n, d) / "context_data"
    result = monitor(generator(case, folder, seed), start=start, step=step, growth=growth,
                     max_samples=max_samples)
    relevant = relevant_variables(case, n)
    accepted = result["accepted"]
    correct = sorted(accepted) == sorted(relevant)
    return {
        "case": case, "n_unknowns": n, "difficulty": d or "", "seed": seed,
        "relevant": " ".join(relevant), "accepted": " ".join(accepted), "correct": int(correct),
        "false_positives": len(set(accepted) - set(relevant)),
        "irrelevant_accepted": len(set(accepted) & set(IRRELEVANT)),
        "missed": len(set(relevant) - set(accepted)),
        "first_alarm": result["first_alarm"] or "",
        # samples at which the last relevant variable was accepted (correct runs only)
        "detected_at": (max(result["accepted_at"][v] for v in relevant)
                        if correct and relevant else ""),
        "stop_reason": result["stop_reason"],
        "time_per_check": round(result["time_per_check"], 6),
        "accepted_at": json.dumps(result["accepted_at"]),
        "gains": json.dumps(result["gains"]),
        # gains of the candidates without an effect, at every identification step
        "irrelevant_gains": json.dumps([t["gain"] for step in result["rankings"]
                                        for name, t in step.items() if name not in relevant]),
        # (relative gain, significant, gain/H >= epsilon) of the same candidates
        "irrelevant_tests": json.dumps([[t["ratio"], t["significant"], t["relevant"]]
                                        for step in result["rankings"]
                                        for name, t in step.items() if name not in relevant]),
    }


def summarise(rows):
    summary = []
    for case in CASES:
        for n, d in variants():
            runs = [r for r in rows if r["case"] == case and r["n_unknowns"] == n
                    and r["difficulty"] == (d or "")]
            k = len(runs)
            summary.append({
                "case": case, "variant": variant_name(n, d), "n_unknowns": n,
                "difficulty": d or "", "runs": k,
                "correct": round(100 * sum(r["correct"] for r in runs) / k, 1),
                "false_positive": round(100 * sum(r["false_positives"] > 0 for r in runs) / k, 1),
                "irrelevant": round(100 * sum(r["irrelevant_accepted"] > 0 for r in runs) / k, 1),
                "alarm": round(100 * sum(bool(r["first_alarm"]) for r in runs) / k, 1),
                "missed": round(100 * sum(r["missed"] > 0 for r in runs) / k, 1),
            })
    return summary


def observations(rows, case, n, d):
    """Mean and standard deviation of the samples to detection over the correct runs of a
    variant of ``case`` (None if no run detected all its relevant variables)."""
    xs = [r["detected_at"] for r in rows if r["case"] == case and r["n_unknowns"] == n
          and r["difficulty"] == (d or "") and r["detected_at"] != ""]
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    std = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return mean, std


def latex_table(summary, rows):
    """Rows: variants; columns: % correct runs per case study (false-positive % if any) and the
    observations to detection over the correct runs per case study, mean (std)."""
    names = " & ".join(LABELS[c] for c in CASES)
    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{Correct runs (\%)} & "
             r"\multicolumn{4}{c}{Observations to identification, mean (std)}\\",
             r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
             f"Unknowns & Shift & {names} & {names}" + r"\\", r"\midrule"]
    last = max(SEPARATION)
    for n, d in variants():
        cells = []
        for case in CASES:
            s = next(x for x in summary if x["case"] == case and x["variant"] == variant_name(n, d))
            cell = f"{s['correct']:.0f}"
            if s["false_positive"]:
                cell += f" ({s['false_positive']:.0f})"
            cells.append(cell)
        for case in CASES:
            obs = observations(rows, case, n, d) if n else None
            cells.append(f"{obs[0]:,.0f} ({obs[1]:,.0f})".replace(",", "\\,") if obs else "---")
        diff = f"{SEPARATION[d]:g}$\\sigma$" if d else "---"
        lines.append(f"{n} & {diff} & " + " & ".join(cells) + r"\\")
        if n == 0 or d == last:
            lines.append(r"\midrule" if n < 3 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def gains_table(rows):
    """Rows: variants with unknown unknowns; cells: the mean gain MI_1 / MI_2 / MI_3 (nats) of
    the first, second and third variable added, over the correct runs of each case study; last
    row: the mean (maximum) gain of the irrelevant candidates over all identification steps."""
    lines = [r"\begin{tabular}{@{}llllll@{}}", r"\toprule",
             "Unknowns & Shift & " + " & ".join(LABELS[c] for c in CASES) + r"\\", r"\midrule"]
    last = max(SEPARATION)
    for n in (1, 2, 3):
        for d in SEPARATION:
            cells = []
            for case in CASES:
                runs = [json.loads(r["gains"]) for r in rows if r["case"] == case
                        and r["n_unknowns"] == n and r["difficulty"] == d and r["correct"]]
                if runs:
                    means = [sum(g[i]["gain"] for g in runs) / len(runs) for i in range(n)]
                    cells.append(" / ".join(f"{m:.2f}" for m in means)
                                 + (f"$^{{{len(runs)}}}$" if len(runs) < 30 else ""))
                else:
                    cells.append("---")
            lines.append(f"{n} & {SEPARATION[d]:g}$\\sigma$ & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    cells = []
    for case in CASES:
        xs = [g for r in rows if r["case"] == case for g in json.loads(r["irrelevant_gains"])]
        cells.append(f"{sum(xs) / len(xs):.3f} ({max(xs):.3f})" if xs else "---")
    lines += [r"\multicolumn{2}{@{}l}{Irrelevant} & " + " & ".join(cells) + r"\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def difficulty_figure(path, seed=0, n_samples=2000):
    """Path-1 costs of the robot with 2 relevant variables, at the difficulties of
    ``FIGURE_DIFFICULTIES``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.0), sharey=True)
    for ax, d in zip(axes, FIGURE_DIFFICULTIES):
        folder = HERE / "examples" / "robot" / variant_name(2, d) / "context_data"
        samples = generator("robot", folder, seed).sample_many(n_samples)
        xs = [value["path1"] for _, value in samples]
        mean = sum(xs) / len(xs)
        ax.hist([x / mean for x in xs], bins=40, color=COLORS["robot"], edgecolor="white", lw=0.5)
        ax.set_title(f"shift {SEPARATION[d]:g}σ", fontsize=9, loc="left")
        ax.set_xlabel("normalised cost of path 1", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("samples", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--growth", type=float, default=0.05,
                        help="after the first checks, check again after this share of new data")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    jobs = [(case, n, d, seed, args.start, args.step, args.max_samples, args.growth)
            for case in CASES for n, d in variants() for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one_run, jobs, chunksize=1))

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tables" / "rq1.tex").write_text(latex_table(summary, rows))
    (OUTPUT / "tables" / "rq1_gains.tex").write_text(gains_table(rows))
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    difficulty_figure(OUTPUT / "figures" / "rq1_difficulty.pdf")

    total = len(rows)
    print(f"{total} runs: {sum(r['correct'] for r in rows)} correct, "
          f"{sum(r['false_positives'] > 0 for r in rows)} with a false positive, "
          f"{sum(r['missed'] > 0 for r in rows)} missed a variable")
    for s in summary:
        print(f"  {s['case']:9} {s['variant']:6} correct {s['correct']:5.1f}%  "
              f"FP {s['false_positive']:5.1f}% (z {s['irrelevant']:5.1f}%)  "
              f"missed {s['missed']:5.1f}%  alarm {s['alarm']:5.1f}%")
