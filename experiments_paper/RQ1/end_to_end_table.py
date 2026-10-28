"""End-to-end results of the four case studies (Algorithm 1 + PRISM controllers), RQ1.2.

Reads the runs of ``experiments/run.py`` saved in ``results/end_to_end/<case>/seed<s>/`` (one
per case study and seed, with the case study's example data and 1 000 samples) and writes
``results/end_to_end.csv`` and ``experiments_paper/tables/e2e.tex``: how many runs found all three
unknown unknowns, the mean expected cost of the controller without and with the variables
found, and the gain.

Run (after the runs):
    for p in robot consensus fx uav; do for s in 0 1 2 3 4 5 6 7 8 9; do
        python experiments/run.py --problem $p --seed $s \
            --outputfolder experiments_paper/RQ1/results/end_to_end/$p/seed$s
    done; done
    python experiments_paper/RQ1/end_to_end_table.py
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CASES, LABELS, OUTPUT  # noqa: E402

RESULTS = HERE / "results"


def row(case):
    runs = [json.loads(next(seed.iterdir()).joinpath("summary.json").read_text())
            for seed in sorted((RESULTS / "end_to_end" / case).iterdir())]
    without = statistics.mean(r["reward_without_u"] for r in runs)
    with_u = statistics.mean(r["reward_with_u"] for r in runs)
    return {"case": case, "runs": len(runs),
            "all_found": sum(sorted(r["known_variables"]) == sorted(CASES[case]["variables"])
                             for r in runs),
            "cost_without": round(without, 2), "cost_with": round(with_u, 2),
            "gain_percent": round(100 * (without - with_u) / without, 1)}


def latex_table(rows):
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Case study & Without & With & Cost reduction\\", r"\midrule"]
    for r in rows:
        lines.append(f"{LABELS[r['case']]} & "
                     f"{r['cost_without']:.1f} & {r['cost_with']:.1f} & "
                     f"{r['gain_percent']:.1f}\\%\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    (OUTPUT / "tables").mkdir(exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    rows = [row(case) for case in CASES]
    with open(RESULTS / "end_to_end.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "tables" / "e2e.tex").write_text(latex_table(rows))
    for r in rows:
        print(r)
