"""RQ4: recall@k and consistency of the parameters hypothesised by the LLMs.

Reads the proposals of every run (``results/<case>/<llm>/run<i>.json``, or
``run<i>_corrected.json`` written by ``repair.py``) and the concept mapping
``results/concepts.csv``, one row per case study and distinct (normalised) proposal name:

- ``models``: the LLMs that proposed the name, with the number of answers;
- ``RQ1_params_match``: the candidate of Table II that the name matches by the keyword rules in
  ``MATCH`` (empty if none), used for recall@k;
- ``concept_authors_classified``: the concept the name stands for, assigned by the authors to merge names of the
  same concept (e.g. ``rainfall`` and ``precipitation``), used for consistency and plausibility;
  if empty, the normalised name is its own concept.

New names are appended to ``concepts.csv`` on every run, and existing rows are kept, so the
authors' concepts are never overwritten.

Measures, per LLM (mean over the case studies and runs):
- recall@k = |G ∩ L_k| / |G|, for the candidates G of the case study (Table II) and the
  candidates matched by the top-k proposals L_k, k = 5, 10;
- consistency: the mean Jaccard similarity (the table reports diversity = 1 - consistency) J(L, L') = |L ∩ L'| / |L ∪ L'| between the sets
  of concepts of the top-10 proposals of every pair of runs of the LLM on the same case study;
  and the mean Jaccard similarity between the runs of every pair of LLMs.

Writes ``results/rq4_summary.csv`` and ``experiments_paper/tables/rq4.tex``.

Run (after ``query.py`` and ``repair.py``):
    python experiments_paper/RQ4/analyse.py
"""
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent))
from common import OUTPUT  # noqa: E402

CASES = ["robot", "consensus", "fx", "uav"]
# Proprietary model first (size undisclosed), then the open-weight models from largest to smallest.
LLMS = ["gemini", "gemma_large", "llama", "qwen", "gemma_medium", "gemma_small"]
LABELS = {"gemini": "Gemini", "gemma_small": "Gemma 1B", "gemma_medium": "Gemma 4B",
          "gemma_large": "Gemma 27B", "llama": "Llama 8B", "qwen": "Qwen 7B"}
KS = (5, 10)

# Keyword rules matching a proposal name to the candidates of Table II. They are deliberately
# strict: borderline names (e.g. "time of day" for light, "database performance" for a busy
# disk) are left unmatched.
MATCH = {
    "robot": {
        "light": r"(?<!traffic )\blight(ing)?\b|sunlight|illumination|\bdark|shadow|glare"
                 r"|brightness|visib",
        "wet": r"\bwet|moist|\brain|precipitation|\bmud|puddle|\bwater|humidity|\bdew\b"
               r"|irrigation|weather(?! - wind)",
        "busy": r"\bbusy|traffic(?! light)|pedestrian|worker|people|human (activity|presence)"
                r"|tractor|machinery|other (robot|vehicle|equipment)|agricultural equipment",
    },
    "consensus": {
        "busy": r"\bbusy|server load|cpu (load|usage|utili[sz]ation)|system load"
                r"|background (process|task|job)|resource contention|noisy neighbo",
        "remote": r"remote|geograph|\blocation|physical distance|data ?cent(er|re)|region"
                  r"|proximity|round.trip",
        "congested": r"congest|packet loss|(network|link) (load|bandwidth|traffic)"
                     r"|cross.traffic",
    },
    "fx": {
        "volatile": r"volatil",
        "news": r"\bnews|announcement|press release|economic (calendar|news|event|release|data)"
                r"|macroeconomic",
        "disk_busy": r"\bdisk (i/?o|usage|load|contention|busy|activity|latency|queue)"
                     r"|\bi/o\b|storage (performance|latency|i/?o|contention)",
    },
    "uav": {
        "operator_busy": r"operator.*(busy|workload|availab|task load|multi-?task|queue"
                         r"|concurrent|cognitive load)|operator workload",
        "signal_noise": r"signal|radio|\blink\b|interference|connectivity|communication"
                        r"|network (connection|quality)",
        "low_light": r"\blight(ing)?\b|illumination|\bdark|night|sunlight|glare|shadow",
    },
}
CONCEPT_FIELDS = ["case", "name", "count", "RQ1_params_match", "concept_authors_classified", "models"]


def normalise(name):
    name = re.sub(r"\(.*?\)", "", str(name).lower())
    name = re.sub(r"[^a-z0-9/&+ -]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def load_runs():
    """``{(case, llm): [ranked list of normalised names of each run]}``."""
    runs = {}
    for case in CASES:
        for llm in LLMS:
            lists = []
            for raw in sorted((RESULTS / case / llm).glob("run*.txt")):
                for path in (raw.with_suffix(".json"), raw.with_name(raw.stem + "_corrected.json")):
                    if path.exists():
                        proposals = json.loads(path.read_text())
                        ranked = sorted(proposals, key=lambda p: (p.get("id") is None, p.get("id", 0)))
                        lists.append([normalise(p["name"]) for p in ranked if p.get("name")])
                        break
            runs[case, llm] = lists
    return runs


def auto_match(case, name):
    for candidate, pattern in MATCH[case].items():
        if re.search(pattern, name):
            return candidate
    return ""


def update_concepts(runs):
    """Read ``concepts.csv``, add the names not in it yet, write it back; return its rows."""
    path = RESULTS / "concepts.csv"
    rows = {}
    if path.exists():
        with open(path, newline="") as f:
            rows = {(r["case"], r["name"]): r for r in csv.DictReader(f)}
    counts, by_model = {}, {}
    for (case, llm), lists in runs.items():
        for names in lists:
            for name in names:
                counts[case, name] = counts.get((case, name), 0) + 1
                models = by_model.setdefault((case, name), {})
                models[llm] = models.get(llm, 0) + 1
    for (case, name), count in counts.items():
        row = rows.setdefault((case, name), {"case": case, "name": name,
                                             "concept_authors_classified": ""})
        row["count"] = count
        row["models"] = "; ".join(f"{llm}({n})" for llm, n in by_model[case, name].items())
        row["RQ1_params_match"] = auto_match(case, name)
    ordered = sorted(rows.values(), key=lambda r: (CASES.index(r["case"]), -int(r["count"]), r["name"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, CONCEPT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return rows


def match(rows, case, name):
    row = rows[case, name]
    return row["RQ1_params_match"]


def concept(rows, case, name):
    return rows[case, name]["concept_authors_classified"].strip() or name


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(runs, rows):
    summary = []
    for llm in LLMS:
        recall = {k: [] for k in KS}
        within = []
        for case in CASES:
            truth = set(MATCH[case])
            sets = []
            for names in runs[case, llm]:
                for k in KS:
                    found = {match(rows, case, n) for n in names[:k]} - {""}
                    recall[k].append(len(found & truth) / len(truth))
                sets.append({concept(rows, case, n) for n in names[:10]})
            within += [jaccard(a, b) for a, b in combinations(sets, 2)]
        summary.append({"llm": llm, "runs": sum(len(runs[c, llm]) for c in CASES),
                        **{f"recall@{k}": round(mean(recall[k]), 3) for k in KS},
                        "jaccard_within": round(mean(within), 3)})
    between = {}
    for x, y in combinations(LLMS, 2):
        values = []
        for case in CASES:
            sx = [{concept(rows, case, n) for n in names[:10]} for names in runs[case, x]]
            sy = [{concept(rows, case, n) for n in names[:10]} for names in runs[case, y]]
            values += [jaccard(a, b) for a in sx for b in sy]
        between[x, y] = mean(values)
    return summary, between


def latex_table(summary, validity, between):
    """Diversity = 1 - mean Jaccard similarity (the higher, the more different concepts)."""
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"LLM & Valid JSON & recall@5 & recall@10 & Diversity\\", r"\midrule"]
    for row in summary:
        lines.append(f"{LABELS[row['llm']]} & {validity[row['llm']]:.0f}\\% & "
                     f"{row['recall@5']:.2f} & {row['recall@10']:.2f} & "
                     f"{1 - row['jaccard_within']:.2f}\\\\")
    lines += [r"\midrule",
              r"\multicolumn{4}{@{}l}{Diversity between LLMs (mean over all pairs)} & "
              f"{1 - mean(between.values()):.2f}\\\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    runs = load_runs()
    rows = update_concepts(runs)
    summary, between = summarise(runs, rows)
    with open(RESULTS / "json_validity.csv", newline="") as f:
        validity = {r["llm"]: float(r["valid_percent"]) for r in csv.DictReader(f)}
    with open(RESULTS / "rq4_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(summary[0]) + ["valid_json_percent"])
        writer.writeheader()
        writer.writerows({**r, "valid_json_percent": validity[r["llm"]]} for r in summary)
    (OUTPUT / "tables").mkdir(exist_ok=True)
    (OUTPUT / "tables" / "rq4.tex").write_text(latex_table(summary, validity, between))
    for r in summary:
        print(r, f"valid JSON {validity[r['llm']]:.0f}%")
    for (x, y), value in sorted(between.items(), key=lambda kv: -kv[1]):
        print(f"between {x} and {y}: {value:.3f}")
    filled = sum(1 for r in rows.values() if r["concept_authors_classified"].strip())
    print(f"concepts.csv: {len(rows)} names, {filled} with a concept")


if __name__ == "__main__":
    main()
