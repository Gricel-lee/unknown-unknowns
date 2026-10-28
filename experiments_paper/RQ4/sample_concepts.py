"""RQ4: sample the concepts whose plausibility is checked against the literature.

For each case study, takes the 5 most frequent concepts and 5 of the least frequent ones
(drawn at random, with a fixed seed, among the concepts proposed least often), from
``results/concepts.csv`` (a name without a concept is its own concept). Writes
``results/sampled_concepts.csv`` with an empty ``reference`` column, in which the authors
record a published study showing that the factor affects a similar system (left empty if
none is found). If the file already exists, the references already entered are kept.

Run (after ``analyse.py``):
    python experiments_paper/RQ4/sample_concepts.py
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CASES = ["robot", "consensus", "fx", "uav"]
N = 5
SEED = 0
FIELDS = ["case", "group", "rank", "concept", "count", "llms", "example_names", "reference"]


def main():
    with open(RESULTS / "concepts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out_path = RESULTS / "sampled_concepts.csv"
    old = {}
    if out_path.exists():
        with open(out_path, newline="") as f:
            old = {(r["case"], r["concept"]): r["reference"] for r in csv.DictReader(f)}

    rng = random.Random(SEED)
    sampled = []
    for case in CASES:
        count, llms, names = Counter(), defaultdict(set), defaultdict(list)
        for r in rows:
            if r["case"] != case:
                continue
            concept = r["concept_authors_classified"] or r["name"]
            count[concept] += int(r["count"])
            llms[concept] |= {m.split("(")[0] for m in r["models"].split("; ")}
            names[concept].append(r["name"])
        ranked = sorted(count, key=lambda c: (-count[c], c))
        frequent = ranked[:N]
        rarest = min(count.values())
        rare = sorted(rng.sample(sorted(c for c in count if count[c] == rarest), N))
        for group, concepts in (("most frequent", frequent), ("least frequent", rare)):
            for i, c in enumerate(concepts, 1):
                sampled.append({"case": case, "group": group, "rank": i, "concept": c,
                                "count": count[c], "llms": len(llms[c]),
                                "example_names": "; ".join(names[c][:3]),
                                "reference": old.get((case, c), "")})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(sampled)
    for r in sampled:
        print(f"{r['case']:9} {r['group']:14} {r['count']:>3}  {r['concept']}")


if __name__ == "__main__":
    main()
