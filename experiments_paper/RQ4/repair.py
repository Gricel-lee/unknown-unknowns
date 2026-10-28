"""RQ4: correct the answers whose JSON is malformed, and report the share of valid answers.

``query.py`` saves ``run<i>.json`` only when the answer contains a valid JSON list. For every
other answer (``run<i>.txt`` without ``run<i>.json``), this script applies syntactic fixes
that do not change the content (missing commas between fields or objects, trailing commas,
comments, strings closed by a typographic quote, single quotes) and, if the result is a valid list of proposals with a name, saves
it as ``run<i>_corrected.json``. Answers that cannot be corrected are discarded, i.e., they
have neither file and are listed as such.

Writes ``results/json_validity.csv``: per LLM, the number of answers, of valid answers, of
corrected answers and of discarded answers.

Run (after ``query.py``):
    python experiments_paper/RQ4/repair.py
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def candidate_text(text):
    """The part of the answer that should be the JSON list (inside a ```json fence, if any)."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("["), text.rfind("]")
    return text[start:end + 1] if start != -1 and end > start else None


def fix(text):
    """Syntactic fixes only: the values are kept as they are."""
    text = re.sub(r"//[^\n]*", "", text)                                  # line comments
    text = re.sub(r",\s*([\]}])", r"\1", text)                            # trailing commas
    text = re.sub(r'(["\d\]}]|true|false|null)\s*\n(\s*")', r"\1,\n\2", text)  # missing comma before a key
    text = re.sub(r"}\s*\n(\s*){", r"},\n\1{", text)                      # missing comma between objects
    return text


def valid(proposals):
    return (isinstance(proposals, list) and proposals
            and all(isinstance(p, dict) and p.get("name") for p in proposals))


def correct(text):
    """The corrected list of proposals, or None if the answer cannot be corrected."""
    body = candidate_text(text)
    if body is None:
        return None
    fixed = fix(body)
    # fallbacks, tried only if needed: a string closed by a typographic quote, single quotes
    closing_quotes = re.sub(r'[“”](\s*[,\n}\]])', r'"\1', fixed)
    for attempt in (fixed, closing_quotes, closing_quotes.replace("'", '"')):
        try:
            proposals = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if valid(proposals):
            return proposals
    return None


def main():
    counts = Counter()
    discarded = []
    for raw in sorted(RESULTS.glob("*/*/run*.txt")):
        llm = raw.parent.name
        counts[llm, "answers"] += 1
        if raw.with_suffix(".json").exists():
            counts[llm, "valid"] += 1
            continue
        proposals = correct(raw.read_text())
        if proposals is None:
            counts[llm, "discarded"] += 1
            discarded.append(str(raw.relative_to(RESULTS)))
            continue
        counts[llm, "corrected"] += 1
        out = raw.with_name(raw.stem + "_corrected.json")
        out.write_text(json.dumps(proposals, indent=2) + "\n")

    llms = sorted({llm for llm, _ in counts})
    with open(RESULTS / "json_validity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["llm", "answers", "valid", "corrected", "discarded", "valid_percent"])
        for llm in llms:
            n = counts[llm, "answers"]
            row = [llm, n, counts[llm, "valid"], counts[llm, "corrected"], counts[llm, "discarded"],
                   round(100 * counts[llm, "valid"] / n, 1)]
            writer.writerow(row)
            print(f"{llm:13} answers {n:3}  valid {row[2]:3} ({row[5]:5.1f}%)  "
                  f"corrected {row[3]:3}  discarded {row[4]:3}")
    for path in discarded:
        print(f"discarded: {path}")


if __name__ == "__main__":
    main()
