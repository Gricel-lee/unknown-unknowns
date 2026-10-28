"""RQ4: ask Gemini for online services that could provide the most frequent concepts.

For each case study, sends ``case_studies_prompts/<case>_prompt.txt``, which asks for services
estimating its two most frequent concepts (``../results/concepts.csv``), once to Gemini (same
model and settings as ``../query.py``). Saves the answer in ``responses/<case>.md`` and one row
per query in ``services.csv`` (case, model that answered, date).

Answers that already exist are skipped. Run (from the repository root, with the Gemini
environment variables of ``../query.py`` set):
    python experiments_paper/RQ4/service_prompt/query_services.py
"""
import csv
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from query import MODELS, ask_gemini  # noqa: E402

PROMPTS = HERE / "case_studies_prompts"
RESPONSES = HERE / "responses"
CASES = ["robot", "consensus", "fx", "uav"]


def main():
    RESPONSES.mkdir(exist_ok=True)
    log = HERE / "services.csv"
    new_log = not log.exists()
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["case", "model_version", "date", "prompt", "response"])
        for case in CASES:
            prompt_file = PROMPTS / f"{case}_prompt.txt"
            out = RESPONSES / f"{case}.md"
            if out.exists():
                continue
            text, version = ask_gemini(prompt_file.read_text(), MODELS["gemini"])
            out.write_text(text)
            writer.writerow([case, version, date.today().isoformat(),
                             str(prompt_file.relative_to(HERE)), str(out.relative_to(HERE))])
            print(f"{case}: {len(text)} characters ({version})")


if __name__ == "__main__":
    main()
