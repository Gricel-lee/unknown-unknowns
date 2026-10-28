"""RQ4 (hypothesising unknown unknowns): query the LLMs with the case-study prompts.

Sends every prompt in ``case_studies_prompts/<case>_prompt.txt`` ``--runs`` times to each LLM
and saves, for run i, the raw answer (``results/<case>/<llm>/run<i>.txt``) and the JSON list
extracted from it (``run<i>.json``, absent if the answer contains no valid JSON list). Every
query is a new, independent request (no conversation history), and one row per run is
appended to ``results/runs.csv`` with the model that served it and the date.

LLMs (the model names are in ``MODELS``; check them before running):
- gemini: Google Gemini, either through the Gemini API (key in GEMINI_API_KEY) or through
  Vertex AI with a Google Cloud service account (GOOGLE_GENAI_USE_VERTEXAI=true,
  GOOGLE_APPLICATION_CREDENTIALS=<key file>, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION);
- gemma_small, gemma_medium, gemma_large, llama, qwen: open-weight models served locally by
  Ollama (https://ollama.com); pull them first, e.g. ``ollama pull gemma3:1b``.

Runs that already have a ``.txt`` file are skipped, so the script can be stopped and resumed.
If an Ollama answer takes longer than ``--timeout`` seconds (default 1200, i.e., 20 minutes),
the remaining runs of that LLM are skipped.

Run (from the repository root):
    pip install google-genai
    export the Gemini variables above
    python experiments_paper/RQ4/query.py                      # all LLMs, cases and runs
    python experiments_paper/RQ4/query.py --llms gemini --cases robot --runs 1
"""
import argparse
import csv
import json
import os
import re
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "case_studies_prompts"
RESULTS = HERE / "results"
CASES = ["robot", "consensus", "fx", "uav"]

# Model of each LLM. The Ollama tags are examples: use the sizes reported in the paper.
MODELS = {
    "gemini": "gemini-3.5-flash",
    "gemma_small": "gemma3:1b",
    "gemma_medium": "gemma3:4b",
    "gemma_large": "gemma3:27b",
    "llama": "llama3.1:8b",
    "qwen": "qwen2.5:7b",
}
OLLAMA_URL = "http://localhost:11434/api/chat"
FIELDS = ["case", "llm", "run", "model_version", "interface", "date", "file"]


def ask_gemini(prompt, model):
    from google import genai
    client = genai.Client()   # Gemini API or Vertex AI, from the environment variables above
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text, getattr(response, "model_version", None) or model


def ask_ollama(prompt, model, timeout):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(OLLAMA_URL, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        answer = json.load(reply)
    return answer["message"]["content"], answer.get("model", model)


def ask(llm, prompt, timeout):
    """(answer, model version, interface) of one query."""
    model = MODELS[llm]
    if llm == "gemini":
        return (*ask_gemini(prompt, model), "vertex-ai" if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true" else "gemini-api")
    return (*ask_ollama(prompt, model, timeout), "ollama")


def extract_json(text):
    """The first JSON list in the answer (inside a ```json fence or not), or None."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def log_run(row):
    path = RESULTS / "runs.csv"
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llms", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1200,
                        help="seconds after which an Ollama LLM is given up (default 20 minutes)")
    args = parser.parse_args()

    too_slow = set()
    for case in args.cases:
        prompt = (PROMPTS / f"{case}_prompt.txt").read_text()
        for llm in args.llms:
            if llm in too_slow:
                continue
            folder = RESULTS / case / llm
            folder.mkdir(parents=True, exist_ok=True)
            for run in range(1, args.runs + 1):
                raw = folder / f"run{run}.txt"
                if raw.exists():
                    continue
                try:
                    text, version, interface = ask(llm, prompt, args.timeout)
                except TimeoutError:
                    print(f"{case}/{llm}/run{run}: no answer within {args.timeout:.0f} s; skipping {llm}")
                    too_slow.add(llm)
                    break
                except Exception as error:   # report and go on with the next query
                    print(f"{case}/{llm}/run{run}: FAILED ({error})")
                    continue
                raw.write_text(text)
                proposals = extract_json(text)
                if proposals is not None:
                    (folder / f"run{run}.json").write_text(json.dumps(proposals, indent=2) + "\n")
                log_run({"case": case, "llm": llm, "run": run, "model_version": version,
                         "interface": interface, "date": date.today().isoformat(),
                         "file": str(raw.relative_to(RESULTS))})
                status = f"{len(proposals)} proposals" if proposals is not None else "NO JSON"
                print(f"{case}/{llm}/run{run}: {status} ({version})")


if __name__ == "__main__":
    main()
