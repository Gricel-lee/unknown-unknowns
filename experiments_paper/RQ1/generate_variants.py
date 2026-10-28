"""Write the RQ1 variants of the four case studies.

For each case study, ``examples/<case>/<variant>/context_data/`` gets a prior and a
``context_distributions.txt`` over the case study's three variables and the two irrelevant ones
``z1``, ``z2`` (``common.IRRELEVANT``: uniform, independent, no effect), in which only some of
the three variables affect the costs:

- ``u0``: no variable has an effect (all three are distractors);
- ``u<k>_d<d>``: k relevant variables (see ``common.RELEVANT``), each shifting the costs it
  affects by ``common.SEPARATION[d]`` standard deviations.

All five variables are logged in every variant, so the irrelevant ones are distractors.

Run: python experiments_paper/RQ1/generate_variants.py
"""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (CASES, EXAMPLES, IRRELEVANT, SEPARATION,  # noqa: E402
                    relevant_variables, variant_name, variants)

OUT = HERE / "examples"


def write_prior(source, target, names):
    """The case study's prior, times a uniform and independent prior of the irrelevant variables."""
    prior = {}
    for line in source.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            combo, p = (x.strip() for x in line.split("="))
            prior[combo] = float(p)
    lines = [f"# Probability of each ({', '.join(names + IRRELEVANT)}) combination; "
             "must add up to 1."]
    for combo, p in prior.items():
        for extra in itertools.product("01", repeat=len(IRRELEVANT)):
            lines.append(f"{combo}{''.join(extra)} = {p / 2 ** len(IRRELEVANT):.8g}")
    target.write_text("\n".join(lines) + "\n")


def write_variant(case, n_unknowns, difficulty):
    spec = CASES[case]
    relevant = relevant_variables(case, n_unknowns)
    folder = OUT / case / variant_name(n_unknowns, difficulty) / "context_data"
    folder.mkdir(parents=True, exist_ok=True)
    write_prior(EXAMPLES / case / "context_data" / "context_prior.txt",
                folder / "context_prior.txt", spec["variables"])

    names = spec["variables"] + IRRELEVANT
    shift = SEPARATION[difficulty] if difficulty else 0.0
    header = [f"# RQ1 variant of {case}: {n_unknowns} relevant variable(s) {relevant}"
              + (f", difficulty {difficulty} (shift = {shift} std)" if difficulty else ""),
              f"# Distractors (no effect): {[v for v in names if v not in relevant]}",
              "# Format: {env:<" + ",".join(names) + ">, <output>_mean:<m>, <output>_std:<s>, ...}"]
    lines = []
    for combo in itertools.product((0, 1), repeat=len(names)):   # the z's have no effect
        values = dict(zip(names, combo))
        fields = [f"env:{''.join(map(str, combo))}"]
        for output in spec["outputs"]:
            mean, std = spec["base"][output]
            mean += sum(spec["effect"][v][output] * shift * std for v in relevant if values[v])
            fields += [f"{output}_mean:{round(mean, 4)}", f"{output}_std:{std}"]
        lines.append("{" + ", ".join(fields) + "}")
    (folder / "context_distributions.txt").write_text("\n".join(header + lines) + "\n")
    return folder


if __name__ == "__main__":
    for case in CASES:
        for n, d in variants():
            write_variant(case, n, d)
    print(f"Wrote {len(CASES) * len(list(variants()))} variants in {OUT}")
