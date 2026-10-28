"""Run: python experiments/run.py [--problem {robot,consensus,fx,uav}]
                        [--prior PRIOR.txt] [--distributions DISTRIBUTIONS.txt] [--templates DIR]
                        [--identifier {mi,shap,ctree,unimodality}]
                        [--config CONFIG.json] [--samples N] [--alpha A] [--epsilon E]
                        [--m_max M] [--n_min N]
                        [--permutations N] [--bins N] [--ctree_alpha A] [--seed S]
                        [--outputfolder DIR] [--prism PRISM_BIN]

Discover the unknown unknowns of a system from its logs and synthesise a controller with them.

1. Monitor: collect ``--samples`` observations of the system (simulated with the data
   generator): the values of the logged context variables and the cost of each action.
2. Detect: split the costs by the known variables and test every bin for multimodality.
3. Identify: ask the identifier chosen with ``--identifier`` which candidate variables to add
   to the known ones:
   - ``mi`` (default): rank the candidates by GAIN(D, W, K) = MI(O; W | K); keep those whose gain
     is significant at level alpha / |C| and removes at least a fraction epsilon of the
     remaining uncertainty, GAIN / H(O | K) >= epsilon; add the one with the highest gain;
   - ``shap``: the same with the candidates' Shapley importance (mean |SHAP value|) and the
     share of the remaining variation it accounts for;
   - ``ctree``: the candidates a conditional inference tree splits on;
   - ``unimodality``: the first combination of candidates that makes every bin unimodal.
   Steps 2-3 are repeated until the costs are unimodal, no candidate is accepted, or
   ``m_max`` iterations are done.
4. Synthesise: fill the problem's PRISM template with the known variables, and with none for
   comparison, and compute the optimal policy and its expected cost with PRISM.

The metaparameters n_min, alpha, m_max and epsilon are read from
``examples/<problem>/config.json`` (each with its description); the command-line options
override them. The analysis uses only the logs collected in step 1. Each run creates
``<outputfolder>/<date>_<time>/`` with:

- ``data.csv``: the logs;
- ``detection.csv``: the dip test of every bin and output, at every iteration;
- ``identification.csv``: the identifier's ranking of the candidates, at every iteration;
- ``model_without_u.mdp`` / ``model_with_u.mdp`` (and their PRISM strategy and states files):
  the PRISM models;
- ``policy.csv``: the action chosen in each environment by each model's optimal policy;
- ``summary.json``: the known variables, why the identification stopped, and the expected costs;
- ``data_distrib_env<i>.png`` / ``data_hist_env<i>.png``: the costs of each action.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from unknown_unknowns.controller_generator import DEFAULT_PRISM, PrismController
from unknown_unknowns.data_generator import DataGenerator, Variable, output_value
from unknown_unknowns.detector import all_unimodal, check_partition
from unknown_unknowns.identifier import IDENTIFIERS

HERE = Path(__file__).parent
ROOT = HERE.parent   # repository root: holds examples/ and results/
EXAMPLES = ROOT / "examples"   # <problem>/context_data/ and <problem>/prism_templates/
OUTPUT_FOLDER = ROOT / "results"

# Each problem: its context variables, its folder in examples/ (with context_data/ and
# prism_templates/), and the x label of its plots ({i} = 1, 2: the output's number,
# {output} = its name).
PROBLEMS = {
    "robot": {
        "variables": [Variable("light", [0, 1]), Variable("wet", [0, 1]), Variable("busy", [0, 1])],
        "examples": EXAMPLES / "robot",
        "xlabel": "cost travel path_{i}",
    },
    "consensus": {
        "variables": [Variable("busy", [0, 1]), Variable("remote", [0, 1]),
                      Variable("congested", [0, 1])],
        "examples": EXAMPLES / "consensus",
        "xlabel": "latency server_{i} (ms)",
    },
    "fx": {
        "variables": [Variable("volatile", [0, 1]), Variable("news", [0, 1]),
                      Variable("disk_busy", [0, 1])],
        "examples": EXAMPLES / "fx",
        "xlabel": "time {output} analysis",
    },
    "uav": {
        "variables": [Variable("operator_busy", [0, 1]), Variable("signal_noise", [0, 1]),
                      Variable("low_light", [0, 1])],
        "examples": EXAMPLES / "uav",
        "xlabel": "image analysis time ({output})",
    },
}

DETECTION_FIELDS = ["iteration", "known_vars", "bin", "output", "unimodal", "p_value"]
CONFIG_KEYS = ["n_min", "alpha", "m_max", "epsilon"]


def load_config(path):
    """The metaparameters ``{name: value}`` of a config.json file (see examples/<problem>/)."""
    config = json.loads(Path(path).read_text())
    return {k: config[k]["value"] for k in CONFIG_KEYS if k in config}


# =============================================================================================
# Output files
# =============================================================================================

def create_run_folder(output_folder=OUTPUT_FOLDER):
    """Create and return ``<output_folder>/<date>_<time>/``."""
    folder = Path(output_folder) / f"{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_csv(path, fields, rows):
    """Write ``rows`` (dicts) to ``path``; ``fields=None`` uses every key, in order of appearance."""
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_data(samples, names, outputs, folder):
    """Write the logs to ``data.csv``, one row per sample, with a column per variable and per
    output (e.g. ``light,wet,busy,path1,path2``)."""
    data_file = folder / "data.csv"
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*names, *(o or "value" for o in outputs)])
        writer.writerows([*(context[n] for n in names), *(output_value(value, o) for o in outputs)]
                         for context, value in samples)
    return data_file


def save_plots(gen, samples, outputs, folder, xlabel="cost travel path_{i}"):
    """For each output, save a histogram of all its values (``data_distrib_env<i>.png``) and a
    ridgeline histogram split by every variable (``data_hist_env<i>.png``), with x label
    ``xlabel`` (``{i}`` = the output's number, ``{output}`` = its name). Returns the paths."""
    import matplotlib.pyplot as plt

    paths = []
    for i, output in enumerate(outputs, 1):
        suffix = f"_env{i}" if output else ""
        label = xlabel.format(i=i, output=output) if output else "environment value"
        for name, ax in [
                ("data_distrib", gen.plot_single_distribution(samples=samples, output=output,
                                                              xlabel=label)),
                ("data_hist", gen.plot_ridgeline_hist(samples=samples, output=output,
                                                      xlabel=label))]:
            path = folder / f"{name}{suffix}.png"
            ax.figure.tight_layout()
            ax.figure.savefig(path, dpi=150)
            plt.close(ax.figure)
            paths.append(path)
    return paths


def policy_rows(name, controller):
    """One row per environment of ``controller``: the action of its optimal policy there."""
    return [{"model": name,
             "environment": ", ".join(f"{v}={x}" for v, x in env.items()) or "(all)",
             "action": action}
            for env, action in zip(controller.environments(), controller.controller)]


# =============================================================================================
# Detection and identification
# =============================================================================================

def detection_rows(iteration, known, results):
    return [{"iteration": iteration, "known_vars": " ".join(v.name for v in known),
             "bin": ", ".join(f"{v}={x}" for v, x in where.items()) or "(all)",
             "output": output, "unimodal": unimodal,
             "p_value": None if p is None else round(p, 4)}
            for where, output, unimodal, p in results]


def identification_rows(iteration, known, ranking, selected):
    return [{"iteration": iteration, "known_vars": " ".join(v.name for v in known), **row,
             "selected": bool(row["accepted"]) and set(row["candidate"].split()) <= set(selected)}
            for row in ranking]


def discover(samples, variables, outputs, identifier, alpha=0.05, m_max=3, n_min=10, known=()):
    """Detect and identify the unknown unknowns in the logs ``samples`` (Algorithm 1).

    Detects multimodality in the bins of the known variables (bins with fewer than ``n_min``
    samples are not tested); while some bin is multimodal and at most ``m_max`` iterations have
    been done, asks ``identifier`` (an :class:`~unknown_unknowns.identifier.Identifier`) which
    candidates (the variables not known yet) to add. Stops when the costs are unimodal, no
    candidate is accepted, no candidate is left, or after ``m_max`` iterations.

    ``known`` are the variables already known (none by default). Returns ``(known,
    stop_reason, detection_log, identification_log)``, with ``known`` the known variables
    followed by the added ones, in the order they were added.
    """
    known, detection_log, identification_log = list(known), [], []
    m = 1

    # ----- Detect: multimodality in the bins of the known variables -----
    results = check_partition(samples, known, alpha, outputs, n_min)
    detection_log += detection_rows(m, known, results)

    while m <= m_max and not all_unimodal(results):
        # ----- Identify: the candidates to add -----
        candidates = [v for v in variables if v not in known]
        if not candidates:
            break
        selected, ranking = identifier.identify(samples, [v.name for v in known],
                                                [v.name for v in candidates], outputs)
        identification_log += identification_rows(m, known, ranking, selected)
        if not selected:
            break
        known += [v for name in selected for v in candidates if v.name == name]
        m += 1

        # ----- Detect again, with the new known variables -----
        results = check_partition(samples, known, alpha, outputs, n_min)
        detection_log += detection_rows(m, known, results)

    # ----- Why the loop stopped -----
    if all_unimodal(results):
        stop_reason = "the costs are unimodal in every bin of the known variables"
    elif m > m_max:
        stop_reason = f"m_max reached ({m_max} iterations); some bin is still multimodal"
    elif len(known) == len(variables):
        stop_reason = "no candidate variable left"
    else:
        stop_reason = "no candidate is accepted by the identifier"
    return known, stop_reason, detection_log, identification_log


# =============================================================================================
# The pipeline
# =============================================================================================

def make_identifier(name, alpha=0.05, epsilon=0.05, bins=10, permutations=200, ctree_alpha=0.01,
                    seed=0):
    """The identifier called ``name`` (one of ``IDENTIFIERS``), with its parameters."""
    if name == "mi":
        return IDENTIFIERS["mi"](alpha, epsilon, bins, permutations, seed)
    if name == "shap":
        return IDENTIFIERS["shap"](alpha, epsilon, permutations, seed)
    if name == "ctree":
        return IDENTIFIERS["ctree"](ctree_alpha)
    return IDENTIFIERS[name](alpha)


def run(gen, identifier, n_samples=1000, alpha=0.05, m_max=3, n_min=10,
        output_folder=OUTPUT_FOLDER,
        prism_bin=DEFAULT_PRISM, templates=EXAMPLES / "robot" / "prism_templates",
        xlabel="cost travel path_{i}", settings=None):
    """Monitor, detect, identify and synthesise (see the module docstring). Returns the summary."""
    folder = create_run_folder(output_folder)
    outputs = gen.outputs or [None]

    # ----- Monitor: collect the logs of the system -----
    samples = gen.sample_many(n_samples)

    # ----- Detect and identify the unknown unknowns in the logs -----
    known, stop_reason, detection_log, identification_log = discover(
        samples, gen.variables, outputs, identifier, alpha, m_max, n_min)
    known_names = [v.name for v in known]
    for row in identification_log:
        if row["selected"]:
            print(f"iteration {row['iteration']}: adding {row['candidate']} ({identifier.name})")
    print(f"Known variables: {known_names} ({stop_reason})")

    # ----- Synthesise: PRISM models and optimal policies, without and with the known variables -----
    without_u = PrismController(folder, templates, data=samples, name="model_without_u",
                                prism_bin=prism_bin)
    with_u = PrismController(folder, templates, data=samples, unknowns=known_names,
                             name="model_with_u", prism_bin=prism_bin)
    without_u.get_controller()
    with_u.get_controller()
    reward_without, reward_with = without_u.get_metrics()["reward"], with_u.get_metrics()["reward"]
    print(f"Expected cost: {reward_without:.4g} without the unknown unknowns, "
          f"{reward_with:.4g} with them")

    # ----- Save the outputs -----
    summary = {**(settings or {}), "identifier": identifier.name, "samples": n_samples,
               "alpha": alpha, "m_max": m_max, "n_min": n_min,
               "known_variables": known_names, "stop_reason": stop_reason,
               "reward_without_u": reward_without, "reward_with_u": reward_with,
               "policy_without_u": without_u.controller, "policy_with_u": with_u.controller}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(folder / "detection.csv", DETECTION_FIELDS, detection_log)
    write_csv(folder / "identification.csv", None, identification_log)
    write_csv(folder / "policy.csv", ["model", "environment", "action"],
              policy_rows("without_u", without_u) + policy_rows("with_u", with_u))
    save_data(samples, gen.names, outputs, folder)
    save_plots(gen, samples, outputs, folder, xlabel)
    print(f"Results saved to {folder}")
    return summary


if __name__ == "__main__":

    # Examples (from the repository root):
    # ./.venv/bin/python ./experiments/run.py --problem robot
    # ./.venv/bin/python ./experiments/run.py --problem consensus --samples 2000 --epsilon 0.1
    # ./.venv/bin/python ./experiments/run.py --problem fx --identifier shap
    # ./.venv/bin/python ./experiments/run.py --problem uav      (PRISM takes ~10 s for this model)

    parser = argparse.ArgumentParser(
        description="Discover the unknown unknowns of a system from its logs.")
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="robot",
                        help="which problem to run (default: %(default)s)")
    parser.add_argument("--prior", type=Path,
                        help="txt file with the probability of each combination "
                             "(default: examples/<problem>/context_data/context_prior.txt)")
    parser.add_argument("--distributions", type=Path,
                        help="txt file with the mean and std of each output per combination "
                             "(default: examples/<problem>/context_data/context_distributions.txt)")
    parser.add_argument("--templates", type=Path,
                        help="folder with the PRISM templates <k>unknowns.mdp and property.prop "
                             "(default: examples/<problem>/prism_templates)")
    parser.add_argument("--identifier", choices=sorted(IDENTIFIERS), default="mi",
                        help="how candidates are identified: mi (conditional mutual information), "
                             "shap (Shapley values), ctree (conditional inference tree) or "
                             "unimodality (combinations that make every bin unimodal) "
                             "(default: %(default)s)")
    parser.add_argument("--config", type=Path,
                        help="json file with the metaparameters n_min, alpha, m_max and epsilon "
                             "(default: examples/<problem>/config.json)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="number of observations in the logs (default: %(default)s)")
    parser.add_argument("--alpha", type=float,
                        help="significance level of the dip test and of the gain "
                             "(Bonferroni-corrected) (default: from the config file)")
    parser.add_argument("--epsilon", type=float,
                        help="minimum fraction of the remaining uncertainty H(O | K) that a "
                             "candidate must remove (default: from the config file)")
    parser.add_argument("--m_max", type=int,
                        help="maximum number of iterations of the identification loop, each "
                             "adding at most one variable (default: from the config file)")
    parser.add_argument("--n_min", type=int,
                        help="minimum number of samples for a bin to be dip-tested "
                             "(default: from the config file)")
    parser.add_argument("--permutations", type=int, default=200,
                        help="permutations of the gain's permutation test (default: %(default)s)")
    parser.add_argument("--bins", type=int, default=10,
                        help="equal-frequency bins used to estimate the cost distributions "
                             "(default: %(default)s)")
    parser.add_argument("--ctree_alpha", type=float, default=0.01,
                        help="significance level of the tree's splits, for --identifier ctree "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the data generator (default: %(default)s)")
    parser.add_argument("--outputfolder", type=Path, default=OUTPUT_FOLDER,
                        help="folder in which each run creates its <date>_<time> results folder "
                             "(default: %(default)s)")
    parser.add_argument("--prism", type=Path, default=DEFAULT_PRISM,
                        help="PRISM executable (default: %(default)s)")
    args = parser.parse_args()
    problem = PROBLEMS[args.problem]
    config = load_config(args.config or problem["examples"] / "config.json")
    for key in CONFIG_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, config[key])

    gen = DataGenerator(
        variables=problem["variables"],
        prior=args.prior or problem["examples"] / "context_data" / "context_prior.txt",
        distributions=(args.distributions
                       or problem["examples"] / "context_data" / "context_distributions.txt"),
        seed=args.seed,
    )
    identifier = make_identifier(args.identifier, args.alpha, args.epsilon, args.bins,
                                 args.permutations, args.ctree_alpha, args.seed)
    run(gen, identifier, n_samples=args.samples, alpha=args.alpha, m_max=args.m_max,
        n_min=args.n_min, output_folder=args.outputfolder, prism_bin=args.prism,
        templates=args.templates or problem["examples"] / "prism_templates",
        xlabel=problem["xlabel"],
        settings={"problem": args.problem, "epsilon": args.epsilon, "bins": args.bins,
                  "permutations": args.permutations, "ctree_alpha": args.ctree_alpha,
                  "seed": args.seed})
