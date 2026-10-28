# Unknown unknowns

A self-adaptive system decides with a model of itself and its environment. When the
environment depends on a variable that the model does not have (an *unknown unknown*), the
costs the system observes are a mixture of several distributions, and the policy it
synthesises is optimal only for an "average" environment that never occurs. This repository
detects such variables in the system's logs, adds them to a PRISM model of the system, and
synthesises a better controller.

The approach works on the system's logs in four steps:

1. **Monitor.** Collect observations: the values of the logged context variables and the cost
   of each action.
2. **Detect.** Split the logged costs by the variables the model knows, and test every bin
   for unimodality (Hartigan's dip test). A multimodal bin means a variable is missing.
3. **Identify.** Rank the candidate variables by GAIN = MI(O; W | K), the information a
   candidate W adds about the costs O given the known variables K. Keep the candidates whose
   gain is significant (permutation test, Bonferroni-corrected) and removes at least a fraction
   epsilon of the remaining uncertainty H(O | K); add the one with the highest gain. Steps 2-3
   repeat until the costs are unimodal, no candidate is accepted, or after m_max iterations
   (one variable added per iteration). The metaparameters are in `examples/<case>/config.json`.
4. **Synthesise.** Fill a PRISM MDP template with the probabilities and costs estimated for
   each combination of the known variables, and compute the optimal policy with PRISM.

## Repository layout

```
src/unknown_unknowns/      the package
  data_generator/          samples of (context, costs) from a prior and Normal distributions
  detector/                the dip test of every bin of the known variables
  identifier/              which variables explain the costs (Identifier interface): conditional
                           mutual information (gain), Shapley values (SHAP), conditional
                           inference trees, unimodality-based search
  controller_generator/    the Controller interface and the PRISM controller
examples/                  four case studies: robot, consensus, fx, uav (see examples/README.md)
  <case>/context_data/     prior and cost distributions of the example data
  <case>/prism_templates/  MDP templates for 0-3 unknowns, and the property to optimise
experiments/run.py         the pipeline on one case study (monitor, detect, identify, synthesise)
experiments_paper/         the paper's experiments (RQ1 effectiveness, RQ2 latency,
                           RQ3 scalability); results go to experiments_paper/RQ*/results/,
                           LaTeX tables and figures to experiments_paper/tables/ and figures/
tests/                     unit tests (pytest)
```

## Setup

Requires Python 3.9+ and [PRISM](https://www.prismmodelchecker.org) (by default
`~/prism/bin/prism`; set `$PRISM` or pass `--prism` otherwise).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . matplotlib pytest
pytest
```

## Running

One case study, end to end (results in `results/<date>_<time>/`):

```bash
python experiments/run.py --problem robot                     # or consensus, fx, uav
python experiments/run.py --problem robot --identifier shap   # or mi (default), ctree, unimodality
```

The paper's experiments (LaTeX tables and figures are written to `experiments_paper/tables/`
and `experiments_paper/figures/`):

```bash
python experiments_paper/RQ1/generate_variants.py
python experiments_paper/RQ1/run_rq1.py --seeds 30     # ~10 min on 6 cores
python experiments_paper/RQ2/run_rq2.py                # uses the RQ1 runs
python experiments_paper/RQ3/run_rq3.py                # ~10 min (mostly the UAV model)
```
