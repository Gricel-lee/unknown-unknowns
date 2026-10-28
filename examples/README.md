# Case studies for the unknown-unknowns pipeline

Each case study is a decision problem modelled as a PRISM **MDP**. A hidden environment, made
of three binary context variables, changes the cost of the available actions. The pipeline in
`experiments/run.py` collects logs from the environment, detects multimodal costs
(`unknown_unknowns.detector`), identifies the hidden variables that explain them
(`unknown_unknowns.identifier`), and builds two PRISM controllers: one that ignores the hidden
variables and one that uses the variables found (`unknown_unknowns.controller_generator`). The gap between their expected costs is the gain
from detecting the unknown unknowns.

| Case study | Source | Decision (MDP actions) | Unknown unknowns | Cost per sample (`c1`, `c2`) | PCTL property | Cost without / with the unknowns |
|---|---|---|---|---|---|---|
| `robot` | Robot example | `action1` / `action2`: travel path 1 or path 2 | `light`, `wet`, `busy` | `path1`, `path2`: travel cost of each path | `R{"cost"}min=? [ F "end" ]` | 31.4 / 28.3 |
| `consensus` | PRISM [consensus](http://www.prismmodelchecker.org/casestudies/consensus_prism.php) (`coin2.nm`, [AH90]), simplified | `flip1` / `flip2`: which server flips the next coin of the shared coin | `busy`, `remote`, `congested` | `server1`, `server2`: latency of one flip (ms) | `R{"latency"}min=? [ F "agreed" ]` | 88.0 / 70.0 |
| `fx` | Annotated FXWorkflow model (MonoGenerator, 2017); source to be cited | `marketwatch` / `fundanalysis`: expert mode (market watch, then technical analysis) or normal mode (fundamental analysis) | `volatile`, `news`, `disk_busy` | `technical`, `fundamental`: time of each analysis | `R{"time"}min=? [ F "done" ]` | 46.5 / 41.0 |
| `uav` | PRISM [human-in-the-loop UAV](http://www.prismmodelchecker.org/casestudies/human-uav.php) (`uav-mdp.nm`) | `ask_operator` / `onboard`: who analyses the image at each waypoint | `operator_busy`, `signal_noise`, `low_light` | `operator`, `onboard`: time to analyse one image | `R{"time"}min=? [ F stop ]` | 595.5 / 566.6 |

The last column is the minimum expected cost with the true means of the example data: first
when the hidden variables are ignored (only the average costs are known), then when all three
are known. A run of `run.py` gives values close to these, estimated from the samples.

In every case study, each unknown changes which action is cheaper or how much the actions cost,
and the detector needs all three variables before every cost is unimodal in every bin:

- **robot:** path 2 is cheaper when the ground is dry and path 1 when it is wet.
- **consensus:** `busy` slows server 1, `remote` slows server 2 and `congested` slows both.
- **fx:** fundamental analysis is better only when the market is volatile and there is no news.
  `disk_busy` slows both analyses by the same factor, so it changes the time but not the decision.
- **uav:** the onboard classifier is used when the operator is slow to reach (busy or noisy
  link) and the light is good. In low light the operator is used, unless both problems occur.

## Folder layout

Each `<case>/` folder contains:

- `context_data/context_prior.txt`: the probability of each combination of the three variables.
- `context_data/context_distributions.txt`: the Normal distribution of each cost for each
  combination, in the form `{env:000, path1_mean:26, path1_std:1.0, path2_mean:25, path2_std:1.0}`.
  The names before `_mean` and `_std` become the outputs of each sample (`path1` and `path2`
  here). **Their order sets the costs**: the first output fills `c1` in the templates, the
  second `c2`, so list them in the order of the actions they cost.
- `prism_templates/<k>unknowns.mdp` for k = 0 to 3: the MDP with k hidden variables, with every
  probability and cost left as an undefined constant.
- `prism_templates/property.prop`: the PCTL property whose minimum is the controller's cost
  (the "PCTL property" column of the table).
- `prism_templates/3unknowns_example.mdp`: the 3-unknown model with all
  values filled in, to test in PRISM directly. It gives the "with the unknowns" value of the
  table.

In the combination strings (`000`, `101`, ...), the digits give the values of the variables in
the order in which `run.py` declares them, e.g. `light, wet, busy` for the robot.

## Templates

All templates follow the same pattern:

1. An `environment` module samples `env_id` once, at the start, with probabilities `p<bits>`.
   With no unknowns there is a single environment.
2. The original model then runs, and the controller chooses between the two actions.
3. The reward structure charges `c1_<bits>` for one action and `c2_<bits>` for the other in
   each environment (`c1` and `c2` with no unknowns).

`<bits>` gives the values of `u1, u2, u3` in order, e.g. `p10` is the probability of u1=1, u2=0.
`PrismController` (`src/unknown_unknowns/controller_generator/prism/`)
estimates `p<bits>` as the fraction of samples in each environment and `c<A>_<bits>` as the
average cost there. It fills the template for the number of variables found so far, renames
`u1, u2, ...` to the real variable names in the comments, and runs PRISM. The controller is the
action the optimal strategy takes in each environment, e.g. `["action2"]` with no unknowns.

## Running

From the repository root:

```bash
./.venv/bin/python experiments/run.py --problem robot       # or consensus, fx, uav
```

Each run creates `results/<date>_<time>/` (at the repository root) with the logs, the
detection and identification steps (`detection.csv`, `identification.csv`), the PRISM models
and strategies, the policies (`policy.csv`), the expected costs (`summary.json`) and the data
plots. PRISM is called twice per run (without and with the variables found), which takes a few
seconds for robot, consensus and fx and about 20 seconds for the uav model (about 400,000
states).

## Adding a case study

1. Write the templates `0unknowns.mdp` to `3unknowns.mdp` in `<case>/prism_templates/`, with
   the constant names above, and the property to minimise in `<case>/prism_templates/property.prop`.
2. Add the example data in `<case>/context_data/`.
   List the outputs in the order of the costs `c1`, `c2` they fill.
3. Add the case to `PROBLEMS` in `experiments/run.py` (its variables, folder and plot label).

No code is needed: `PrismController` reads everything else from the templates and the data.
The decision for each environment is read from PRISM's strategy in the state right after
`env_id` is set, so the templates must set `env_id` first and nothing else in that step.
