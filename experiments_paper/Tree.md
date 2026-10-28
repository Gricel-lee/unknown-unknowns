# Conditional inference trees: how the significance level causes false positives

The `ctree` identifier (`src/unknown_unknowns/identifier/ctree.py`, after Hothorn, Hornik &
Zeileis, 2006) grows a tree of the costs on the logged context variables. In every node it
tests, for each variable, whether the costs (both actions' costs, as one vector) are
independent of it; it corrects the p-values for the number of variables tested; and it splits
on the most significant variable only if its corrected p-value is below the significance level
**α**. The variables the tree splits on anywhere are the ones added to the model.

α is the metaparameter that trades **false positives** (a distractor, i.e. a logged variable
with no effect, is added) against **misses** (a relevant variable is not added). A false
positive is not harmful for the controller (by the monotonicity theorem it cannot make it
worse), but it enlarges the PRISM model for nothing, so its rate is worth reporting.

There are three distinct ways in which α leads to false positives.

## 1. The nominal error rate of each tree

Each node's test is a family of tests at level α (Bonferroni over the variables). When no
variable has an effect, the root node therefore splits by chance in about a fraction α of the
runs, and a distractor is added. With relevant variables present, a distractor can also enter
in a node below them.

Measured on the u0 variants (no variable has an effect) and the hard variants (3σ) of the RQ1
examples, 4 case studies × 4 variants × 5 seeds, 1 000 samples per run:

| Identifier | Exactly right | False positive | Missed |
|---|---|---|---|
| ctree, α = 0.05 | 72/80 | 8 | 0 |
| ctree, α = 0.05, depth correction | 74/80 | 6 | 0 |
| ctree, α = 0.01 | **80/80** | **0** | 0 |
| ctree, α = 0.01, 500 samples | 80/80 | 0 | 0 |
| conditional MI with permutation test (for comparison) | 79/80 | 1 | 0 |

At α = 0.05, the false-positive rate (6-8 of 80, i.e. 7.5-10%) is close to the nominal 5% per
run, plus a little from the deep nodes (see 2). Lowering α to 0.01 removed every false positive
without causing a single miss: the relevant variables' p-values were far below either level
(effects of 3σ give p-values around 1e-16 with 500-1 000 samples). The conditional-MI
identifier was stricter at α = 0.05 because it corrects for 7 combinations (α/7) instead of 3
variables (α/3), and its permutation p-values cannot go below 1/201.

## 2. Many nodes: noise splits deep in the tree

The original method tests every node at level α. A grown tree has many nodes, so the chance
that *some* deep node splits on noise adds up with the size of the tree. For prediction this is
harmless (the split changes the prediction little), but for identification it adds an
irrelevant variable. Example (robot, u1 hard variant, seed 3, α = 0.05): the tree correctly
splits on `wet`, then splits the `wet = 1` branch on the distractor `busy` with p = 0.022,
although the two leaves' mean costs differ by only 0.3 (22.81 vs 23.11).

`ctree(..., adjust_depth=True)` (the default) divides α by the number of nodes that can be
tested at a node's depth (2^depth for binary variables). It reduced the false positives above
from 8 to 6; the remaining ones are the nominal rate of 1. When no variable has an effect, only
the root is ever tested, so the correction makes no difference there.

## 3. Repeated checks: every check is a new chance

In a monitoring loop that re-grows the tree from scratch at every check (as in the measurements
below, where identification runs at every check), a run with 26 checks
the tree is tested 26 times on (overlapping) data, and the chance that *some* check adds a
distractor is much larger than α. Because the tree is re-grown each time, such a distractor is
usually removed again at a later check (a *transient* false positive). Example (robot u0,
α = 0.01): `busy` was added at 780 samples, pruned at 800, added at 820 and pruned at 860; the
run ended with no variable.

Measured on the u0 variants, 4 case studies × 10 seeds, checks every 20 samples from 500 to
1 000 (26 checks per run):

| α | Runs with a false positive at some check | ... at the last check | Checks with a false positive |
|---|---|---|---|
| 0.05 | 5/40 | 4/40 | 65/1 040 (6.2%) |
| 0.01 | 3/40 | **0/40** | 19/1 040 (1.8%) |

The share of checks with a false positive is close to α (6.2% for 0.05, 1.8% for 0.01), as
expected for a test at that level. What a run *ends* with depends on how persistent the chance
association is: at α = 0.01 every transient false positive was gone by the last check.
Growing the tree only when the dip test finds a multimodal bin tests far less often, and so
avoids most transient false positives.

## Suggestions for the evaluation section

- **Sweep α** (e.g. 0.001, 0.01, 0.05, 0.1) on the RQ1 variants and report false positives
  (u0 and distractors in u1-u3) against misses (hard variants): this shows the trade-off, and
  that a stricter α costs little detection power when effects are large.
- **Separate transient from final false positives**, i.e. a distractor added at some check vs
  one present at the end of the run; the first measures the cost of repeated testing, the second
  the model actually used by the controller.
- **Compare the triggers**: identifying only after the dip test finds multimodality tests rarely
  (few chances for false positives, but it misses effects that do not make the costs
  multimodal); identifying at every check tests more often.
- **Compare the identifiers** on the same runs: dip-based search, conditional MI with
  permutation tests, and ctree (which is also about 80 times faster: about 5 ms per tree against
  400 ms per MI ranking with 1 000 samples).

The numbers above come from short ad-hoc runs (seeds 0-4 or 0-9) made while developing the
identifier; a dedicated script in `experiments_paper/` would make them reproducible.
