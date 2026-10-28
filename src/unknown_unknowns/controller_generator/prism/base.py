"""Controller generator that fills a PRISM MDP template from data and verifies it.

The data is the samples from the data generator, ``[({name: value}, x), ...]``. The model
is one of a problem's templates (``<k>unknowns.mdp``), picked by the number of unknowns:

- with no unknowns, the model has a single environment (the original model);
- otherwise, one environment per combination of the unknowns accepted so far (those
  whose distributions are unimodal), with unknown ``i`` playing the role of ``u<i>``.

The probabilities and action costs of the template are estimated from the data, and the
metric is the minimum expected cost (``reward``) that PRISM computes for the model. Each
problem subclasses :class:`PrismController` and sets the class attributes below (see
``examples/robot/controller.py`` and ``examples/consensus/controller.py``).
"""
import itertools
import re
from pathlib import Path

import numpy as np

from ..controller import Controller
from .tools import DEFAULT_PRISM, check


class PrismController(Controller):
    """Generates a controller by filling a PRISM template from the data and verifying it.

    ``templates`` is the problem's folder with ``0unknowns.mdp``, ``1unknowns.mdp``, ... and
    ``property.prop``, the PCTL property whose minimum is the reward (e.g.
    ``examples/robot/prism_templates/``). ``output_folder / f"{name}.mdp"`` holds the last
    filled model.

    Each sample's value is ``{output: cost}`` (e.g. ``{"path1": 26.3, "path2": 24.8}``), and
    the outputs, in their order, fill the template's costs ``c1``, ``c2``, ... (``c1_<bits>``,
    ``c2_<bits>``, ... per environment); this is the order of the names in
    ``context_distributions.txt``. ``action_costs`` (``{A: function(sample value) -> cost}``)
    overrides this when a cost needs a formula.

    The templates must start by setting ``env_id`` (from 0) and change nothing else in that
    step: the policy's decision for each environment is read in the state right after it.
    """
    def __init__(self, output_folder, templates, data=None, unknowns=(), name="model",
                 prism_bin=DEFAULT_PRISM, action_costs=None):
        super().__init__(data)
        self.templates = Path(templates)
        self.unknowns = list(unknowns)
        self.prism_file = Path(output_folder) / f"{name}.mdp"
        self.strategy_file = self.prism_file.with_suffix(".strat")
        self.states_file = self.prism_file.with_suffix(".sta")
        self.prism_bin = Path(prism_bin)
        self.action_costs = action_costs
        self.property_file = self.templates / "property.prop"

    def update_unknowns(self, unknowns):
        """Replace the unknown variables (names) the model is split by."""
        self.unknowns = list(unknowns)

    def constants(self):
        """Values for the template's constants, estimated from ``self.data``.

        ``p<bits>`` is the fraction of samples in that environment and ``c<action>_<bits>``
        the average cost of the action there, where ``<bits>`` gives the value of
        u1, u2, ... in order (e.g. ``p10`` = u1=1, u2=0). With no unknowns there is no
        ``p`` and the costs are just ``c<action>``.
        """
        k = len(self.unknowns)
        action_costs = self.action_costs or self.costs_from_outputs()
        consts = {}
        for bits in itertools.product((0, 1), repeat=k):
            suffix = "".join(map(str, bits))
            where = dict(zip(self.unknowns, bits))
            values = [x for context, x in self.data
                      if all(context[name] == v for name, v in where.items())]
            if k:
                consts[f"p{suffix}"] = len(values) / len(self.data)
            for action, cost in action_costs.items():
                key = f"c{action}_{suffix}" if k else f"c{action}"
                # an environment without samples has probability 0, so its cost doesn't matter
                consts[key] = float(np.mean([cost(x) for x in values])) if len(values) else 0.0
        if k and not np.isclose(sum(v for c, v in consts.items() if c.startswith("p")), 1):
            raise ValueError(f"unknowns {self.unknowns} must only take the values 0 and 1")
        return consts

    def costs_from_outputs(self):
        """``{1: value -> value[first output], 2: value -> value[second output], ...}``, from the
        outputs of the samples in their order."""
        value = self.data[0][1]
        if not isinstance(value, dict):
            raise ValueError("samples have a single value; give action_costs= to say which cost "
                             "it is")
        return {i: (lambda v, o=output: v[o]) for i, output in enumerate(value, 1)}

    def name_unknowns(self, text):
        """Write the actual variables in the template's comments: ``u1=X`` becomes
        ``light=X`` and ``(e.g. light, wet)`` becomes ``(u1=light, u2=wet)``."""
        mapping = ", ".join(f"u{i}={name}" for i, name in enumerate(self.unknowns, 1))
        lines = []
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("//"):
                line = re.sub(r"\bu(\d+)=", lambda m: f"{self.unknowns[int(m[1]) - 1]}=", line)
                line = re.sub(r"\(e\.g\. [^)]*\)", f"({mapping})", line)
            lines.append(line)
        return "".join(lines)

    def generate_prism_file_from_unknowns(self):
        """Write the template for ``len(self.unknowns)`` unknowns, with the constants filled in,
        to ``self.prism_file``."""
        template = self.templates / f"{len(self.unknowns)}unknowns.mdp"
        if not template.exists():
            raise FileNotFoundError(f"no PRISM template for {len(self.unknowns)} unknowns: {template}")
        text = self.name_unknowns(template.read_text())
        for name, value in self.constants().items():
            text, n = re.subn(rf"const double {name};", f"const double {name} = {value!r};", text)
            if n != 1:
                raise ValueError(f"{template.name} has no constant {name!r}")
        self.prism_file.parent.mkdir(parents=True, exist_ok=True)
        self.prism_file.write_text(text)
        return self.prism_file

    def get_prism_verif_res(self):
        """Run PRISM on ``self.prism_file`` with the property in ``self.property_file`` and
        return the result.

        The optimal strategy is written to ``self.strategy_file`` and the states (with the
        variable names) to ``self.states_file``."""
        return check(self.prism_file, self.property_file, prism_bin=self.prism_bin,
                     strategy_file=self.strategy_file, states_file=self.states_file)

    def environments(self):
        """The environments of the model in ``env_id`` order, as ``{unknown: value}`` dicts
        (``[{}]`` with no unknowns)."""
        k = len(self.unknowns)
        return [{name: (env >> i) & 1 for i, name in enumerate(self.unknowns)}
                for env in range(2 ** k)]

    def read_strategy(self):
        """The action the optimal strategy takes in each environment (``env_id`` order),
        read from ``self.strategy_file``, e.g. ``["action2"]`` with no unknowns.

        Lines look like ``(v1,v2,...)=action``. The decision is read in the state right after
        the environment is set: the initial state (the one with ``env_id=0``) with ``env_id``
        changed. The position of ``env_id`` comes from the header of ``self.states_file``."""
        names = self.states_file.read_text().splitlines()[0].strip("()").split(",")
        env_index = names.index("env_id")
        states = []
        for line in self.strategy_file.read_text().splitlines():
            found = re.fullmatch(r"\(([^)]*)\)=(\S*)", line.strip())
            if found:
                values = found[1].split(",")
                env_id = int(values.pop(env_index))
                states.append((env_id, values, found[2]))
        initial = [values for env_id, values, _ in states if env_id == 0]
        if len(initial) != 1:
            raise ValueError(f"expected one state with env_id=0 in {self.strategy_file}, "
                             f"found {len(initial)}")
        actions = {env_id: action for env_id, values, action in states
                   if env_id > 0 and values == initial[0]}
        return [actions[env] for env in sorted(actions)]

    def get_controller(self):
        """Build the model from the data, verify it, and return the policy: the action taken
        in each environment, e.g. ``["action2"]`` with no unknowns, or ``["action2",
        "action1"]`` for u1=0 and u1=1 (see :meth:`environments`).

        The minimum expected cost is stored in ``self.metrics["reward"]``.
        """
        if self.data is None:
            raise ValueError("no data available yet to generate the controller")
        self.generate_prism_file_from_unknowns()
        self.metrics = {"reward": self.get_prism_verif_res()}
        self.controller = self.read_strategy()
        return self.controller

    def get_metrics(self):
        return self.metrics
