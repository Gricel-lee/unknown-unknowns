"""Running PRISM from Python."""
import os
import re
import subprocess
from pathlib import Path

# The PRISM executable: $PRISM if set, otherwise ~/prism/bin/prism.
DEFAULT_PRISM = Path(os.environ.get("PRISM", Path.home() / "prism" / "bin" / "prism"))


def run_prism(args, prism_bin=DEFAULT_PRISM, cwd=None):
    """Run PRISM with ``args`` and return its standard output; raise if PRISM reports an error."""
    args = [str(a) for a in args]
    out = subprocess.run([str(prism_bin), *args], cwd=cwd, capture_output=True, text=True)
    if "Error" in out.stdout or out.returncode != 0:
        errors = [line for line in out.stdout.splitlines() if "rror" in line]
        raise RuntimeError(f"PRISM failed: {' '.join(args)}: {errors} {out.stderr[-500:]}")
    return out.stdout


def prism_result(stdout):
    """The (last) ``Result:`` value in PRISM's output."""
    found = re.findall(r"^Result:\s*([-0-9.eE]+)", stdout, flags=re.M)
    if not found:
        raise RuntimeError("no result in PRISM output:\n" + stdout)
    return float(found[-1])


def check(model, prop, consts=None, prism_bin=DEFAULT_PRISM, strategy_file=None, states_file=None):
    """Model-check a property on ``model`` and return the result.

    ``prop`` is the property itself (``'R{"cost"}min=? [ F "end" ]'``) or the path of a
    properties file, whose first property is checked. ``consts`` (e.g. ``{"p0": 0.3}``)
    gives values for undefined constants of the model.
    With ``strategy_file``, PRISM also writes the optimal strategy there, one
    ``(state variables)=action`` line per reachable state. With ``states_file``, it writes the
    states, starting with a ``(variable names)`` header.
    """
    if isinstance(prop, Path):
        args = [model, prop, "-prop", "1"]
    else:
        args = [model, "-pf", prop]
    args += ["-epsilon", "1e-10"]   # tight precision for iterative methods
    if consts:
        args += ["-const", ",".join(f"{k}={v!r}" for k, v in consts.items())]
    if strategy_file:
        args += ["-exportstrat", f"{strategy_file}:type=actions"]
    if states_file:
        args += ["-exportstates", states_file]
    return prism_result(run_prism(args, prism_bin))
