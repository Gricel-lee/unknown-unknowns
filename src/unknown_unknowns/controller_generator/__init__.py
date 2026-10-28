"""Generate controllers from data.

Typical use::

    from unknown_unknowns.controller_generator import PrismController

    controller = PrismController(output_folder, templates="examples/robot/prism_templates",
                                 data=samples, unknowns=["light"])
    policy = controller.get_controller()          # e.g. ["action2", "action1"]
    reward = controller.get_metrics()["reward"]
"""
from .controller import Controller
from .prism import DEFAULT_PRISM, PrismController

__all__ = ["Controller", "PrismController", "DEFAULT_PRISM"]
