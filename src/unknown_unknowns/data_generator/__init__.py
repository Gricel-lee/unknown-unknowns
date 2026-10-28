"""Sample context variables and an environment value conditioned on them.

Typical use::

    from unknown_unknowns.data_generator import DataGenerator, Variable

    gen = DataGenerator(
        variables=[Variable("light", [0, 1]), Variable("wet", [0, 1]), Variable("busy", [0, 1])],
        prior="context_prior.txt",
        seed=0,
    )
    gen.set_distribution((0, 0, 0), mean=5, std=1)
    ...
    context, value = gen.sample()
"""
from .generator import DataGenerator, Normal, Variable, output_value
from .prior import load_distributions, load_prior

__all__ = ["DataGenerator", "Normal", "Variable", "output_value", "load_distributions", "load_prior"]
