"""Identify which context variables explain multimodal costs.

Every identifier implements :class:`Identifier` (``identify(samples, known, candidates,
outputs) -> (variables to add, ranking)``):

- :class:`MutualInfoIdentifier` (``mutual_info``): conditional mutual information between each
  candidate and the costs given the known variables, with a permutation test.
- :class:`ShapIdentifier` (``shap``): exact Shapley values of the candidates for predicting the
  costs, with a permutation test.
- :class:`CtreeIdentifier` (``ctree``): conditional inference trees (Hothorn, Hornik & Zeileis,
  2006).
- :class:`UnimodalityIdentifier` (``unimodality``): combinations of candidates that make every
  bin unimodal and whose variables have a significant effect.

Typical use::

    from unknown_unknowns.identifier import MutualInfoIdentifier

    identifier = MutualInfoIdentifier(alpha=0.05, epsilon=0.05)
    selected, ranking = identifier.identify(samples, known=[], candidates=["light", "wet", "busy"],
                                            outputs=["path1", "path2"])
"""
from .base import Identifier
from .ctree import CtreeIdentifier, ctree, format_tree, independence_test, variables_used
from .mutual_info import (MutualInfoIdentifier, conditional_entropy,
                          conditional_mutual_information, identify, mutual_information,
                          rank_candidates, rank_gains, score, select_variables)
from .shap import ShapIdentifier, importance, shapley_values
from .unimodality import UnimodalityIdentifier, candidates, has_effect, identification_step

IDENTIFIERS = {cls.name: cls for cls in (MutualInfoIdentifier, ShapIdentifier, CtreeIdentifier,
                                         UnimodalityIdentifier)}

__all__ = ["Identifier", "IDENTIFIERS", "MutualInfoIdentifier", "ShapIdentifier",
           "CtreeIdentifier", "UnimodalityIdentifier",
           "mutual_information", "conditional_mutual_information", "conditional_entropy",
           "score", "rank_gains", "identify", "rank_candidates", "select_variables",
           "shapley_values", "importance",
           "ctree", "independence_test", "variables_used", "format_tree",
           "has_effect", "candidates", "identification_step"]
