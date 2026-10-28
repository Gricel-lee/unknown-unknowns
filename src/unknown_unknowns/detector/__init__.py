"""Detect that the model of a system is missing something: the logged costs are multimodal.

Typical use::

    from unknown_unknowns.detector import all_unimodal, check_partition

    results = check_partition(samples, known_variables, alpha=0.05, outputs=["path1", "path2"])
    if not all_unimodal(results):
        ...   # some bin of the known variables has multimodal costs
"""
from .multimodality import all_tested, all_unimodal, check_partition, detect, dip_test, select

__all__ = ["select", "dip_test", "detect", "check_partition", "all_unimodal", "all_tested"]
