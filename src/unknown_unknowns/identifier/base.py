"""Interface that every identifier implements."""
from abc import ABC, abstractmethod


class Identifier(ABC):
    """Decide which candidate variables explain the costs in the logs, given the known ones.

    Subclasses implement :meth:`identify`. Each row of the ranking it returns describes one
    candidate (or combination of candidates) and has at least the keys ``"candidate"`` (names,
    separated by spaces), ``"score"`` and ``"accepted"``; other keys depend on the method.
    """
    name = None

    @abstractmethod
    def identify(self, samples, known, candidates, outputs):
        """Return ``(variables to add, ranking)``.

        ``samples`` are the logs ``[({name: value}, {output: cost}), ...]``, ``known`` and
        ``candidates`` lists of variable names, ``outputs`` the cost names. The variables to add
        are a list of names (empty if no candidate is accepted), most important first.
        """
