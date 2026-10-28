"""Interface that every controller generator implements."""
from abc import ABC, abstractmethod


class Controller(ABC):
    """Generic controller generator, to be subclassed for each problem (e.g. PRISM).

    Subclasses must implement :meth:`get_controller` and :meth:`get_metrics`;
    Python refuses to instantiate a subclass that doesn't.
    """

    def __init__(self, data=None):
        self.data = data          # data available to generate the controller
        self.controller = None    # the last generated controller
        self.metrics = {}         # metrics about the last generated controller

    def update_data(self, new_data):
        """Replace the data used to generate the controller."""
        self.data = new_data

    @abstractmethod
    def get_controller(self):
        """Generate the controller from ``self.data`` and return it."""

    @abstractmethod
    def get_metrics(self):
        """Return a dict of metrics about the last generated controller."""
