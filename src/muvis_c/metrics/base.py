from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Base class for additional evaluation metrics.

    Subclass this to compute extra columns alongside the default
    RMSE-based ``u_rel``.  Register via ``testbed.add_metric(...)``.

    Attributes
    ----------
    name : str
        Column name used in the raw DataFrame for the scalar value.

    Examples
    --------
    >>> class MAEMetric(Metric):
    ...     name = "mae"
    ...
    ...     def compute(self, y_true, y_pred):
    ...         return float(np.mean(np.abs(y_true - y_pred)))
    """

    name: str

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value.

        Parameters
        ----------
        y_true : ndarray, shape (N,)
        y_pred : ndarray, shape (N,)

        Returns
        -------
        float
        """
        ...
