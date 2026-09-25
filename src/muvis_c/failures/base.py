from abc import ABC, abstractmethod

import torch


class SeverityFailure(ABC):
    """Base class for severity-parameterised sensor failures.

    Assumes z-score normalised input (mu ~ 0, sigma ~ 1 per channel).

    Parameters
    ----------
    k : float
        Maximum perturbation scale.

    Examples
    --------
    Implement a custom failure mode::

        class SpikeFailure(SeverityFailure):
            \"\"\"Inject a spike at the midpoint of the sequence.\"\"\"

            def apply(self, X, feature_idx, severity):
                out = X.clone()
                mid = out.shape[1] // 2
                out[:, mid, feature_idx] += severity * self.k
                return out

    Then use it with the ``Testbed``::

        from muvis_c import Testbed

        testbed = Testbed(dataset=..., failures=[SpikeFailure(k=5.0)])
    """

    def __init__(self, k: float = 3.0):
        self.k = k

    @abstractmethod
    def apply(
        self, X: torch.Tensor, feature_idx: int, severity: float
    ) -> torch.Tensor:
        """Return a corrupted **copy** of *X* at the given severity in [0, 1]."""
        ...

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"
