from muvis_c.failures.base import SeverityFailure
from muvis_c.failures.modes import (
    BiasSeverity,
    NoiseSeverity,
    ScalingSeverity,
    TimeVaryingScalingSeverity,
    LinearDriftSeverity,
    NonlinearDriftSeverity,
    OutliersSeverity,
    TrimmingVaryingSeverity,
    TrimmingConstantSeverity,
    HardFaultSeverity,
)

__all__ = [
    "SeverityFailure",
    "BiasSeverity",
    "NoiseSeverity",
    "ScalingSeverity",
    "TimeVaryingScalingSeverity",
    "LinearDriftSeverity",
    "NonlinearDriftSeverity",
    "OutliersSeverity",
    "TrimmingVaryingSeverity",
    "TrimmingConstantSeverity",
    "HardFaultSeverity",
    "SEVERITY_REGISTRY",
    "build_all_failures",
    "build_severity_failure",
]

SEVERITY_REGISTRY = {
    "bias": BiasSeverity,
    "noise": NoiseSeverity,
    "scaling": ScalingSeverity,
    "time_varying_scaling": TimeVaryingScalingSeverity,
    "linear_drift": LinearDriftSeverity,
    "nonlinear_drift": NonlinearDriftSeverity,
    "outliers": OutliersSeverity,
    "trimming_varying": TrimmingVaryingSeverity,
    "trimming_constant": TrimmingConstantSeverity,
    "hard_fault": HardFaultSeverity,
}


def build_severity_failure(name: str, k: float = 3.0, **kwargs) -> SeverityFailure:
    """Instantiate a severity failure by registry name."""
    if name not in SEVERITY_REGISTRY:
        raise ValueError(
            f"Unknown failure '{name}'. "
            f"Available: {list(SEVERITY_REGISTRY.keys())}"
        )
    return SEVERITY_REGISTRY[name](k=k, **kwargs)


def build_all_failures(k: float = 3.0) -> list:
    """Instantiate all registered failure modes with the given *k*."""
    return [cls(k=k) for cls in SEVERITY_REGISTRY.values()]
