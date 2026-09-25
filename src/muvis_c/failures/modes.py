"""Severity-parameterised sensor failure modes."""

import torch

from muvis_c.failures.base import SeverityFailure


class BiasSeverity(SeverityFailure):
    """Additive constant bias:  x' = x + s * k"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        out[:, :, feature_idx] += severity * self.k
        return out


class NoiseSeverity(SeverityFailure):
    """Additive Gaussian noise:  noise ~ N(0, s * k)"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        noise = torch.randn_like(out[:, :, feature_idx]) * (severity * self.k)
        out[:, :, feature_idx] += noise
        return out


class ScalingSeverity(SeverityFailure):
    """Multiplicative scaling:  x' = x * (1 + s * (k - 1))"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        factor = 1.0 + severity * (self.k - 1.0)
        out[:, :, feature_idx] *= factor
        return out


class TimeVaryingScalingSeverity(SeverityFailure):
    """Time-varying multiplicative scaling.

    scale(t) = 1 + s * (k - 1) * t / T
    """

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        T = out.shape[1]
        t = torch.arange(T, dtype=torch.float32, device=X.device) / max(T - 1, 1)
        factor = 1.0 + severity * (self.k - 1.0) * t
        out[:, :, feature_idx] *= factor.unsqueeze(0)
        return out


class LinearDriftSeverity(SeverityFailure):
    """Linear drift:  drift(t) = s * k * (t / T)"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        T = out.shape[1]
        t = torch.arange(T, dtype=torch.float32, device=X.device) / max(T - 1, 1)
        drift = severity * self.k * t
        out[:, :, feature_idx] += drift.unsqueeze(0)
        return out


class NonlinearDriftSeverity(SeverityFailure):
    """Quadratic drift:  drift(t) = s * k * (t / T)^2"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        T = out.shape[1]
        t = torch.arange(T, dtype=torch.float32, device=X.device) / max(T - 1, 1)
        drift = severity * self.k * t ** 2
        out[:, :, feature_idx] += drift.unsqueeze(0)
        return out


class OutliersSeverity(SeverityFailure):
    """Sporadic outliers whose probability scales with severity.

    Parameters
    ----------
    k     : deviation scale (default 3.0)
    p_max : outlier probability at s = 1 (default 0.3)
    """

    def __init__(self, k: float = 3.0, p_max: float = 0.3):
        super().__init__(k)
        self.p_max = p_max

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        B, T, _ = out.shape
        prob = severity * self.p_max
        mask = (torch.rand(B, T, device=X.device) < prob).float()
        dev = (torch.rand(B, T, device=X.device) * 2 - 1) * self.k
        out[:, :, feature_idx] += mask * dev
        return out


class TrimmingVaryingSeverity(SeverityFailure):
    """Soft clipping with damping; bounds narrow with severity.

    Parameters
    ----------
    k              : max half-width (default 3.0)
    damping_factor : pull strength toward bound (default 0.4)
    """

    def __init__(self, k: float = 3.0, damping_factor: float = 0.4):
        super().__init__(k)
        self.damping_factor = damping_factor

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        hw = (1.0 - severity) * self.k
        lo, hi = -hw, hw
        feat = out[:, :, feature_idx]
        below = feat < lo
        above = feat > hi
        damped_lo = lo + (feat - lo) * self.damping_factor
        damped_hi = hi + (feat - hi) * self.damping_factor
        out[:, :, feature_idx] = torch.where(
            below, damped_lo, torch.where(above, damped_hi, feat)
        )
        return out


class TrimmingConstantSeverity(SeverityFailure):
    """Hard clipping — same as TrimmingVarying with damping = 0."""

    def __init__(self, k: float = 3.0):
        super().__init__(k)
        self._inner = TrimmingVaryingSeverity(k=k, damping_factor=0.0)

    def apply(self, X, feature_idx, severity):
        return self._inner.apply(X, feature_idx, severity)


class HardFaultSeverity(SeverityFailure):
    """Sensor stuck-at fault:  x'[t] = s * k"""

    def apply(self, X, feature_idx, severity):
        out = X.clone()
        out[:, :, feature_idx] = severity * self.k
        return out



