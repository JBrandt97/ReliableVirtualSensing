import numpy as np
import torch

from muvis_c.metrics.base import Metric


def _detect_boot_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_BOOT_DEVICE = _detect_boot_device()


class RMSEMetric(Metric):
    """Root Mean Squared Error."""

    name = "rmse"

    def compute(self, y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def compute_with_uncertainty(self, y_true, y_pred, seed=42, n_boot=200):
        torch.manual_seed(seed)
        N = len(y_true)
        yt = torch.tensor(y_true, dtype=torch.float32, device=_BOOT_DEVICE)
        yp = torch.tensor(y_pred, dtype=torch.float32, device=_BOOT_DEVICE)
        idx = torch.randint(0, N, (n_boot, N), device=_BOOT_DEVICE)
        vals = (yt[idx] - yp[idx]).pow(2).mean(dim=1).sqrt()
        if _BOOT_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elif _BOOT_DEVICE.type == "mps":
            torch.mps.synchronize()
        vals_np = vals.cpu().numpy()
        return {
            "value": float(np.mean(vals_np)),
            "ci_lower": float(np.percentile(vals_np, 2.5)),
            "ci_upper": float(np.percentile(vals_np, 97.5)),
        }
