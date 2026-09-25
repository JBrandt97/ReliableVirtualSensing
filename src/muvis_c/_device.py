"""Auto-detect the best available torch device."""

import torch


def resolve_device(device: str = "auto") -> str:
    """Return a concrete device string.

    Parameters
    ----------
    device : str
        ``"auto"`` (default) tries cuda → mps → cpu.
        Any other value (``"cuda"``, ``"mps"``, ``"cpu"``) is passed
        through unchanged.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
