"""Masking utilities and training helpers for the F2F self-supervised pretraining phase.

The F2F backbone is a TSTransformerEncoder (see architectures.py).
This module provides the masking strategies, loss, optimizer, and scheduler
used during pretraining.

Sources
-------
- https://openreview.net/forum?id=9aElHWiZ72
- https://github.com/JBrandt97/FaultsToFeatures
- https://github.com/gzerveas/mvts_transformer
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Masking utilities (pretraining only)
# ---------------------------------------------------------------------------

def _geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask


def sample_binary_mask(X, device, masking_ratio=0.15, mean_mask_length=3):
    """Sample a geometric binary mask for each sample in the batch."""
    mask = torch.ones_like(X).int().to(device)
    for b in range(X.size(0)):
        seq_len, n_feat = X[b].shape
        m = np.ones((seq_len, n_feat), dtype=bool)
        for f in range(n_feat):
            m[:, f] = _geom_noise_mask_single(seq_len, mean_mask_length, masking_ratio)
        mask[b] = torch.tensor(m, dtype=torch.int, device=device)
    return mask


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true, mask):
        masked_pred = torch.masked_select(y_pred, ~mask)
        masked_true = torch.masked_select(y_true, ~mask)
        return self.mse_loss(masked_pred, masked_true)


class BiasMask:
    """Apply additive bias where mask is False."""
    def __init__(self, device, lower_bound=-1, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.device = device

    def apply_mask(self, X, binary_mask):
        tensor = X.clone().to(self.device)
        bias = torch.FloatTensor(tensor.size(0), 1, tensor.size(2)).uniform_(
            self.lower_bound, self.upper_bound
        ).to(self.device)
        tensor = tensor + bias * (~binary_mask.bool()).int()
        return tensor


class MeanMask:
    """Zero out values where mask is False."""
    def __init__(self, device):
        self.device = device

    def apply_mask(self, X, binary_mask):
        return X.clone().to(self.device) * binary_mask.to(self.device)


class NoiseMask:
    """Add Gaussian noise where mask is False."""
    def __init__(self, device, noise_level=0.4):
        self.noise_level = noise_level
        self.device = device

    def apply_mask(self, X, binary_mask):
        tensor = X.clone().to(self.device)
        noise = torch.normal(mean=0, std=self.noise_level, size=tensor.shape).to(self.device)
        tensor = tensor + noise * (~binary_mask.bool()).int()
        return tensor


MASK_REGISTRY = {
    "bias": lambda device, cfg: BiasMask(device, cfg.get("bias_bounds", [-1, 1])[0], cfg.get("bias_bounds", [-1, 1])[1]),
    "mean": lambda device, cfg: MeanMask(device),
    "noise": lambda device, cfg: NoiseMask(device, cfg.get("noise_level", 0.4)),
}


# ---------------------------------------------------------------------------
# Optimizer & scheduler helpers (pretraining / fine-tuning)
# ---------------------------------------------------------------------------

def configure_f2f_optimizer(model, learning_rate=0.001, weight_decay=0.01, betas=(0.9, 0.999)):
    """AdamW with separate decay / no-decay parameter groups."""
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.Linear,)):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                no_decay.add(fpn)
            elif "pe" in pn:
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    # Parameters not explicitly categorised go to decay
    remaining = param_dict.keys() - decay - no_decay
    decay |= remaining

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


class WarmupCosineSchedule:
    def __init__(self, optimizer, total_steps, warmup_steps, initial_lr, min_lr, peak_lr):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.T_max = total_steps - warmup_steps
        self.lr_increment = (peak_lr - initial_lr) / max(warmup_steps, 1)
        self._step = -1

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            lr = self.initial_lr + self._step * self.lr_increment
        else:
            progress = (self._step - self.warmup_steps) / max(self.T_max, 1)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr
