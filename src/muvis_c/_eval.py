import gc
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from muvis_c._data import load_dataset
from muvis_c._device import resolve_device
from muvis_c.failures.base import SeverityFailure


def _log(msg: str):
    import sys
    tqdm.write(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Predict-fn builders for the three model registration modes
# ---------------------------------------------------------------------------

def _build_predict_fn_from_model(model_obj, device, batch_size):
    """Build a predict_fn from a PyTorch nn.Module or sklearn-like object."""
    if hasattr(model_obj, "parameters"):
        # PyTorch model
        model_obj = model_obj.to(device)
        model_obj.eval()
        is_mps = device == "mps"

        def _predict(X_tensor):
            loader = DataLoader(
                TensorDataset(X_tensor), batch_size=batch_size, shuffle=False
            )
            preds = []
            with torch.no_grad():
                for (batch,) in loader:
                    out = model_obj(batch.to(device))
                    preds.append(out.detach().cpu())
                    if is_mps:
                        torch.mps.synchronize()
            return torch.cat(preds).numpy()

        return _predict

    elif hasattr(model_obj, "predict"):
        # sklearn / tree-like model
        def _predict(X_tensor):
            X_flat = X_tensor.reshape(X_tensor.shape[0], -1).numpy()
            return model_obj.predict(X_flat)

        return _predict

    else:
        raise TypeError(
            f"Cannot build predict_fn from {type(model_obj).__name__}. "
            "Provide a PyTorch nn.Module (with .parameters()) or an "
            "sklearn-like object (with .predict())."
        )


def _build_predict_fn_from_callable(fn):
    """Wrap a user-provided callable so it matches the internal convention.

    The callable should accept a numpy (N, T, C) array and return (N,).
    Internally the sweep passes torch tensors, so we convert.
    """
    def _predict(X_tensor):
        X_np = X_tensor.numpy() if isinstance(X_tensor, torch.Tensor) else X_tensor
        return np.asarray(fn(X_np))

    return _predict


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    dataset,
    data_root,
    models,
    failures,
    target_features,
    extra_metrics,
    severity_steps,
    seed,
    device,
    scaler,
    batch_size=256,
    log_dir=None,
):
    """Run the severity sweep for a single dataset across models, channels and failures.

    RMSE and bootstrap RMSE are always computed.  Extra metrics (from
    ``add_metric``) add additional columns.

    Returns a list of dicts (one per severity step per feature per failure
    per model).
    """
    from muvis_c.metrics.builtins import RMSEMetric
    from muvis_c._seeding import seed_everything

    rmse_metric = RMSEMetric()
    seed_everything(seed)

    severities = np.linspace(0.0, 1.0, severity_steps + 1)
    all_rows = []

    # ── optional logging ──────────────────────────────────────────────
    file_handler = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        from muvis_c._logging import setup_logging, cleanup_logging
        file_handler = setup_logging(str(log_dir), "INFO", filename="severity.log")
        _log("Starting Severity Robustness Evaluation")

    # ── Load dataset ──────────────────────────────────────────────────
    X_train, y_train, X_test_np, y_test, dataset_id = load_dataset(
        dataset, data_root
    )

    n_models = len(models)
    n_failures = len(failures)
    n_severities = len(severities)
    num_features = X_train.shape[2]

    # Train/val split for scaler fitting (match original behaviour)
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, shuffle=True
    )
    del X_train

    # Fit scaler
    if scaler is not None:
        scaler_inst = scaler.__class__()
        scaler_inst.fit(X_train_sub.reshape(-1, num_features))
        X_test = torch.tensor(
            scaler_inst.transform(
                X_test_np.reshape(-1, num_features)
            ).reshape(X_test_np.shape),
            dtype=torch.float32,
        )
        X_train_scaled = scaler_inst.transform(
            X_train_sub.reshape(-1, num_features)
        ).reshape(X_train_sub.shape)
    else:
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        X_train_scaled = X_train_sub

    del X_test_np

    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    y_train_mean = float(y_train_sub.mean())
    del X_train_scaled

    # ── Baseline (mean predictor) ─────────────────────────────────────
    baseline_preds = np.full_like(y_test, fill_value=y_train_mean)
    baseline_rmse = rmse_metric.compute(y_test, baseline_preds)
    baseline_boot = rmse_metric.compute_with_uncertainty(
        y_test, baseline_preds, seed=seed
    )
    _log(f"Baseline RMSE: {baseline_rmse:.4f}  "
         f"Bootstrap: {baseline_boot['value']:.4f}")

    # ── Determine features to corrupt ─────────────────────────────────
    features = (
        list(range(num_features))
        if target_features == "all"
        else list(target_features)
    )
    n_features = len(features)

    total_steps = n_models * n_failures * n_features * n_severities
    pbar = tqdm(total=total_steps, desc=dataset_id, unit="step")

    # ── model loop ────────────────────────────────────────────────────
    for m_idx, (model_name, model_spec) in enumerate(models.items()):
        pbar.set_description(f"{dataset_id} | {model_name} ({m_idx+1}/{n_models})")

        # Build predict_fn
        if model_spec["predict_fn"] is not None:
            predict_fn = _build_predict_fn_from_callable(
                model_spec["predict_fn"]
            )
        elif model_spec["model"] is not None:
            predict_fn = _build_predict_fn_from_model(
                model_spec["model"], device, batch_size=batch_size
            )
        else:
            raise RuntimeError(f"No model source for '{model_name}'")

        # ── Clean baseline predictions ────────────────────────────────
        clean_preds = predict_fn(X_test)
        clean_rmse = rmse_metric.compute(y_test, clean_preds)
        clean_boot = rmse_metric.compute_with_uncertainty(
            y_test, clean_preds, seed=seed
        )
        pbar.set_postfix_str(
            f"Clean RMSE: {clean_rmse:.4f} "
            f"[{clean_boot['ci_lower']:.4f}, {clean_boot['ci_upper']:.4f}]"
        )

        # Pre-compute clean values for extra metrics
        clean_extra = {}
        for em in extra_metrics:
            clean_extra[em.name] = em.compute(y_test, clean_preds)

        # ── failure × feature × severity loop ─────────────────────────
        for f_idx, failure in enumerate(failures):
            fail_name = type(failure).__name__

            for feat_pos, feat_idx in enumerate(features):
                for s_idx, s in enumerate(severities):
                    X_corr = failure.apply(
                        X_test, feature_idx=feat_idx, severity=s
                    )
                    preds_s = predict_fn(X_corr)
                    del X_corr

                    # Metric values
                    err_rmse = rmse_metric.compute(y_test, preds_s)
                    boot_vals = rmse_metric.compute_with_uncertainty(
                        y_test, preds_s, seed=seed
                    )

                    row = {
                        "model": model_name,
                        "dataset": dataset_id,
                        "failure_mode": fail_name,
                        "feature_idx": feat_idx,
                        "severity": float(s),
                        "k": failure.k,
                        # primary metric (RMSE)
                        "rmse": err_rmse,
                        # baseline
                        "baseline_rmse": baseline_rmse,
                        "baseline_boot_mean": baseline_boot["value"],
                        # clean reference
                        "clean_rmse": clean_rmse,
                        "clean_boot_mean": clean_boot["value"],
                        "clean_ci_lower": clean_boot["ci_lower"],
                        "clean_ci_upper": clean_boot["ci_upper"],
                        # corrupted bootstrap
                        "boot_mean": boot_vals["value"],
                        "ci_lower": boot_vals["ci_lower"],
                        "ci_upper": boot_vals["ci_upper"],
                    }

                    # Extra metrics
                    for em in extra_metrics:
                        row[em.name] = em.compute(y_test, preds_s)
                        row[f"clean_{em.name}"] = clean_extra[em.name]

                    all_rows.append(row)

                    pbar.set_postfix_str(
                        f"{fail_name} [{f_idx+1}/{n_failures}] "
                        f"feat {feat_pos+1}/{n_features} "
                        f"RMSE={err_rmse:.3f}"
                    )
                    pbar.update(1)
                    del preds_s

        del predict_fn, clean_preds
        gc.collect()

    # Free dataset memory
    del X_test, X_train_flat, X_train_sub, y_test, y_train, y_train_sub
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    pbar.close()

    if file_handler is not None:
        from muvis_c._logging import cleanup_logging
        cleanup_logging(file_handler)

    return all_rows
