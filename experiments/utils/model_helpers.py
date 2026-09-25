"""Shared helpers for loading trained models and building prediction functions.

Used by ``run_severity_eval``.
"""

import logging

import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset

import muvis_c as rob
from experiments.models import architectures
from experiments.utils.checkpoint import (
    resolve_log_dir,
    find_nn_checkpoint,
    find_tree_checkpoint,
    load_config_from_log,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_nn_model(log_dir, device, feat_dim):
    """Load a trained NN model from its checkpoint directory."""
    conf = load_config_from_log(log_dir)
    arch_conf = conf["Architecture"]
    model_params = arch_conf["parameters"].copy()
    model_params["feat_dim"] = feat_dim

    model = architectures.__dict__[arch_conf["class"]](**model_params).to(device)
    checkpoint_path = find_nn_checkpoint(log_dir)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if not isinstance(model, (architectures.TST, architectures.F2F)):
        model = torch.compile(model)

    logging.info(f"Loaded NN model from {checkpoint_path}")
    return model, conf


def load_tree_model(log_dir):
    """Load a trained tree model from its checkpoint directory."""
    checkpoint_path = find_tree_checkpoint(log_dir)
    model = joblib.load(checkpoint_path)
    conf = load_config_from_log(log_dir)
    logging.info(f"Loaded tree model from {checkpoint_path}")
    return model, conf


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_nn(model, X: torch.Tensor, device, batch_size=256, flatten=False):
    """Run batched NN inference.  Accepts and returns torch tensors (CPU)."""
    model.eval()
    if flatten:
        X = X.reshape(X.shape[0], -1)

    is_mps = (device == "mps")
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                out = model(batch.to(device))

            preds.append(out.detach().float().cpu())
            if is_mps:
                torch.mps.synchronize()

    return torch.cat(preds)


def predict_tree(model, X: torch.Tensor):
    """Run tree inference.  Accepts torch tensor, returns numpy array."""
    X_flat = X.reshape(X.shape[0], -1).numpy()
    return model.predict(X_flat)


# ---------------------------------------------------------------------------
# Build a unified predict_fn for any model type
# ---------------------------------------------------------------------------
def build_predict_fn(model_class, dataset_id, logs_root, device, batch_size):
    """Resolve checkpoint, load model, return ``predict_fn(X_np) -> np.ndarray``."""
    model_log_dir = resolve_log_dir(model_class, dataset_id, logs_root=logs_root)
    train_conf = load_config_from_log(model_log_dir)
    exp_type = train_conf["experiment_type"]

    if exp_type in ("nn", "f2f", "pgd", "isensd"):
        dataset_class_name = train_conf["Training"].get("dataset_class", "SequentialDataset")
        info = rob.dataset_info(dataset_id)
        feat_dim = (
            info["n_channels"] * info["seq_len"]
            if dataset_class_name == "FlattenedDataset"
            else info["n_channels"]
        )
        model, _ = load_nn_model(model_log_dir, device, feat_dim)
        flatten = (dataset_class_name == "FlattenedDataset")

        def nn_predict(X: np.ndarray) -> np.ndarray:
            X_t = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
            return predict_nn(model, X_t, device, batch_size, flatten=flatten).numpy()
        return nn_predict

    elif exp_type == "tree":
        model, _ = load_tree_model(model_log_dir)

        def tree_predict(X: np.ndarray) -> np.ndarray:
            X_t = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
            return predict_tree(model, X_t)
        return tree_predict

    else:
        raise ValueError(f"Unknown experiment type '{exp_type}' for {model_class}")
