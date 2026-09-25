"""Input Sensor Dropout (ISensD) training.

Uses the same F2F architecture (backbone + regression head) but applies
random per-channel (per-sensor) dropout during training to robustify the
model against missing or degraded sensor inputs.

Two masking modes are supported:
  - **bernoulli**: each feature channel is independently zeroed with
    probability ``sensd_ratio``.
  - **nr** (No Ratio): uniformly sample one of the 2^C - 1 non-empty
    channel subsets, then zero out the rest.

Follows the same contract as ``run_nn_experiments.run_experiment``:
    (model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper)
and saves ``best_model.pth`` + ``config.yaml`` in the standard log directory.
"""

import logging
import os
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models import architectures, datasets
from muvis_c._seeding import seed_everything, seed_worker
from muvis_c._logging import setup_logging, cleanup_logging
from muvis_c._dataset import MuViSDataset
from experiments.utils.testing import bootstrap_testset


# ---------------------------------------------------------------------------
# Sensor dropout
# ---------------------------------------------------------------------------

def _sensor_dropout(X, mode, sensd_ratio):
    """Apply per-channel sensor dropout to a batch.

    Parameters
    ----------
    X           : (batch, seq_len, feat_dim)
    mode        : "bernoulli" | "nr"
    sensd_ratio : drop probability per channel (used in bernoulli mode)

    Returns
    -------
    X_masked : same shape, with zeroed-out channels
    """
    B, T, C = X.shape

    if mode == "nr":
        # Sample one of 2^C - 1 non-empty subsets per sample
        keep = torch.zeros(B, 1, C, device=X.device)
        for i in range(B):
            subset = 0
            while subset == 0:
                subset = torch.randint(0, 2**C, (1,)).item()
            bits = [(subset >> c) & 1 for c in range(C)]
            keep[i, 0, :] = torch.tensor(bits, dtype=X.dtype, device=X.device)
    else:
        # Bernoulli: each channel kept with probability (1 - sensd_ratio)
        keep = (torch.rand(B, 1, C, device=X.device) >= sensd_ratio).float()
        # Ensure at least one channel is kept per sample
        all_dropped = keep.sum(dim=2, keepdim=True) == 0
        if all_dropped.any():
            fix = torch.zeros_like(keep)
            fix[:, :, 0] = 1.0
            keep = torch.where(all_dropped, fix, keep)

    return X * keep


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_isensd(model, train_loader, val_loader, train_conf, sensd_conf,
                  device, model_path):
    """Supervised training with input sensor dropout augmentation."""

    opt_conf = train_conf["optimizer_params"]
    num_epochs = train_conf["num_epochs"]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_conf["lr"])

    # ISensD config
    mode = sensd_conf.get("mode", "bernoulli")
    sensd_ratio = sensd_conf.get("sensd_ratio", 0.2)

    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply sensor dropout
            inputs_masked = _sensor_dropout(inputs, mode, sensd_ratio)

            optimizer.zero_grad()
            outputs = model(inputs_masked)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation (on clean data — no dropout)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_path:
                torch.save(model.state_dict(), model_path)
                improved = " | ★ saved"

        logging.info(
            f"[ISensD] Epoch {epoch:>4}/{num_epochs} "
            f"| Train RMSE: {np.sqrt(epoch_loss):.4f} "
            f"| Val RMSE: {np.sqrt(val_loss):.4f}{improved}"
        )

    if model_path:
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded best model from {model_path}")

    return model, best_val_loss


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(conf, log_level="INFO"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    arch_conf = conf["Architecture"]
    train_conf = conf["Training"]
    sensd_conf = conf["SensorDropout"]

    # Logging
    file_handler = None
    log_dir = None
    if log_level:
        log_dir = os.path.join(
            "logs",
            "ISensD_" + train_conf["dataset_id"] + "_" + timestamp,
        )
        file_handler = setup_logging(log_dir, log_level)
        logging.info("Starting ISensD Training Experiment")
        logging.info(f"Configuration: {conf}")
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(conf, f)

    # Seeding
    seed = train_conf.get("seed", 42)
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Device
    if "device" not in train_conf:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = train_conf["device"]
    logging.info(f"Device: {device}")

    # Data
    dataset = MuViSDataset.get_dataset(train_conf["dataset_id"])
    X_full, y_full = dataset.get_data(split="train")
    X_test, y_test = dataset.get_data(split="test")

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=seed, shuffle=True,
    )

    num_features = X_train.shape[2]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    dataset_class = train_conf.get("dataset_class", "SequentialDataset")
    batch_size = train_conf.get("batch_size", 256)

    train_ds = datasets.__dict__[dataset_class](X_train_scaled, y_train)
    val_ds = datasets.__dict__[dataset_class](X_val_scaled, y_val)
    test_ds = datasets.__dict__[dataset_class](X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model_params = arch_conf["parameters"].copy()
    model_params["feat_dim"] = num_features
    model = architectures.__dict__[arch_conf["class"]](**model_params).to(device)

    # Train with sensor dropout augmentation
    logging.info("═" * 50)
    logging.info("ISensD Training")
    logging.info("═" * 50)

    model_path = os.path.join(log_dir, "best_model.pth") if log_dir else None
    model, best_val_loss = _train_isensd(
        model, train_loader, val_loader, train_conf, sensd_conf, device, model_path,
    )

    # Test evaluation
    model.eval()
    criterion = nn.MSELoss()
    running_test_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    test_loss = running_test_loss / len(test_loader.dataset)
    preds_np = torch.cat(all_preds, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    boot_mean, ci_lower, ci_upper = bootstrap_testset(preds_np, labels_np, seed=seed)
    logging.info(f"Bootstrap Test RMSE: {boot_mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    if file_handler:
        cleanup_logging(file_handler)

    return model, np.sqrt(best_val_loss), np.sqrt(test_loss), boot_mean, ci_lower, ci_upper
