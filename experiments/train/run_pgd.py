"""PGD adversarial training (Projected Gradient Descent perturbations).

Follows the same contract as ``run_nn_experiments.run_experiment``:
    (model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper)
and saves ``best_model.pth`` + ``config.yaml`` in the standard log directory.
"""

import logging
import os
import yaml
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# PGD attack
# ---------------------------------------------------------------------------

def _pgd_attack(model, X, y, criterion, epsilon, step_size, num_steps):
    """Generate PGD adversarial examples within an L∞ ε-ball around X.

    Parameters
    ----------
    model     : nn.Module (in eval mode, gradients w.r.t. input only)
    X         : (batch, seq_len, feat_dim)  clean inputs
    y         : (batch,)  targets
    criterion : loss function
    epsilon   : L∞ perturbation budget
    step_size : per-step perturbation magnitude (α)
    num_steps : number of PGD iterations (K)

    Returns
    -------
    X_adv : adversarial inputs (same shape as X, detached)
    """
    X_adv = X.clone().detach()
    # Random start within ε-ball
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)

    for _ in range(num_steps):
        X_adv.requires_grad_(True)
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + step_size * X_adv.grad.sign()
            # Project back into ε-ball
            X_adv = torch.clamp(X_adv, X - epsilon, X + epsilon)

        X_adv = X_adv.detach()

    return X_adv


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def _save_adv_panel(X_clean, X_adv, epoch, out_dir, n_samples=6):
    """Save a 6-panel figure comparing clean vs adversarial signals."""
    X_c = X_clean[:n_samples].detach().cpu().numpy()
    X_a = X_adv[:n_samples].detach().cpu().numpy()
    n = min(n_samples, X_c.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        # Plot first feature channel
        ax.plot(X_c[i, :, 0], label="clean", linewidth=0.8)
        ax.plot(X_a[i, :, 0], label="adv", linewidth=0.8, alpha=0.8)
        ax.set_title(f"Sample {i}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"PGD adversarial examples — epoch {epoch}", fontsize=11)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"adv_epoch_{epoch:04d}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_adversarial(model, train_loader, val_loader, train_conf, adv_conf,
                       device, model_path, vis_dir=None):
    """Supervised training with PGD adversarial examples."""

    opt_conf = train_conf["optimizer_params"]
    num_epochs = train_conf["num_epochs"]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_conf["lr"])

    # PGD config
    epsilon = adv_conf.get("epsilon", 0.3)
    step_size = adv_conf.get("step_size", 0.1)
    num_steps = adv_conf.get("num_steps", 7)
    adv_ratio = adv_conf.get("adv_ratio", 0.5)
    vis_every = adv_conf.get("vis_every", 10)

    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        vis_saved = False

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Split batch: clean portion + adversarial portion
            n_adv = max(1, int(inputs.size(0) * adv_ratio))
            n_clean = inputs.size(0) - n_adv

            # Generate PGD adversarial examples (model in eval mode for stable BN)
            model.eval()
            inputs_adv = _pgd_attack(
                model, inputs[n_clean:], labels[n_clean:],
                criterion, epsilon, step_size, num_steps,
            )
            model.train()

            # Visualise first batch at regular intervals
            if vis_dir and not vis_saved and epoch % vis_every == 0:
                _save_adv_panel(inputs[n_clean:], inputs_adv, epoch, vis_dir)
                vis_saved = True

            if n_clean > 0:
                X = torch.cat([inputs[:n_clean], inputs_adv], dim=0)
                y = labels  # labels unchanged for both clean and adv
            else:
                X = inputs_adv
                y = labels

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation (on clean data)
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
            f"[AdvTrain] Epoch {epoch:>4}/{num_epochs} "
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
    adv_conf = conf["Adversarial"]

    # Logging
    file_handler = None
    log_dir = None
    if log_level:
        log_dir = os.path.join(
            "logs",
            "PGD_" + train_conf["dataset_id"] + "_" + timestamp,
        )
        file_handler = setup_logging(log_dir, log_level)
        logging.info("Starting PGD Training Experiment")
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

    # Train with adversarial augmentation
    logging.info("═" * 50)
    logging.info("PGD Training")
    logging.info("═" * 50)

    model_path = os.path.join(log_dir, "best_model.pth") if log_dir else None
    vis_dir = os.path.join(log_dir, "adv_vis") if log_dir else None
    model, best_val_loss = _train_adversarial(
        model, train_loader, val_loader, train_conf, adv_conf, device, model_path,
        vis_dir=vis_dir,
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
