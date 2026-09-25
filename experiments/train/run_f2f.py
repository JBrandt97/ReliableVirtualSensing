"""Two-phase F2F training: self-supervised pretraining → supervised fine-tuning.

Follows the same contract as ``run_nn_experiments.run_experiment``:
    (model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper)
and saves ``best_model.pth`` + ``config.yaml`` in the standard log directory so
that the evaluation / severity pipeline can load the model without changes.
"""

import copy
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
from experiments.models.f2f_components import (
    sample_binary_mask,
    MaskedMSELoss,
    MASK_REGISTRY,
    configure_f2f_optimizer,
    WarmupCosineSchedule,
)


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------

def _pretrain(model, pretrain_head, train_loader, val_loader,
              pretrain_conf, device, seed, seq_len, num_features):
    """Self-supervised pretraining with masked reconstruction."""

    compose = nn.ModuleList([model.backbone, pretrain_head]).to(device)
    compose.train()

    opt_conf = pretrain_conf["optimizer_params"]
    num_epochs = pretrain_conf["num_epochs"]
    total_steps = len(train_loader) * num_epochs

    optimizer = configure_f2f_optimizer(
        compose,
        learning_rate=opt_conf["peak_lr"],
        weight_decay=opt_conf.get("weight_decay", 0.0),
        betas=(opt_conf.get("beta_1", 0.9), opt_conf.get("beta_2", 0.999)),
    )
    scheduler = WarmupCosineSchedule(
        optimizer, total_steps,
        warmup_steps=total_steps // opt_conf.get("warmup_div", 20),
        initial_lr=opt_conf.get("initial_lr", 1e-5),
        min_lr=opt_conf.get("min_lr", 1e-6),
        peak_lr=opt_conf["peak_lr"],
    )

    mask_conf = pretrain_conf.get("masking", {})
    masking_ratio = mask_conf.get("masking_ratio", 0.15)
    mean_mask_length = mask_conf.get("mean_mask_length", 3)
    mask_names = mask_conf.get("masks", ["mean", "bias", "noise"])
    multi_task_weights = mask_conf.get("multi_task_weights", [1] * len(mask_names))
    grad_clip = opt_conf.get("grad_clipping", 1.0)

    masks = {name: MASK_REGISTRY[name](device, mask_conf) for name in mask_names if name in MASK_REGISTRY}

    criterion = MaskedMSELoss()
    best_val_loss = float("inf")
    best_backbone_sd = None

    for epoch in range(1, num_epochs + 1):
        compose.train()
        running_loss = 0.0
        n_batches = 0

        for inputs, _ in train_loader:          # ignore labels during pretraining
            inputs = inputs.to(device)
            optimizer.zero_grad()
            scheduler.step()

            X_split = list(torch.tensor_split(inputs, len(masks), dim=0))
            loss = torch.tensor(0.0, device=device)

            for idx, (mask_name, mask_fn) in enumerate(masks.items()):
                X_m = X_split[idx]
                binary_mask = sample_binary_mask(X_m, device, masking_ratio, mean_mask_length)
                X_masked = mask_fn.apply_mask(X_m, binary_mask)

                emb = model._backbone_embed(X_masked)              # (B, T, d_model)
                emb_flat = emb.reshape(X_masked.size(0), -1)
                pred = pretrain_head(emb_flat)[0]                    # first head
                pred = pred.reshape(X_m.size(0), seq_len, num_features)
                loss = loss + multi_task_weights[idx] * criterion(pred, X_m, binary_mask.bool())

            loss.backward()
            if scheduler._step > scheduler.warmup_steps:
                torch.nn.utils.clip_grad_norm_(compose.parameters(), max_norm=grad_clip)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation
        compose.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                X_split = list(torch.tensor_split(inputs, len(masks), dim=0))
                loss = torch.tensor(0.0, device=device)
                for idx, (mask_name, mask_fn) in enumerate(masks.items()):
                    X_m = X_split[idx]
                    binary_mask = sample_binary_mask(X_m, device, masking_ratio, mean_mask_length)
                    X_masked = mask_fn.apply_mask(X_m, binary_mask)
                    emb = model._backbone_embed(X_masked)              # (B, T, d_model)
                    emb_flat = emb.reshape(X_masked.size(0), -1)
                    pred = pretrain_head(emb_flat)[0]
                    pred = pred.reshape(X_m.size(0), seq_len, num_features)
                    loss = loss + multi_task_weights[idx] * criterion(pred, X_m, binary_mask.bool())
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_backbone_sd = copy.deepcopy(model.backbone.state_dict())
            improved = " | ★ best"

        logging.info(f"[Pretrain] Epoch {epoch:>4}/{num_epochs} | train={train_loss:.5f} | val={val_loss:.5f}{improved}")

    logging.info("Using backbone from final pretrain epoch")
    return model


# ---------------------------------------------------------------------------
# Fine-tuning loop (standard supervised training)
# ---------------------------------------------------------------------------

def _finetune(model, train_loader, val_loader, train_conf, device, model_path, freeze_backbone=False):
    """Supervised fine-tuning with MSE loss. Saves best checkpoint."""

    opt_conf = train_conf["optimizer_params"]
    num_epochs = train_conf["num_epochs"]

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_conf["lr"])

    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        if freeze_backbone:
            model.head.train()
        else:
            model.train()

        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
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
            f"[Finetune] Epoch {epoch:>4}/{num_epochs} "
            f"| Train RMSE: {np.sqrt(epoch_loss):.4f} "
            f"| Val RMSE: {np.sqrt(val_loss):.4f}{improved}"
        )

    if model_path:
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded best fine-tuned model from {model_path}")

    return model, best_val_loss


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(conf, log_level="INFO"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    arch_conf = conf["Architecture"]
    train_conf = conf["Training"]
    pretrain_conf = conf["Pretraining"]

    # Logging
    file_handler = None
    log_dir = None
    if log_level:
        log_dir = os.path.join("logs", arch_conf["class"] + "_" + train_conf["dataset_id"] + "_" + timestamp)
        file_handler = setup_logging(log_dir, log_level)
        logging.info("Starting F2F Experiment")
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
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
    seq_len = X_train.shape[1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    dataset_class = train_conf.get("dataset_class", "SequentialDataset")
    train_ds = datasets.__dict__[dataset_class](X_train_scaled, y_train)
    val_ds = datasets.__dict__[dataset_class](X_val_scaled, y_val)
    test_ds = datasets.__dict__[dataset_class](X_test_scaled, y_test)

    pt_batch = pretrain_conf.get("batch_size", train_conf.get("batch_size", 256))
    ft_batch = train_conf.get("batch_size", 256)

    train_loader_pt = DataLoader(train_ds, batch_size=pt_batch, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader_pt = DataLoader(val_ds, batch_size=pt_batch, shuffle=False)
    train_loader_ft = DataLoader(train_ds, batch_size=ft_batch, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader_ft = DataLoader(val_ds, batch_size=ft_batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=ft_batch, shuffle=False)

    # Build model
    model_params = arch_conf["parameters"].copy()
    model_params["feat_dim"] = num_features
    model = architectures.__dict__[arch_conf["class"]](**model_params).to(device)

    # Phase 1: Pretrain
    logging.info("═" * 50)
    logging.info("Phase 1: Self-supervised pretraining")
    logging.info("═" * 50)

    pt_head_conf = pretrain_conf.get("pretrain_head", {})
    d_model = model_params["d_model"]
    embed_dim = d_model * seq_len
    pretrain_head = architectures._F2FPretrainHead(
        input_size=embed_dim,
        hidden_layer=pt_head_conf.get("hidden_layer", 512),
        dropout=pt_head_conf.get("dropout", 0.0),
        output_size=seq_len * num_features,
        num_heads=pt_head_conf.get("num_heads", 1),
    ).to(device)

    _pretrain(model, pretrain_head, train_loader_pt, val_loader_pt,
              pretrain_conf, device, seed, seq_len, num_features)
    del pretrain_head

    # Phase 2: Fine-tune — freeze backbone, train the head
    logging.info("═" * 50)
    logging.info("Phase 2: Supervised fine-tuning")
    logging.info("═" * 50)

    # Re-initialise head weights so fine-tuning starts fresh
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model_path = os.path.join(log_dir, "best_model.pth") if log_dir else None
    model, best_val_loss = _finetune(model, train_loader_ft, val_loader_ft,
                                      train_conf, device, model_path, freeze_backbone=True)

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
