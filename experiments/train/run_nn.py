import logging
import os
import yaml
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models import architectures
from experiments.models import datasets
from muvis_c._seeding import *
from muvis_c._logging import setup_logging, cleanup_logging
from muvis_c._dataset import MuViSDataset
from experiments.utils.testing import bootstrap_testset

def train_model(model, criterion, optimizer, train_dataloader, val_dataloader=None, num_epochs=25, device='cpu', model_path=None):
    model.train()
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataloader.dataset)

        #rmse
        report_str = f'Epoch {epoch:<4}/ {num_epochs - 1:<4} | Train RMSE: {np.sqrt(epoch_loss):.4f} '

        # Validation
        if val_dataloader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    val_outputs = model(inputs)
                    loss = criterion(val_outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
            val_loss = running_val_loss / len(val_dataloader.dataset)
            report_str += f'| Val RMSE: {np.sqrt(val_loss):.4f}'
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_path:
                    report_str += f" | New best model saved"
                    torch.save(model.state_dict(), model_path)
            model.train()
        logging.info(report_str)
        
    # load best model for evaluation
    if model_path:
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded best model from {model_path}")
    
    return model, best_val_loss

def run_experiment(conf, log_level="INFO"):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    arch_conf = conf['Architecture']
    train_conf = conf['Training']
    
    file_handler = None
    log_dir = None
    if log_level:
        log_dir = os.path.join("logs", arch_conf['class'] + "_" + train_conf['dataset_id'] + "_" + timestamp)
        file_handler = setup_logging(log_dir, log_level)
 
        logging.info("Starting Experiment")
        logging.info(f"Configuration: {conf}")
        
        # loggin config yaml
        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(conf, f)

    if "seed" not in train_conf:
        seed = 42
        logging.info(f"Random seed not specified, using {seed}")
    else:
        seed = train_conf['seed']
        logging.info(f'Specified Seed: {seed}')
        
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    if "device" not in train_conf:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info(f"No device specified in configuration. Using device: {device}")
    else:
        device = train_conf['device']
        
    dataset = MuViSDataset.get_dataset(train_conf['dataset_id'])
    X_full, y_full = dataset.get_data(split='train')
    X_test, y_test = dataset.get_data(split='test')

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=seed, shuffle=True
    )

    num_features = X_train.shape[2]
    
    # standard scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    # setup dataloader
    train_dataset = datasets.__dict__[train_conf['dataset_class']](X_train_scaled, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=train_conf['batch_size'], shuffle=True,worker_init_fn=seed_worker, generator=g)
    
    val_dataset = datasets.__dict__[train_conf['dataset_class']](X_val_scaled, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=train_conf['batch_size'], shuffle=False)

    test_dataset = datasets.__dict__[train_conf['dataset_class']](X_test_scaled, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=train_conf['batch_size'], shuffle=False)

    # instantiate model
    model_params = arch_conf['parameters']
    model_params['feat_dim'] = train_dataset.feat_dim
    model = architectures.__dict__[arch_conf['class']](**model_params).to(device)

    # define criterion and optimizer
    criterion_class = getattr(nn, train_conf['loss_function'])
    criterion = criterion_class()

    optimizer_class = getattr(optim, train_conf['optimizer'])
    opt_params = train_conf["optimizer_params"]
    optimizer = optimizer_class(model.parameters(), **opt_params)
    # run training
    logging.info("Starting Training")
    model_path = os.path.join(log_dir, "best_model.pth") if log_dir else None
    if model_path:
        logging.info(f"Best model will be saved to: {model_path}")
    model, best_val_loss = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=train_conf['num_epochs'], device=device, model_path=model_path)
    logging.info("Training completed")
    # evaluate on test set
    model.eval()
    running_test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * inputs.size(0)

            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    test_loss = running_test_loss / len(test_dataloader.dataset)

    preds_np = torch.cat(all_preds, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    boot_mean, ci_lower, ci_upper = bootstrap_testset(preds_np, labels_np, seed=seed)
    logging.info(f'Bootstrap Test RMSE: {boot_mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]')

    # returns mses
    if file_handler:
        cleanup_logging(file_handler)
    return model, np.sqrt(best_val_loss), np.sqrt(test_loss), boot_mean, ci_lower, ci_upper