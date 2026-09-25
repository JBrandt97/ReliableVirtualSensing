import logging
import os
import yaml
from datetime import datetime

import numpy as np
import joblib
import xgboost as xgb
import catboost as cb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from muvis_c._seeding import *
from muvis_c._logging import setup_logging, cleanup_logging
from muvis_c._dataset import MuViSDataset
from experiments.utils.testing import bootstrap_testset

MODEL_MAP = {
    'XGBRegressor': xgb.XGBRegressor,
    'CatBoostRegressor': cb.CatBoostRegressor
}

def run_experiment(conf, log_level='INFO'):
    
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

    seed = train_conf.get("seed", 42)
    logging.info(f"Seed: {seed}")
    seed_everything(seed)

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

    # flatten for tree models
    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
    X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

    model_name = arch_conf['class']
    model_params = arch_conf['parameters']

    if model_name not in MODEL_MAP:
        logging.error(f"Error: Model '{model_name}' not recognized.")
        return
    
    model_class = MODEL_MAP[model_name]
    model = model_class(**model_params)

    logging.info(f"Training {model_name}...")
    model.fit(X_train_flat, y_train)

    # Save model checkpoint
    if log_dir:
        model_path = os.path.join(log_dir, "model.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

    # Validation Metrics
    val_preds = model.predict(X_val_flat)
    val_mse = mean_squared_error(y_val, val_preds)
    val_rmse = np.sqrt(val_mse)
    logging.info(f'Validation RMSE: {val_rmse:.4f}')

    # Test Metrics
    test_preds = model.predict(X_test_flat)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)

    boot_mean, ci_lower, ci_upper = bootstrap_testset(test_preds, y_test, seed=seed)

    logging.info(f'Bootstrap Test RMSE: {boot_mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]')

    if file_handler:
        cleanup_logging(file_handler)
    return model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper