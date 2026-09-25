import os
import glob
import re
from pathlib import Path


def resolve_log_dir(model_class: str, dataset_id: str, logs_root: str = "logs") -> str:
    """Find the most recent log directory for a (model_class, dataset_id) pair.

    Handles both flat datasets (e.g. 'BeijingPM10Quality') and nested ones
    (e.g. 'REVS/2013_Targa_Sixty_Six') where the log structure is:
        logs/LSTM_REVS/2013_Targa_Sixty_Six_<timestamp>/
    vs:
        logs/LSTM_BeijingPM10Quality_<timestamp>/
    """
    if "/" in dataset_id:
        # Nested dataset: logs/<Model>_<parent>/<child>_<timestamp>/
        parent, child = dataset_id.split("/", 1)
        base_dir = os.path.join(logs_root, f"{model_class}_{parent}")
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(
                f"No log directory found at '{base_dir}' for {model_class}/{dataset_id}"
            )
        # Match child directories: <child>_<timestamp>
        pattern = os.path.join(base_dir, f"{child}_*")
        candidates = sorted(glob.glob(pattern))
    else:
        # Flat dataset: logs/<Model>_<dataset>_<timestamp>/
        pattern = os.path.join(logs_root, f"{model_class}_{dataset_id}_*")
        candidates = sorted(glob.glob(pattern))

    # Filter to actual directories
    candidates = [c for c in candidates if os.path.isdir(c)]

    if not candidates:
        raise FileNotFoundError(
            f"No log directory found for model='{model_class}', "
            f"dataset='{dataset_id}' in '{logs_root}'"
        )

    # Return the last one (most recent timestamp in lexicographic order)
    return candidates[-1]


def find_nn_checkpoint(log_dir: str) -> str:
    """Return the path to best_model.pth inside a log directory."""
    path = os.path.join(log_dir, "best_model.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
    return path


def find_tree_checkpoint(log_dir: str) -> str:
    """Return the path to model.joblib inside a log directory."""
    path = os.path.join(log_dir, "model.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
    return path


def load_config_from_log(log_dir: str) -> dict:
    """Load the config.yaml stored in a log directory."""
    import yaml
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.yaml found at '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)