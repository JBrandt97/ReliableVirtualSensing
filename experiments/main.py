import argparse
import yaml
import sys
import logging

import gc
import torch

import pandas as pd
from pathlib import Path
from datetime import datetime
from experiments.train.run_nn import run_experiment as run_nn
from experiments.train.run_tree import run_experiment as run_tree
from experiments.train.run_f2f import run_experiment as run_f2f
from experiments.train.run_pgd import run_experiment as run_pgd
from experiments.train.run_isensd import run_experiment as run_isensd
from experiments.eval.run_severity_eval import run_severity_eval


def run_single_experiment(config_path, log_level="INFO"):
    """Run a single experiment."""
    print(f"Reading config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    
    exp_type = conf["experiment_type"]

    if exp_type == "nn":
        run_nn(conf, log_level=log_level)
    elif exp_type == "tree":
        run_tree(conf, log_level=log_level)
    elif exp_type == "f2f":
        run_f2f(conf, log_level=log_level)
    elif exp_type == "pgd":
        run_pgd(conf, log_level=log_level)
    elif exp_type == "isensd":
        run_isensd(conf, log_level=log_level)
    else:
        print(f"Error: Unknown experiment type '{exp_type}' in config.")
        sys.exit(1)

def run_multiple_experiments(config_paths, output_file=None, log_level='INFO'):
    """Run multiple experiments and collect results directly into a wide format."""
    wide_results = {}
    
    for config_path in config_paths:
        print(f"{'='*60}")
        print(f"Running: {config_path}")
        print(f"{'='*60}")
        
        with open(config_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        
        exp_type = conf["experiment_type"]
        
        if exp_type == "nn":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_nn(conf, log_level=log_level)
        elif exp_type == "tree":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_tree(conf, log_level=log_level)
        elif exp_type == "f2f":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_f2f(conf, log_level=log_level)
        elif exp_type == "pgd":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_pgd(conf, log_level=log_level)
        elif exp_type == "isensd":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_isensd(conf, log_level=log_level)
        else:
            raise ValueError(f"Unknown experiment type '{exp_type}'")
        
        model_name = Path(config_path).stem
        dataset_name = Path(config_path).parent.name
        
        print(f"✓ {model_name}: Test RMSE = {test_rmse:.4f}")
        
        if dataset_name not in wide_results:
            wide_results[dataset_name] = {}
            
        wide_results[dataset_name].update({
            f"{model_name}_test_rmse": test_rmse,
            f"{model_name}_boot_mean": boot_mean,
            f"{model_name}_ci_lower": ci_lower,
            f"{model_name}_ci_upper": ci_upper
        })
    
    wide_df = pd.DataFrame.from_dict(wide_results, orient='index')
    wide_df.index.name = 'dataset'
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{timestamp}.csv"
    
    wide_df.to_csv(output_file)

    return wide_df


def run_severity(config_path, output_dir=None, models=None, datasets=None,
                 log_level='INFO'):
    """Run severity-based robustness evaluation from a single config.

    The config contains a ``datasets`` list and shared ``Evaluation`` /
    ``severity`` sections.  Optional *models* and *datasets* CLI filters
    restrict which subset to evaluate without editing the config.

    Saves CSV: per_severity
    """
    with open(config_path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    dataset_ids = conf["datasets"]
    eval_conf = dict(conf["Evaluation"])
    sev_conf = conf.get("severity", {})

    # ── Apply CLI filters ─────────────────────────────────────────────────
    if datasets:
        unknown = set(datasets) - set(dataset_ids)
        if unknown:
            print(f"Error: unknown dataset(s): {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(dataset_ids)}")
            sys.exit(1)
        dataset_ids = [d for d in dataset_ids if d in datasets]

    if models:
        all_models = eval_conf["model_classes"]
        unknown = set(models) - set(all_models)
        if unknown:
            print(f"Error: unknown model(s): {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(all_models)}")
            sys.exit(1)
        eval_conf["model_classes"] = [m for m in all_models if m in models]

    # ── Prepare output directory ────────────────────────────────────────
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"severity_results_{timestamp}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "per_severity.csv"
    header_written = False

    # ── Run per-dataset evaluation ────────────────────────────────────────
    all_per_severity = []

    for dataset_id in dataset_ids:
        print(f"{'='*60}")
        print(f"Severity eval: {dataset_id}")
        print(f"{'='*60}")

        per_dataset_conf = {
            "Evaluation": {**eval_conf, "dataset_id": dataset_id},
            "severity": sev_conf,
        }

        result = run_severity_eval(per_dataset_conf, log_level=log_level)
        all_per_severity.extend(result["per_severity"])

        # Append this dataset's results to CSV immediately
        chunk_df = pd.DataFrame(result["per_severity"])
        chunk_df.to_csv(csv_path, mode='a', header=not header_written, index=False)
        header_written = True

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    per_sev_df = pd.DataFrame(all_per_severity)

    print(f"\nResults saved to {output_dir}/")
    print(f"  per_severity.csv  ({len(per_sev_df)} rows)")
    print("Tip: load the CSV with rob.Results(pd.read_csv(...)).summary() "
          "to print the per-model robustness table.")

    return per_sev_df


def main():
    parser = argparse.ArgumentParser(description='Run Experiments')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    single_parser = subparsers.add_parser('single', help='Run a single experiment')
    single_parser.add_argument('--runconf', type=str, required=True, 
                              help='Path to the YAML config file')
    
    batch_parser = subparsers.add_parser('batch', help='Run multiple experiments')
    batch_parser.add_argument('--configs', nargs='+', required=True,
                             help='Paths to YAML config files')
    batch_parser.add_argument('--output', type=str, default=None,
                             help='Output CSV file path (default: timestamped)')

    severity_parser = subparsers.add_parser('severity', help='Run severity-based robustness evaluation')
    severity_parser.add_argument('--config', type=str, required=True,
                                 help='Path to severity YAML config file')
    severity_parser.add_argument('--output_dir', type=str, default=None,
                                 help='Output directory for CSV files (default: severity_results_<timestamp>)')
    severity_parser.add_argument('--models', nargs='+', default=None,
                                 help='Run only these models (subset of model_classes in config)')
    severity_parser.add_argument('--datasets', nargs='+', default=None,
                                 help='Run only these datasets (subset of datasets in config)')

    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'None'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    if args.log_level == "None":
        args.log_level = None
    
    if args.command == 'single':
        run_single_experiment(args.runconf, log_level=args.log_level)
    elif args.command == 'batch':
        run_multiple_experiments(args.configs, args.output, log_level=args.log_level)
    elif args.command == 'severity':
        run_severity(args.config, args.output_dir, models=args.models,
                     datasets=args.datasets, log_level=args.log_level)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()