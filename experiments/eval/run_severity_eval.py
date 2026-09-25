import os
from datetime import datetime

import numpy as np
import yaml

import muvis_c as rob
from muvis_c.failures import SEVERITY_REGISTRY

from experiments.utils.model_helpers import build_predict_fn


class MAEMetric(rob.Metric):
    name = "mae"

    def compute(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))


def run_severity_eval(conf, log_level="INFO"):
    """Run severity evaluation for one dataset from a config dict.

    Returns
    -------
    dict with a single key ``per_severity`` — a list of dicts (one per
    (model, failure_mode, feature, severity) row) for downstream
    DataFrame construction.
    """
    eval_conf = conf["Evaluation"]
    sev_conf = conf.get("severity", {})
    dataset_id = eval_conf["dataset_id"]

    exclude = [SEVERITY_REGISTRY[n] for n in sev_conf.get("exclude_failure_modes", [])]

    testbed = rob.Testbed(
        dataset=dataset_id,
        data_root=eval_conf.get("data_root", "data/processed"),
        severity_steps=sev_conf.get("n_steps", 20),
        k=sev_conf.get("k", 3.0),
        seed=eval_conf.get("seed", 42),
        device=eval_conf.get("device", "cpu"),
        exclude_failures=exclude or None,
        target_features=eval_conf.get("features_to_corrupt", "all"),
    )
    testbed.add_metric(MAEMetric())

    for model_class in eval_conf["model_classes"]:
        predict_fn = build_predict_fn(
            model_class, dataset_id,
            logs_root=eval_conf.get("logs_root", "logs"),
            device=testbed.device,
            batch_size=eval_conf.get("batch_size", 256),
        )
        testbed.add_model(name=model_class, predict_fn=predict_fn)

    log_dir = None
    if log_level:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", f"Severity_{dataset_id.replace('/', '_')}_{ts}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(conf, f)

    return {"per_severity": testbed.run(log_dir=log_dir).raw.to_dict("records")}
