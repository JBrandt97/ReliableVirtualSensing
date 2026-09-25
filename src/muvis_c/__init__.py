"""MuViS-RB: Multivariate Virtual Sensing Robustness Benchmark.

Quick start::

    import muvis_c as rob

    testbed = rob.Testbed(dataset=rob.PPGDalia)
    testbed.add_model("MyModel", predict_fn=my_predict_fn)
    results = testbed.run()
    results.summary()

Multi-dataset aggregation::

    all_results = []
    for ds_id in rob.BENCHMARK_DATASETS:
        tb = rob.Testbed(dataset=ds_id)
        tb.add_model("MyModel", predict_fn=load_model(ds_id))
        all_results.append(tb.run())
    combined = rob.Results.merge(all_results)
    combined.summary()
"""

from muvis_c.testbed import Testbed, BENCHMARK_DATASETS
from muvis_c.results import Results
from muvis_c.metrics.base import Metric
from muvis_c.failures.base import SeverityFailure
from muvis_c._prepare import prepare_dataset
from muvis_c._benchmarks import DATASET_INFO, dataset_info

# ── Dataset ID constants ──────────────────────────────────────────────
BeijingPM10Quality: str = "BeijingPM10Quality"
BeijingPM25Quality: str = "BeijingPM25Quality"
Panasonic18650PFData: str = "Panasonic18650PFData"
PPGDalia: str = "PPGDalia"
REVS_Monterey2013: str = "REVS/2013_Monterey_Motorsports_Reunion"
REVS_Targa2013: str = "REVS/2013_Targa_Sixty_Six"
REVS_Targa2014: str = "REVS/2014_Targa_Sixty_Six"
TennesseeEastmanProcess: str = "TennesseeEastmanProcess"
VehicleDynamicsDataset: str = "VehicleDynamicsDataset"

# ── Failure mode aliases ──────────────────────────────────────────────
# Short names for use in ``Testbed(failures=[...])`` or ``exclude_failures=[...]``.
from muvis_c.failures.modes import (
    BiasSeverity as Bias,
    NoiseSeverity as Noise,
    ScalingSeverity as Scaling,
    TimeVaryingScalingSeverity as TimeVaryingScaling,
    LinearDriftSeverity as LinearDrift,
    NonlinearDriftSeverity as NonlinearDrift,
    OutliersSeverity as Outliers,
    TrimmingVaryingSeverity as TrimmingVarying,
    TrimmingConstantSeverity as TrimmingConstant,
    HardFaultSeverity as HardFault,
)

__all__ = [
    # Core API
    "Testbed",
    "BENCHMARK_DATASETS",
    "Results",
    "Metric",
    "SeverityFailure",
    "prepare_dataset",
    "DATASET_INFO",
    "dataset_info",
    # Dataset IDs
    "BeijingPM10Quality",
    "BeijingPM25Quality",
    "Panasonic18650PFData",
    "PPGDalia",
    "REVS_Monterey2013",
    "REVS_Targa2013",
    "REVS_Targa2014",
    "TennesseeEastmanProcess",
    "VehicleDynamicsDataset",
    # Failure mode aliases
    "Bias",
    "Noise",
    "Scaling",
    "TimeVaryingScaling",
    "LinearDrift",
    "NonlinearDrift",
    "Outliers",
    "TrimmingVarying",
    "TrimmingConstant",
    "HardFault",
]

__version__ = "0.1.0"
