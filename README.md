# Reliable Virtual Sensing: A Multi-Domain Benchmark for Robustness Under Sensor Failures

**MuViS-C** is a reproducible benchmark for evaluating virtual sensing models under realistic sensor failures. It extends the MuViS nominal-performance benchmark with a systematic robustness evaluation: nine public time-series datasets across six domains (environmental, physiological, chemical-process, battery, motorsport, and automotive), paired with ten parametric failure modes, bias, drift, noise, clipping, outliers, and hard faults, swept over a continuous severity axis. Three complementary measures (mPC, rPC, s_cross) capture average error under corruption, relative degradation, and worst-case fragility.

The benchmark is distributed as a lightweight, pip-installable evaluation library (`muvis-c`) and a separate `experiments/` tree that reproduces the paper experiments. It is open-source and extensible to new datasets, failure modes, measures, and models.

Paper: *Reliable Virtual Sensing: A Multi-Domain Benchmark for Robustness Under Sensor Failures* - https://arxiv.org/abs/2609.18396

---

## Quick Start — Evaluate Your Model

```bash
git clone <REPO_URL> && cd muvis-c
pip install -e .
```

```python
import muvis_c as rob

testbed = rob.Testbed(dataset=rob.PPGDalia)
testbed.add_model("MyModel", predict_fn=my_predict_fn)

results = testbed.run()
results.summary()
```

The `predict_fn` passed to `add_model` must accept a single numpy array of shape `(N, T, C)` and return predictions of shape `(N,)`:

```python
def predict_fn(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) -> predictions: (N,)"""
    ...
```

All 10 failure modes are applied by default. The Testbed expects the processed `.ts` arrays on disk — see [Dataset Preparation](#dataset-preparation-manual) below to generate them from the public sources.

### More examples

```python
# Single dataset
testbed = rob.Testbed(dataset=rob.PPGDalia)

# Only specific failures
testbed = rob.Testbed(dataset=rob.REVS_Targa2013, failures=[rob.Bias(), rob.Noise(), rob.HardFault()])

# All failures except outliers
testbed = rob.Testbed(dataset=rob.REVS_Targa2013, exclude_failures=[rob.Outliers])

# Add a custom failure on top of the defaults
testbed.add_failure(MyCustomFailure(k=5.0))

# Add extra metrics alongside built-in RMSE
testbed.add_metric(MAEMetric())

# Manual data path (if not using auto-download)
testbed = rob.Testbed(dataset=rob.PPGDalia, data_root="/path/to/data")

# Multi-dataset aggregation
all_results = []
for ds_id in rob.BENCHMARK_DATASETS:
    tb = rob.Testbed(dataset=ds_id)
    tb.add_model("MyModel", predict_fn=load_model(ds_id))
    all_results.append(tb.run())
combined = rob.Results.merge(all_results)
combined.summary()
```

### Evaluate on your own dataset

```python
import muvis_c as rob
import pandas as pd

# Convert a DataFrame into windowed (N, T, C) arrays
df = pd.read_csv("my_sensor_data.csv")
X_train, y_train, X_test, y_test = rob.prepare_dataset(
    df,
    target="temperature",    # column to predict
    window_size=24,          # timesteps per sample
    test_split=0.2,          # temporal split (last 20% = test)
)

testbed = rob.Testbed(dataset=(X_train, y_train, X_test, y_test, "MySensors"))
testbed.add_model("MyModel", predict_fn=my_fn)

results = testbed.run()
results.summary()
```

If you already have numpy arrays in `(N, T, C)` shape, skip `prepare_dataset`:

```python
testbed = rob.Testbed()
testbed.add_dataset(X_train, y_train, X_test, y_test, name="MyData")
```

### Data resolution

When you pass a dataset ID (e.g. `rob.PPGDalia`), the Testbed looks for data in this order:

1. Explicit `data_root=` parameter
2. `MUVIS_C_DATA_DIR` environment variable
3. `~/.muvis_c/data/` (default cache; a Zenodo-hosted auto-download will populate it once the deposit is public)

### Available datasets

| Constant | Dataset | Domain | Source |
|---|---|---|---|
| `rob.BeijingPM10Quality` | Beijing PM10 Air Quality | Air quality | [Zenodo](https://zenodo.org/records/3902667) |
| `rob.BeijingPM25Quality` | Beijing PM2.5 Air Quality | Air quality | [Zenodo](https://zenodo.org/records/3902671) |
| `rob.Panasonic18650PFData` | Panasonic 18650PF Battery | Battery SoC | [Mendeley](https://data.mendeley.com/datasets/xf68bwh54v/1) |
| `rob.PPGDalia` | PPG-DaLiA Heart Rate | Wearable / Bio | [UCI](https://archive.ics.uci.edu/dataset/495/ppg+dalia) |
| `rob.REVS_Monterey2013` | REVS 2013 Monterey Motorsports Reunion | Motorsport | [Stanford](https://purl.stanford.edu/tt103jr6546) |
| `rob.REVS_Targa2013` | REVS 2013 Targa Sixty Six | Motorsport | [Stanford](https://purl.stanford.edu/yf219gg2055) |
| `rob.REVS_Targa2014` | REVS 2014 Targa Sixty Six | Motorsport | [Stanford](https://purl.stanford.edu/hd122pw0365) |
| `rob.TennesseeEastmanProcess` | Tennessee Eastman Process | Chemical | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) |
| `rob.VehicleDynamicsDataset` | Vehicle Dynamics | Automotive | [Stanford](https://exhibits.stanford.edu/data/catalog/hh613qz0317) |

All constituent datasets are redistributed under their original licenses; see **Data Licensing** below.

### Available failure modes

| Constant | Failure Mode | Description |
|---|---|---|
| `rob.Bias` | Additive (const.) | Constant offset  |
| `rob.Noise` | Additive (stochastic) | Random noise|
| `rob.Scaling` | Multipl. (const.) | Scaling effect|
| `rob.TimeVaryingScaling` | Multipl.\ (time-var.)| Scale grows over time |
| `rob.LinearDrift` | Drift (linear)| Linear ramp |
| `rob.NonlinearDrift` | Drift (quadratic) | Quadratic ramp |
| `rob.Outliers` | Sporadic spikes| Random spikes |
| `rob.TrimmingVarying` | Saturation (soft)| Damped clipping with narrowing bounds |
| `rob.TrimmingConstant` | Saturation (hard) | Hard clipping with narrowing bounds |
| `rob.HardFault` | Stuck-at | Sensor frozen|

### Results API

```python
results = testbed.run()

results.summary()                                      # nominal / mPC / rPC / s_cross per model
results.raw                                            # full per-severity DataFrame
results.nominal_rmse()                                 # RMSE_clean pivot (dataset × model)
results.mpc()                                          # mean perf. under corruption pivot
results.rpc()                                          # relative perf. under corruption pivot
results.rpc_per_failure()                              # rPC pivot (failure_mode × model)
results.crossing_severity()                            # severity at which model crosses baseline
results.to_csv("output/")                              # save per_severity.csv
results.plot_degradation(model="MyModel", dataset="PPGDalia")  # RMSE(s) curves
```

---

## Reproduce the Paper

The `experiments/` directory contains all training scripts, configs, and model architectures used in the paper. The reported results were produced on a single NVIDIA H100 (80 GB) GPU.

### Setup

```bash
git clone <REPO_URL> && cd MuViS-C
```

Create and activate an environment — either venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

or conda:

```bash
conda create -n muvis-c python=3.13 -y
conda activate muvis-c
```

Then install the package together with the experiments extra:

```bash
pip install -e ".[experiments]"
```

### Dataset Preparation (manual)

`reproduce_all.sh` (and the underlying `train_models.sh` / `run_severity.sh`) invoke the preprocessing and severity configs with their defaults, which expect the raw sources at **`data/raw/`** and the preprocessed `.ts` files at **`data/processed/<DatasetID>/`** relative to the repository root. Place data at exactly these paths to reproduce without editing any configs. Until the Zenodo deposit with the preprocessed arrays is public, download the raw datasets and preprocess them locally:

1. Download each dataset and place it in `data/raw/` so the final layout matches exactly:

    ```
    data/raw/
    ├── BeijingPM10Quality/
    │   ├── BeijingPM10Quality_TRAIN.ts       
    │   └── BeijingPM10Quality_TEST.ts
    ├── BeijingPM25Quality/
    │   ├── BeijingPM25Quality_TRAIN.ts       
    │   └── BeijingPM25Quality_TEST.ts
    ├── Panasonic18650PFData/                 
    │   ├── Train/1/*.mat                     
    │   ├── Validation/*.mat
    │   ├── Test/Test.mat
    │   └── Normalization/                   
    ├── PPGDalia/
    │   └── PPG_FieldStudy/                  
    │       ├── S1/ ... S15/                  
    │       └── PPG_FieldStudy_readme.pdf
    ├── REVS/                                 
    │   ├── 2013_Monterey_Motorsports_Reunion/*.csv
    │   ├── 2013_Targa_Sixty_Six/*.csv
    │   └── 2014_Targa_Sixty_Six/*.csv
    ├── TennesseeEastmanProcess/              
    │   ├── TEP_FaultFree_Training.RData
    │   └── TEP_FaultFree_Testing.RData
    └── VehicleDynamicsDataset/               
        ├── Oct2023/VehicleDynamicsDataset_Oct2023_*.csv
        └── Nov2023/VehicleDynamicsDataset_Nov2023_*.csv
    ```

    Source links, see [Available datasets](#available-datasets). For Panasonic18650PFData, download `Panasonic_NCR18650PF_Data_Normalized.zip`, extract, and rename the extracted folder to `Panasonic18650PFData/`



2. Run preprocessing:

    ```bash
    python -m experiments.data.preprocess
    ```

    This converts raw data into standardized `.ts` files under `data/processed/<DatasetID>/{train,test}.ts`, which is the location `reproduce_all.sh` reads from.

### Run everything

```bash
bash experiments/scripts/reproduce_all.sh
```

Or, on a SLURM cluster, submit the matching example job (adapt the `#SBATCH` directives and conda environment name in [`experiments/scripts/slurm_example.job`](experiments/scripts/slurm_example.job) to your setup first):

```bash
sbatch experiments/scripts/slurm_example.job
```

Or step by step:

```bash
# 1. Train every (model × dataset) combination.
bash experiments/scripts/train_models.sh

# 2. Run severity evaluation for every model × dataset in severity.yaml
bash experiments/scripts/run_severity.sh
```

### Single experiment

```bash
python -m experiments.main single --runconf experiments/configs/training/PPGDalia/TST.yaml
```

### Batch experiments

```bash
python -m experiments.main batch \
  --configs \
    experiments/configs/training/PPGDalia/TST.yaml \
    experiments/configs/training/PPGDalia/ModernTCN.yaml \
  --output results.csv
```

### Severity evaluation

```bash
# All models x all datasets
python -m experiments.main severity --config experiments/configs/severity.yaml

# Subset (CLI overrides the YAML)
python -m experiments.main severity --config experiments/configs/severity.yaml \
    --models PGD ISensD F2F --datasets BeijingPM10Quality
```

### Notebooks

Guided walkthroughs live in [`notebooks/`](notebooks/):
- [`load_data.ipynb`](notebooks/load_data.ipynb) — load data to train your own model outside the Testbed API
- [`quickstart.ipynb`](notebooks/quickstart.ipynb) — test your model
- [`custom_failure_and_metric.ipynb`](notebooks/custom_failure_and_metric.ipynb) — extending the benchmark


---

## Extend the Benchmark

### Add a dataset

1. Place raw data in `data/raw/<YourDataset>/`
2. Add a converter in [`experiments/data/converters.py`](experiments/data/converters.py) (subclass `BaseConverter`, implement `load_raw()`)
3. Register it in [`experiments/data/preprocess.py`](experiments/data/preprocess.py)
4. Run: `python -m experiments.data.preprocess --dataset <YourDataset>`
5. Add training configs under `experiments/configs/training/<YourDataset>/`
6. Add the dataset ID to `experiments/configs/severity.yaml`

### Add a model

1. Define the architecture under [`experiments/models/`](experiments/models/) (NN) or add to the tree-model registry in [`experiments/train/`](experiments/train/)
2. Create config YAMLs under `experiments/configs/training/<Dataset>/<Model>.yaml`
3. Add to `model_classes` in `experiments/configs/severity.yaml`

### Add a robustification strategy

To benchmark a new training strategy (analogous to F2F, PGD, or ISensD):

1. Add a new `run_<method>.py` under [`experiments/train/`](experiments/train/) exposing `run_experiment(conf, log_level=...)` that returns `(model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper)` and writes `best_model.pth` + `config.yaml` to `logs/<Method>_<dataset_id>_<timestamp>/`.
2. Register the method in [`experiments/main.py`](experiments/main.py) under both `run_single_experiment` and `run_multiple_experiments`, matching a new `experiment_type` key.
3. Add the corresponding inference branch to `build_predict_fn` in [`experiments/utils/model_helpers.py`](experiments/utils/model_helpers.py) if it needs special handling at evaluation time (otherwise the default `nn`/`f2f`/`pgd`/`isensd` branch covers it).
4. Create per-dataset training configs under `experiments/configs/training/<Dataset>/<Method>.yaml` with `experiment_type: <method>`.
5. Add the method name to `model_classes` in `experiments/configs/severity.yaml`.

### Add a failure mode

```python
import muvis_c as rob

class MyFailure(rob.SeverityFailure):
    def apply(self, X, feature_idx, severity):
        out = X.clone()
        # corrupt out[:, :, feature_idx] based on severity in [0, 1]
        return out

testbed = rob.Testbed(dataset=rob.PPGDalia)
testbed.add_failure(MyFailure(k=3.0))
```

### Add a metric

```python
class MAEMetric(rob.Metric):
    name = "mae"
    def compute(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

testbed.add_metric(MAEMetric())
```

---

## Repository Structure

```
src/muvis_c/              # pip install muvis-c
  testbed.py                # Testbed API
  results.py                # Results: summary(), plot_*, to_csv()
  _eval.py                  # Core severity sweep loop
  _data.py                  # Dataset loading + cache resolution
  _download.py              # Zenodo auto-download (coming soon)
  failures/                 # 10 built-in sensor failure modes
  metrics/                  # Metric ABC + RMSE

experiments/                # Paper reproduction (not shipped in the package)
  main.py                   # CLI: single | batch | severity
  models/                   # Model architectures (TST, ModernTCN, xLSTMMixer, PatchTSMixer, F2F)
  train/                    # Training scripts (run_nn, run_f2f, run_pgd, run_isensd, run_tree)
  eval/                     # Severity evaluation wrapper
  utils/                    # Checkpoint resolution, model loading, bootstrap
  data/                     # Dataset converters + preprocessing
  configs/                  # YAML configs (training, severity)
  scripts/                  # Shell scripts for full reproduction

notebooks/                  # Worked examples (quickstart, custom model, ...)
```

---

## Data Availability and Licensing

- **Code.** Released under the [MIT License](LICENSE).
- **Preprocessed datasets.** Will be archived on Zenodo and auto-downloaded by the `muvis-c` package. In the meantime, see [Dataset Preparation](#dataset-preparation-manual) for local preprocessing from the original public sources.
- **Constituent datasets.** Each source dataset is redistributed under its original license; upstream `LICENSE`/`README` files will be preserved alongside the processed arrays. Please consult and cite the upstream source when using an individual dataset (links in the *Available datasets* table above).

We follow the FAIR data principles: preprocessed arrays will be **F**indable (DOI), **A**ccessible (Zenodo + HTTPS), **I**nteroperable (`.ts`  format), and **R**eusable (documented schemas and converters under [`experiments/data/converters.py`](experiments/data/converters.py)).

## Intended Use and Limitations

MuViS-C is designed as a **research benchmark** for evaluating the robustness of multivariate virtual sensing models to *sensor-level* degradations. It is **not** a certification tool: passing the benchmark does not guarantee safety in any deployed system. The included failure modes are a broad but non-exhaustive family of degradations; domain-specific failures may require additional, dataset-specific modelling. The constituent datasets cover a limited set of domains and populations; results should not be extrapolated outside each dataset's original collection context.

## Maintenance Plan

- **Hosting.** Source code on GitHub; preprocessed data on Zenodo (permanent DOI).
- **Versioning.** Semantic versioning for the `muvis-c` package; dataset versions are pinned in [`src/muvis_c/_download.py`](src/muvis_c/_download.py).
- **Issues and contributions.** Bug reports and benchmark extensions are tracked as GitHub issues; we welcome PRs adding datasets, failure modes, or baselines.

---

## Citation

If you use MuViS-C in your research, please cite:
https://arxiv.org/abs/2609.18396