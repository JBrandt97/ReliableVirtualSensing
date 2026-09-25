"""Static metadata for the MuViS-C benchmark datasets.

Allows package users to size model architectures against benchmark
datasets without loading the data. Values reflect the default
preprocessing in ``experiments/data/converters.py``; if you re-run
preprocessing with non-default ``sequence_length``, the ``seq_len``
entries here no longer match your local ``.ts`` files.

For custom datasets (or after custom preprocessing), read metadata
from the ``.ts`` header via ``dataset_info()`` below.
"""

from muvis_c._data import _resolve_data_root


DATASET_INFO: dict[str, dict] = {
    "BeijingPM10Quality":                     {"n_channels": 9,  "seq_len": 24,  "target": "PM10"},
    "BeijingPM25Quality":                     {"n_channels": 9,  "seq_len": 24,  "target": "PM25"},
    "Panasonic18650PFData":                   {"n_channels": 7,  "seq_len": 120, "target": "SOC"},
    "PPGDalia":                               {"n_channels": 6,  "seq_len": 512, "target": "heart_rate_bpm"},
    "REVS/2013_Monterey_Motorsports_Reunion": {"n_channels": 22, "seq_len": 20,  "target": "vyCG"},
    "REVS/2013_Targa_Sixty_Six":              {"n_channels": 22, "seq_len": 20,  "target": "vyCG"},
    "REVS/2014_Targa_Sixty_Six":              {"n_channels": 22, "seq_len": 20,  "target": "vyCG"},
    "TennesseeEastmanProcess":                {"n_channels": 33, "seq_len": 20,  "target": "xmeas_35"},
    "VehicleDynamicsDataset":                 {"n_channels": 11, "seq_len": 50,  "target": "tireTemp_fr_degC"},
}


def dataset_info(dataset_id: str, data_root=None) -> dict:
    """Return ``{n_channels, seq_len, target}`` for a dataset.

    For built-in benchmark IDs (see :data:`BENCHMARK_DATASETS`), returns
    the static entry from :data:`DATASET_INFO`.  For any other ID or path,
    reads metadata from the ``.ts`` header on disk.

    Parameters
    ----------
    dataset_id : str
        Benchmark dataset ID (e.g. ``"PPGDalia"``) or a path resolvable
        under ``data_root``.
    data_root : str | Path | None
        Base directory for resolving custom dataset IDs.  Falls back to
        ``MUVIS_C_DATA_DIR`` env var or ``~/.muvis_c/data/``.
    """
    if dataset_id in DATASET_INFO:
        return DATASET_INFO[dataset_id]

    from muvis_c._dataset import MuViSDataset
    ds = MuViSDataset(dataset_id, base_path=_resolve_data_root(data_root))
    meta = ds.get_meta_info()
    X, _ = ds.get_data(split="test")
    return {
        "n_channels": X.shape[2],
        "seq_len": X.shape[1],
        "target": meta.get("target_name"),
    }
