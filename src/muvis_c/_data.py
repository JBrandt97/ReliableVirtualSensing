"""dataset loading helpers.

Supports three input forms:
1. A dataset ID string (resolved under *data_root* via MuViSDataset)
2. A path to a directory containing ``train.ts`` / ``test.ts``
3. A raw tuple ``(X_train, y_train, X_test, y_test)`` of numpy arrays

Data root resolution order:
1. Explicit ``data_root`` parameter
2. ``MUVIS_C_DATA_DIR`` environment variable
3. ``~/.muvis_c/data/`` (default cache directory)
"""

import os
from pathlib import Path

import numpy as np


def _resolve_data_root(data_root=None):
    """Resolve the data root directory.

    Priority: explicit arg > env var > default cache dir.
    """
    if data_root is not None:
        return str(data_root)

    env = os.environ.get("MUVIS_C_DATA_DIR")
    if env:
        return env

    return str(Path.home() / ".muvis_c" / "data")


def load_dataset(source, data_root=None):
    """Load a single dataset.

    Parameters
    ----------
    source : str | Path | tuple
        - ``str`` / ``Path`` pointing to a directory with train.ts / test.ts
        - ``str`` dataset ID (resolved as ``<data_root>/<source>``)
        - ``tuple`` of ``(X_train, y_train, X_test, y_test)`` numpy arrays

    data_root : str | Path | None
        Root directory for resolving dataset IDs.  If *None*, uses
        ``MUVIS_C_DATA_DIR`` env var or defaults to ``~/.muvis_c/data/``.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        All X arrays have shape ``(N, T, C)``.
    dataset_id : str
        Human-readable identifier for the dataset.
    """
    if isinstance(source, tuple):
        if len(source) == 5:
            X_train, y_train, X_test, y_test, name = source
            return X_train, y_train, X_test, y_test, name
        if len(source) == 4:
            X_train, y_train, X_test, y_test = source
            return X_train, y_train, X_test, y_test, "custom"
        raise ValueError(
            "Tuple datasets must have 4 elements "
            "(X_train, y_train, X_test, y_test) or 5 elements "
            "(X_train, y_train, X_test, y_test, name)."
        )

    source = str(source)
    path = Path(source)

    # If source looks like an existing directory with .ts files, use it directly
    if path.is_dir() and (path / "train.ts").exists():
        dataset_id = path.name
        from muvis_c._dataset import MuViSDataset
        ds = MuViSDataset(dataset_id, base_path=str(path.parent))
        X_train, y_train = ds.get_data(split="train")
        X_test, y_test = ds.get_data(split="test")
        return X_train, y_train, X_test, y_test, dataset_id

    # Otherwise treat as dataset ID under data_root
    resolved_root = _resolve_data_root(data_root)
    resolved = Path(resolved_root) / source
    if resolved.is_dir() and (resolved / "train.ts").exists():
        from muvis_c._dataset import MuViSDataset
        ds = MuViSDataset(source, base_path=str(resolved_root))
        X_train, y_train = ds.get_data(split="train")
        X_test, y_test = ds.get_data(split="test")
        return X_train, y_train, X_test, y_test, source

    # Try downloading (will raise NotImplementedError until Zenodo is set up)
    from muvis_c._download import download_dataset
    download_note = ""
    try:
        dest = download_dataset(source, resolved_root)
        from muvis_c._dataset import MuViSDataset
        ds = MuViSDataset(source, base_path=str(resolved_root))
        X_train, y_train = ds.get_data(split="train")
        X_test, y_test = ds.get_data(split="test")
        return X_train, y_train, X_test, y_test, source
    except NotImplementedError as e:
        download_note = (
            "\n\nAutomatic download is not yet available "
            "(Zenodo deposit coming soon). "
            "Follow the 'Dataset Preparation' section of the README to "
            "preprocess the raw sources locally."
        )

    raise FileNotFoundError(
        f"Cannot resolve dataset '{source}'. Checked:\n"
        f"  - as directory: {path}\n"
        f"  - under data_root: {resolved}\n"
        "Provide a path to a directory with train.ts/test.ts, "
        "a dataset ID resolvable under data_root, or a raw "
        "(X_train, y_train, X_test, y_test) tuple.\n\n"
        "Tip: set MUVIS_C_DATA_DIR to point to your processed data, "
        "or pass data_root= explicitly."
        + download_note
    )
