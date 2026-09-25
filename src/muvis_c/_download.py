"""Dataset download helpers.
"""

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry — will be populated once Zenodo deposit is created
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, dict[str, str]] = {}


def _verify_hash(path: str, expected_sha256: str) -> None:
    """Verify SHA-256 hash of a downloaded file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Hash mismatch for {path}:\n"
            f"  expected: {expected_sha256}\n"
            f"  got:      {actual}"
        )


def download_dataset(dataset_id: str, target_dir: str) -> Path:
    """Download processed .ts files for *dataset_id* into *target_dir*.

    Returns the path to the dataset directory (``target_dir / dataset_id``).
    """
    if dataset_id not in DATASET_REGISTRY:
        raise NotImplementedError(
            f"Automatic download for '{dataset_id}' is not yet available.\n"
            "Please download and preprocess the dataset manually.\n"
            "See the README for instructions."
        )

    meta = DATASET_REGISTRY[dataset_id]
    dest = Path(target_dir) / dataset_id
    if dest.exists() and (dest / "train.ts").exists():
        return dest

    url = meta["url"]
    sha256 = meta["sha256"]

    os.makedirs(target_dir, exist_ok=True)
    tmp = str(dest) + ".tar.gz"

    print(f"Downloading {dataset_id} from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    _verify_hash(tmp, sha256)

    with tarfile.open(tmp) as tar:
        tar.extractall(target_dir)
    os.remove(tmp)

    return dest
