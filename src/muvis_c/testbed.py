import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from muvis_c._device import resolve_device
from muvis_c._data import load_dataset
from muvis_c._eval import run_sweep
from muvis_c.failures import SeverityFailure, build_all_failures
from muvis_c.metrics.base import Metric
from muvis_c.results import Results

BENCHMARK_DATASETS = [
    "BeijingPM10Quality",
    "BeijingPM25Quality",
    "Panasonic18650PFData",
    "PPGDalia",
    "REVS/2013_Monterey_Motorsports_Reunion",
    "REVS/2013_Targa_Sixty_Six",
    "REVS/2014_Targa_Sixty_Six",
    "TennesseeEastmanProcess",
    "VehicleDynamicsDataset",
]


class Testbed:
    """Declarative testbed for severity-based robustness evaluation.

    Each Testbed evaluates one dataset.  Use :meth:`Results.merge` to
    combine results across datasets.

    Parameters
    ----------
    dataset : str | Path | tuple | None
        The dataset to evaluate.  Accepts a benchmark dataset ID
        (e.g. ``rob.PPGDalia``), a path to a directory with ``train.ts``
        / ``test.ts``, or a tuple ``(X_train, y_train, X_test, y_test)``
        or ``(X_train, y_train, X_test, y_test, name)``.
        ``None`` (default) means no dataset yet — call
        :meth:`add_dataset` before :meth:`run`.
    data_root : str | Path | None
        Base directory for resolving dataset IDs.  ``None`` (default) uses
        ``MUVIS_C_DATA_DIR`` env var or ``~/.muvis_c/data/``.
    failures : list[SeverityFailure] | None
        Explicit list of failure modes to use.  ``None`` (default) uses all
        10 built-in failure modes.  Mutually exclusive with *exclude_failures*.
    exclude_failures : list[type[SeverityFailure]] | None
        Failure mode **classes** to exclude from the defaults (e.g.
        ``exclude_failures=[rob.Outliers]``).  Mutually exclusive with
        *failures*.
    target_features : list[int] | str
        Feature indices to corrupt, or ``"all"`` (default).
    severity_steps : int
        Number of severity steps in [0, 1] (default 20 -> 21 evaluation points).
    k : float
        Default perturbation scale for failures (default 3.0).
    seed : int
        Random seed (default 42).
    device : str
        ``"auto"`` (default) to auto-detect, or ``"cuda"``/``"mps"``/``"cpu"``.
    scaler
        A scikit-learn-style scaler instance (default ``StandardScaler()``).
        Pass ``None`` to skip scaling.
    """

    def __init__(
        self,
        dataset=None,
        data_root=None,
        failures=None,
        exclude_failures=None,
        target_features="all",
        severity_steps=20,
        k=3.0,
        seed=42,
        device="auto",
        scaler=None,
        batch_size=256,
    ):
        if failures is not None and exclude_failures is not None:
            raise ValueError(
                "Cannot specify both failures= and exclude_failures=. "
                "Use failures= to replace the defaults, or "
                "exclude_failures= to remove specific modes."
            )

        if k == 0:
            logging.warning("k=0 collapses most failure modes to no-ops.")
        elif k == 1:
            logging.warning("k=1 sets scaling mode gains to 1; consider higher k.")

        self._dataset = dataset
        from muvis_c._data import _resolve_data_root
        self.data_root = _resolve_data_root(data_root)
        self.severity_steps = severity_steps
        self.k = k
        self.seed = seed
        self.device = resolve_device(device)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.batch_size = batch_size

        # Resolve failure modes
        if failures is not None:
            for f in failures:
                if not isinstance(f, SeverityFailure):
                    raise TypeError(
                        f"Expected a SeverityFailure instance, got {type(f).__name__}. "
                        "Subclass muvis_c.SeverityFailure to create a custom failure."
                    )
            self._failures = list(failures)
        else:
            all_failures = build_all_failures(k=k)
            if exclude_failures is not None:
                self._failures = [
                    f for f in all_failures
                    if type(f) not in exclude_failures
                ]
            else:
                self._failures = all_failures

        self._target_features = target_features
        self._extra_metrics = []
        self._models = {}

    # -- Configuration methods -----------------------------------------

    def add_metric(self, metric):
        """Add an extra evaluation metric.

        Extra metrics add additional columns (named after ``metric.name``)
        to the raw DataFrame.

        Parameters
        ----------
        metric : Metric
            An instance of a :class:`~muvis_c.metrics.Metric` subclass.
        """
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Expected a Metric instance, got {type(metric).__name__}. "
                "Subclass muvis_c.Metric to create a custom metric."
            )
        self._extra_metrics.append(metric)

    def add_failure(self, failure):
        """Add an extra failure mode on top of the current set.

        Parameters
        ----------
        failure : SeverityFailure
            A :class:`~muvis_c.failures.SeverityFailure` instance.
        """
        if not isinstance(failure, SeverityFailure):
            raise TypeError(
                f"Expected a SeverityFailure instance, got {type(failure).__name__}. "
                "Subclass muvis_c.SeverityFailure to create a custom failure."
            )
        self._failures.append(failure)

    def add_dataset(self, X_train, y_train, X_test, y_test, name="custom"):
        """Set the dataset from numpy arrays.

        Replaces any previously configured dataset.

        Parameters
        ----------
        X_train, y_train, X_test, y_test : numpy.ndarray
            Training and test data.  ``X`` arrays must have shape
            ``(N, T, C)``, ``y`` arrays shape ``(N,)``.
        name : str
            Display name for the dataset in results (default ``"custom"``).
        """
        if self._dataset is not None:
            logging.info("Replacing previously configured dataset.")
        self._dataset = (X_train, y_train, X_test, y_test, name)

    def add_model(self, name, model=None, predict_fn=None):
        """Register a trained model to benchmark.

        Provide exactly one of *model* or *predict_fn*.

        Parameters
        ----------
        name : str
            Display name for the model in results.
        model : nn.Module | sklearn-like, optional
            A PyTorch model (with ``.eval()`` / ``.parameters()``) or an
            sklearn-style object with a ``.predict()`` method.
        predict_fn : callable, optional
            A function ``(X: np.ndarray) -> np.ndarray`` that takes a
            3-D array ``(N, T, C)`` and returns predictions ``(N,)``.
        """
        n_provided = sum(x is not None for x in [model, predict_fn])
        if n_provided != 1:
            raise ValueError(
                "Provide exactly one of model= or predict_fn=."
            )
        self._models[name] = {
            "model": model,
            "predict_fn": predict_fn,
        }

    # -- Execution -----------------------------------------------------

    def run(self, log_dir=None):
        """Execute the robustness sweep.

        Evaluates all registered models x failures x features x severity
        steps on the configured dataset.

        Parameters
        ----------
        log_dir : str | Path | None
            If provided, saves a log file and config snapshot.

        Returns
        -------
        Results
            A :class:`~muvis_c.results.Results` object wrapping the raw
            per-severity DataFrame.
        """
        if self._dataset is None:
            raise RuntimeError(
                "No dataset configured. Pass dataset= to the constructor "
                "or call testbed.add_dataset(...) before run()."
            )
        if not self._failures:
            raise RuntimeError(
                "No failure modes registered. This should not happen with "
                "default settings — did you pass failures=[] explicitly?"
            )
        if not self._models:
            raise RuntimeError(
                "No models registered. "
                "Call testbed.add_model(...) before run()."
            )

        all_rows = run_sweep(
            dataset=self._dataset,
            data_root=self.data_root,
            models=self._models,
            failures=self._failures,
            target_features=self._target_features,
            extra_metrics=self._extra_metrics,
            severity_steps=self.severity_steps,
            seed=self.seed,
            device=self.device,
            scaler=self.scaler,
            batch_size=self.batch_size,
            log_dir=log_dir,
        )

        return Results(raw_rows=all_rows)
