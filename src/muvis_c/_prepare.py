"""Helper to convert tabular data into the (N, T, C) format expected by Testbed."""

import numpy as np


def prepare_dataset(
    df,
    target,
    window_size,
    test_split=0.2,
    stride=1,
    features=None,
):
    """Convert a pandas DataFrame into windowed train/test arrays.

    Takes a time-ordered DataFrame of sensor readings and produces the
    ``(N, T, C)`` arrays that :meth:`~muvis_c.Testbed.add_dataset` expects.

    Parameters
    ----------
    df : pandas.DataFrame
        Time-ordered rows.  Columns are sensor features + target.
    target : str
        Column name of the prediction target (y).
    window_size : int
        Number of timesteps per sample (T).
    test_split : float | int
        If ``float`` in (0, 1): fraction of rows used for the test set
        (split is temporal — last *test_split* fraction becomes test).
        If ``int``: absolute row index where the test set starts.
    stride : int
        Step size between consecutive windows (default 1).
    features : list[str] | None
        Feature column names.  ``None`` (default) uses all columns
        except *target*.

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy.ndarray
        ``X`` arrays have shape ``(N, T, C)``, ``y`` arrays have shape
        ``(N,)``.  The target value is taken from the **last** timestep
        of each window.

    Examples
    --------
    ::

        import pandas as pd
        import muvis_c as rob

        df = pd.read_csv("sensor_data.csv")
        X_train, y_train, X_test, y_test = rob.prepare_dataset(
            df, target="temperature", window_size=24, test_split=0.2,
        )

        testbed = rob.Testbed(dataset=(X_train, y_train, X_test, y_test, "MySensors"))
        testbed.add_model("MyModel", predict_fn=my_fn)
        results = testbed.run()
    """
    if features is None:
        features = [c for c in df.columns if c != target]

    if not features:
        raise ValueError("No feature columns found (all columns match target).")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    X_raw = df[features].to_numpy(dtype=np.float32)
    y_raw = df[target].to_numpy(dtype=np.float32)
    n_rows = len(df)

    # Temporal train/test split
    if isinstance(test_split, float):
        if not 0 < test_split < 1:
            raise ValueError("test_split as float must be in (0, 1).")
        split_idx = int(n_rows * (1 - test_split))
    else:
        split_idx = int(test_split)
        if not 0 < split_idx < n_rows:
            raise ValueError(
                f"test_split index {split_idx} out of range for {n_rows} rows."
            )

    def _window(X, y, start, end):
        windows_X = []
        windows_y = []
        for i in range(start, end - window_size + 1, stride):
            windows_X.append(X[i : i + window_size])
            windows_y.append(y[i + window_size - 1])
        if not windows_X:
            raise ValueError(
                f"No windows of size {window_size} fit in range "
                f"[{start}, {end}) with stride {stride}."
            )
        return np.stack(windows_X), np.array(windows_y, dtype=np.float32)

    X_train, y_train = _window(X_raw, y_raw, 0, split_idx)
    X_test, y_test = _window(X_raw, y_raw, split_idx, n_rows)

    return X_train, y_train, X_test, y_test
