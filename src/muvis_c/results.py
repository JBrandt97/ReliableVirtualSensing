"""Results — per-severity DataFrame with robustness metric computation."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean


class Results:
    """Wraps the raw per-severity DataFrame and provides robustness metrics.

    The sweep produces rows for severities in ``[0, 1]``.  Rows at ``s=0``
    are used only to extract clean / baseline references and for the
    hard-fault-at-zero analysis.  All aggregate metrics (mPC, rPC, nPC,
    s_cross, …) are computed on the **corrupted subset** (``s > 0``)
    unless otherwise noted.

    Parameters
    ----------
    raw_rows : list[dict] | pd.DataFrame
        One dict per (model, dataset, failure_mode, feature_idx, severity).
    """

    def __init__(self, raw_rows):
        if isinstance(raw_rows, pd.DataFrame):
            self.raw = raw_rows
        else:
            self.raw = pd.DataFrame(raw_rows)

    @staticmethod
    def merge(results_list):
        """Merge multiple Results objects into one.

        Useful for combining per-dataset results into a single view::

            all_results = []
            for ds_id in rob.BENCHMARK_DATASETS:
                tb = rob.Testbed(dataset=ds_id)
                tb.add_model("MyModel", predict_fn=load_model(ds_id))
                all_results.append(tb.run())
            combined = rob.Results.merge(all_results)
            combined.summary()

        Parameters
        ----------
        results_list : list[Results]
            Results objects to merge (e.g. one per dataset).

        Returns
        -------
        Results
        """
        if not results_list:
            raise ValueError("Cannot merge an empty list of Results.")
        combined = pd.concat(
            [r.raw for r in results_list], ignore_index=True
        )
        return Results(combined)

    # ── Internal helpers ──────────────────────────────────────────────

    @property
    def _corrupted(self):
        """Rows with severity > 0 (excludes s=0 reference rows)."""
        return self.raw[self.raw["severity"] > 0]

    @property
    def _clean(self):
        """Clean RMSE per (model, dataset) — one row each."""
        return (self.raw
                .groupby(["model", "dataset"])["clean_boot_mean"]
                .first().reset_index())

    @property
    def _baseline(self):
        """Baseline (naive) RMSE per (model, dataset) — one row each."""
        return (self.raw
                .groupby(["model", "dataset"])["baseline_boot_mean"]
                .first().reset_index())

    # ── Metric methods ────────────────────────────────────────────────

    def nominal_rmse(self):
        """Nominal RMSE pivot table (dataset × model).

        Returns the clean (uncorrupted) bootstrap RMSE for each
        model / dataset combination.

        Returns
        -------
        pd.DataFrame
            Index = dataset, columns = model, values = RMSE.
        """
        return self._clean.pivot_table(
            index="dataset", columns="model", values="clean_boot_mean")

    def normalised_performance(self):
        """Normalised performance pivot (dataset × model).

        ``RMSE_clean / RMSE_baseline``.  Values < 1 mean the model
        outperforms the naive mean-predictor baseline.

        Returns
        -------
        pd.DataFrame
        """
        merged = self._clean.merge(self._baseline, on=["model", "dataset"])
        merged["norm_perf"] = (merged["clean_boot_mean"]
                               / merged["baseline_boot_mean"])
        return merged.pivot_table(
            index="dataset", columns="model", values="norm_perf")

    def mpc(self):
        """Mean Performance under Corruption (mPC) pivot (dataset × model).

        Mean of ``boot_mean`` across all corrupted rows (``s > 0``),
        i.e. averaged over failure modes, features, and severity steps.
        Reported in original RMSE units; lower is better.

        Returns
        -------
        pd.DataFrame
        """
        agg = (self._corrupted
               .groupby(["model", "dataset"])["boot_mean"]
               .mean().reset_index()
               .rename(columns={"boot_mean": "mPC"}))
        return agg.pivot_table(
            index="dataset", columns="model", values="mPC")

    def npc(self):
        """Normalised Performance under Corruption (nPC) pivot.

        ``nPC = mPC / RMSE_baseline``.  Scale-free; lower is better.

        Returns
        -------
        pd.DataFrame
        """
        agg = (self._corrupted
               .groupby(["model", "dataset"])["boot_mean"]
               .mean().reset_index()
               .rename(columns={"boot_mean": "mPC"}))
        merged = agg.merge(self._baseline, on=["model", "dataset"])
        merged["nPC"] = merged["mPC"] / merged["baseline_boot_mean"]
        return merged.pivot_table(
            index="dataset", columns="model", values="nPC")

    def rpc(self):
        """Relative Performance under Corruption (rPC) pivot.

        ``rPC = mPC / RMSE_clean``.  A value of 1.0 means no degradation;
        values > 1 signal performance loss under corruption. Lower is better.

        Returns
        -------
        pd.DataFrame
        """
        agg = (self._corrupted
               .groupby(["model", "dataset"])["boot_mean"]
               .mean().reset_index()
               .rename(columns={"boot_mean": "mPC"}))
        merged = agg.merge(self._clean, on=["model", "dataset"])
        merged["rPC"] = merged["mPC"] / merged["clean_boot_mean"]
        return merged.pivot_table(
            index="dataset", columns="model", values="rPC")

    def rpc_per_failure(self):
        """Per-failure-mode rPC pivot (failure_mode × model).

        For each failure mode, the mPC is computed across features and
        severity steps (``s > 0``), then divided by the clean RMSE.
        The result is averaged across datasets.

        Returns
        -------
        pd.DataFrame
            Index = failure_mode, columns = model.
        """
        mpc_fm = (self._corrupted
                  .groupby(["model", "dataset", "failure_mode"])["boot_mean"]
                  .mean().reset_index()
                  .rename(columns={"boot_mean": "mPC_fm"}))
        merged = mpc_fm.merge(self._clean, on=["model", "dataset"])
        merged["rPC_fm"] = merged["mPC_fm"] / merged["clean_boot_mean"]
        agg = (merged.groupby(["model", "failure_mode"])["rPC_fm"]
               .mean().reset_index())
        return agg.pivot_table(
            index="failure_mode", columns="model", values="rPC_fm")

    def crossing_severity(self):
        """Baseline crossing severity (s_cross) pivot (dataset × model).

        The lowest severity ``s > 0`` at which *any* (failure_mode, feature)
        combination causes the corrupted RMSE to reach or exceed the
        baseline (naive predictor) RMSE.  If no crossing occurs, returns NaN.
        Higher is better (model stays below baseline longer).

        Returns
        -------
        pd.DataFrame
        """
        records = []
        for (model, dataset), grp in self.raw.groupby(["model", "dataset"]):
            naive_rmse = grp["baseline_boot_mean"].iloc[0]
            s_cross = np.nan
            for (fm, feat), sub in grp.groupby(["failure_mode", "feature_idx"]):
                sub_s = sub[sub["severity"] > 0].sort_values("severity")
                above = sub_s[sub_s["boot_mean"] >= naive_rmse]
                if not above.empty:
                    s_cross = np.nanmin([s_cross, above["severity"].iloc[0]])
            records.append({
                "model": model,
                "dataset": dataset,
                "s_cross": s_cross,
            })
        df = pd.DataFrame(records)
        return df.pivot_table(
            index="dataset", columns="model", values="s_cross", dropna=False)

    def mpc_hard_fault_zero(self):
        """mPC for hard-fault at severity s=0 (dataset × model).

        Special case: the sensor is stuck at the channel mean (0 for
        z-score normalised data).  This evaluates robustness to complete
        loss of information on each channel individually.

        Returns
        -------
        pd.DataFrame
        """
        hf0 = self.raw[
            (self.raw["failure_mode"].isin(["hard_fault", "HardFaultSeverity"]))
            & (self.raw["severity"] == 0.0)
        ]
        agg = (hf0.groupby(["model", "dataset"])["boot_mean"]
               .mean().reset_index()
               .rename(columns={"boot_mean": "mPC"}))
        return agg.pivot_table(
            index="dataset", columns="model", values="mPC")

    def rpc_hard_fault_zero(self):
        """rPC for hard-fault at severity s=0 (dataset × model).

        ``rPC = mPC_hf0 / RMSE_clean``.

        Returns
        -------
        pd.DataFrame
        """
        hf0 = self.raw[
            (self.raw["failure_mode"].isin(["hard_fault", "HardFaultSeverity"]))
            & (self.raw["severity"] == 0.0)
        ]
        agg = (hf0.groupby(["model", "dataset"])["boot_mean"]
               .mean().reset_index()
               .rename(columns={"boot_mean": "mPC"}))
        merged = agg.merge(self._clean, on=["model", "dataset"])
        merged["rPC"] = merged["mPC"] / merged["clean_boot_mean"]
        return merged.pivot_table(
            index="dataset", columns="model", values="rPC")

    # ── Display ───────────────────────────────────────────────────────

    def summary(self):
        """Print a formatted summary of key robustness metrics."""
        nom = self.nominal_rmse()
        mpc_df = self.mpc()
        rpc_df = self.rpc()
        cs = self.crossing_severity()

        models = sorted(
            set(nom.columns) & set(mpc_df.columns)
            & set(rpc_df.columns) & set(cs.columns)
        )
        datasets = sorted(
            set(nom.index) & set(mpc_df.index)
            & set(rpc_df.index) & set(cs.index)
        )

        w_model = max(20, max((len(m) for m in models), default=20))
        hdr = (f" {'Model':<{w_model}s}"
               f" {'Nom. RMSE':>10s}"
               f" {'mPC':>10s}"
               f" {'rPC':>8s}"
               f" {'s_cross':>8s}")
        sep = (f" {'─' * w_model}"
               f" {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 8}")

        for ds in datasets:
            print(f"\n{'─' * (w_model + 44)}")
            print(f" Dataset: {ds}")
            print(f"{'─' * (w_model + 44)}")
            print(hdr)
            print(sep)
            for m in models:
                n = nom.loc[ds, m] if ds in nom.index and m in nom.columns else np.nan
                mp = mpc_df.loc[ds, m] if ds in mpc_df.index and m in mpc_df.columns else np.nan
                r = rpc_df.loc[ds, m] if ds in rpc_df.index and m in rpc_df.columns else np.nan
                s = cs.loc[ds, m] if ds in cs.index and m in cs.columns else np.nan
                s_str = f"{s:>8.3f}" if not pd.isna(s) else f"{'--':>8s}"
                print(f" {m:<{w_model}s} {n:>10.4f} {mp:>10.4f} {r:>8.3f} {s_str}")
        print()

    # ── Export ────────────────────────────────────────────────────────

    def to_csv(self, output_dir):
        """Save the raw per-severity DataFrame to CSV.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write into (created if needed).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.raw.to_csv(output_dir / "per_severity.csv", index=False)

        n = len(self.raw)
        print(f"Results saved to {output_dir}/")
        print(f"  per_severity.csv  ({n} rows)")

    # ── Plotting ──────────────────────────────────────────────────────

    def plot_degradation(self, dataset=None, model=None, failure=None,
                         feature_idx=None, ax=None, show=True):
        """Plot RMSE vs severity curves with optional filters.

        Parameters
        ----------
        dataset, model, failure, feature_idx : str | int | None
            Filters to narrow down the curves.  ``None`` means "all".
        ax : matplotlib Axes, optional
        show : bool
            Call ``plt.show()`` at the end (default ``True``).
        """
        df = self.raw.copy()
        if dataset is not None:
            df = df[df["dataset"] == dataset]
        if model is not None:
            df = df[df["model"] == model]
        if failure is not None:
            df = df[df["failure_mode"] == failure]
        if feature_idx is not None:
            df = df[df["feature_idx"] == feature_idx]

        if df.empty:
            print("No matching data for the given filters.")
            return

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        grouped = df.groupby(
            ["model", "failure_mode", "severity"], sort=False
        )["boot_mean"].mean().reset_index()

        for (mdl, fail), sub in grouped.groupby(["model", "failure_mode"]):
            sub = sub.sort_values("severity")
            ax.plot(sub["severity"], sub["boot_mean"], label=f"{mdl} / {fail}")

        ax.set_xlabel("Severity s")
        ax.set_ylabel("RMSE")
        ax.set_title("Degradation Curves")
        ax.legend(fontsize="small", loc="best")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        if show:
            plt.show()
        return ax
