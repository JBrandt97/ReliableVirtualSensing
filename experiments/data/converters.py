"""Dataset converters for raw → processed `.ts` files.

Reused from the MuViS benchmark codebase.

Sources
-------
- https://arxiv.org/abs/2603.24602
- https://github.com/noah-puetz/MuViS
"""

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import pickle
import logging
from typing import List, Tuple, Dict, Any, Union

import pyreadr
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

from muvis_c._ts import write_dataframe_to_tsfile, load_from_tsfile_to_dataframe

class BaseConverter(ABC):
    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_raw(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Logic to load specific file formats (.mat, .ts, .csv).
        Must return:
           1. A tuple of lists: (test_cases, train_cases) or just a list of all cases
           2. Metadata dictionary
        """
        pass

    def run(self):
        """The common workflow"""
        logging.debug(f"Starting conversion for {self.__class__.__name__}...")
        
        # Expecting ((test_cases, train_cases), meta) OR (all_cases, meta)
        data, meta = self.load_raw()
        
        if not data["test"] or not data["train"]:
            raise RuntimeError("No valid sequences generated.")
    
        # Already split
        X_train, y_train = self.build_panel_dataframe(data["train"])
        X_test, y_test = self.build_panel_dataframe(data["test"])
        
        logging.debug(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        logging.debug("Writing train.ts ...")
        self.safe_write_tsfile(X_train, self.output_dir, "train", y_train)

        logging.debug("Writing test.ts ...")
        self.safe_write_tsfile(X_test, self.output_dir, "test", y_test)

        logging.debug(f"Conversion complete. Files saved to: {self.output_dir}")
        
        self.prepend_ts_metadata("train.ts", meta)
        self.prepend_ts_metadata("test.ts", meta)
        
    def prepend_ts_metadata(self, filename: str, meta: Dict[str, Any]):
        ts_path = self.output_dir / filename
        
        header = [
            f"#Target: {meta.get('target_name', 'target')}\n",
            f"#FeatureColumns: {', '.join(meta.get('feature_names', []))}\n",
            f"#FeatureDimensions: {meta.get('n_features', 0)}\n",
            "\n"
        ]
    
        # Read all, write header + original
        content = ts_path.read_text(encoding="utf-8")
        with ts_path.open("w", encoding="utf-8") as f:
            f.writelines(header)
            f.write(content)
        
    def safe_write_tsfile(self, data: pd.DataFrame, path: Path, problem_name: str, labels: Union[pd.Series, np.ndarray, list]):
        """
        Safely writes .ts file ensuring consistent format and label handling.
        """
        tmp_dir = Path.cwd() / f"_{problem_name}_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        labels = labels.astype(str)

        write_dataframe_to_tsfile(
            data=data,
            path=str(tmp_dir), # util expects string path usually
            problem_name=problem_name,
            equal_length=True,
            class_label=labels,
            class_value_list=labels
        )

        src = tmp_dir / problem_name / f"{problem_name}.ts"
        dest = path / f"{problem_name}.ts"
        shutil.move(str(src), str(dest))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def split_into_sequences(self, df: pd.DataFrame, seq_len: int, target_column: str, step: int = 1) -> List[Tuple[pd.DataFrame, Any]]:
        n = len(df)
        if step < 1:
            raise ValueError("Step must be at least 1.")
        
        sequences = []
        for start in range(0, n - seq_len + 1, step):
            end = start + seq_len
            
            # Fast slice
            target_value = df[target_column].iloc[end - 1] 
            
            feature_segment = df.iloc[start:end].drop(columns=[target_column]).copy()
            
            sequences.append((feature_segment, target_value))

        return sequences

    def build_panel_dataframe(self, cases: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(cases)
        if "target_val" in df.columns:
            y = df.pop("target_val").astype(str)
        else:
            raise ValueError("Cases dictionaries must contain 'target_val' key.")
        return df, y


class REVSConverter(BaseConverter):
    def __init__(self, 
                 raw_dir: str, 
                 output_dir: str, 
                 target_column: str = "vyCG",
                 sequence_length: int = 20,
                 sequence_step: int = 1,
                 resampling_interval: str = "50ms"):
        super().__init__(raw_dir, output_dir)
        
        self.feature_columns = [
            "handwheelAngle", "throttle", "brake", "vxCG", "axCG", "ayCG",
            "yawRate", "rollRate", "pitchRate", "rollAngle", "pitchAngle",
            "chassisAccelFL", "chassisAccelFR", "chassisAccelRL", "chassisAccelRR",
            "wheelAccelFL", "wheelAccelFR", "wheelAccelRL", "wheelAccelRR",
            "deflectionFL", "deflectionFR", "deflectionRR",
        ]
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        self.resampling_interval = resampling_interval
        
        self.csvs_to_exclude = {
            "20140221_02_01_03_250lm.csv",
            "20140221_04_01_03_250lm.csv",
            "20140221_03_01_03_250lm.csv",
            "20130817_01_01_02_grandsport.csv"
        }
        self.test_files = {
            "20130816_01_01_02_grandsport.csv",
            "20130222_01_02_03_grandsport.csv",
            "20140221_01_02_03_250lm.csv"
        }
        
    def load_raw(self) -> Tuple[Tuple[List, List], Dict]:
        csv_paths = sorted(self.raw_dir.glob("**/*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")

        all_cases = {"test": [], "train": []}

        for path in tqdm(csv_paths, desc=f"Processing REVS {self.raw_dir.name}"):
            if path.name in self.csvs_to_exclude:
                continue

            sequences = self._process_single_csv(path)
            if not sequences:
                continue

            split = "test" if path.name in self.test_files else "train"
            all_cases[split].extend(sequences)

        logging.debug(f"Loaded {len(all_cases['test'])} test sequences, {len(all_cases['train'])} train sequences.")
        
        meta = {
            "feature_names": self.feature_columns,
            "target_name": self.target_column,
            "n_features": len(self.feature_columns),
        }
        return all_cases, meta

    def _process_single_csv(self, path: Path) -> List[Dict]:
        """Process a single CSV file and return a list of sequence dicts."""
        # Detect year
        year = None
        if "2013" in path.name: year = 2013
        elif "2014" in path.name: year = 2014

        # Read CSV
        df = pd.read_csv(path, skiprows=10, header=[0, 1], encoding="unicode_escape", sep=",")
        df.columns = df.columns.droplevel(-1)

        # Filter cols
        available_cols = [c for c in self.feature_columns + [self.target_column, "time"] if c in df.columns]
        df = df[available_cols]

        # Resample
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").resample(self.resampling_interval).first()
        df = df.reset_index(drop=True)

        # 2013 Sign Correction
        if year == 2013:
            for col in ["chassisAccelRL", "chassisAccelRR"]:
                if col in df.columns:
                    df[col] *= -1
                    
        # Filter sequences by speed condition
        mask = df["vxCG"] >= 3
        # Group continuous segments where mask is True
        segment_ids = (mask != mask.shift()).cumsum()
        
        sequence_dicts = []
        for _, segment in df.groupby(segment_ids):
            # Check if this group is valid (mask is true)
            if not mask.loc[segment.index[0]]:
                continue
                
            if len(segment) < self.sequence_length:
                continue

            seqs = self.split_into_sequences(
                segment, 
                seq_len=self.sequence_length, 
                target_column=self.target_column, 
                step=self.sequence_step
            )

            for feature_df, target_val in seqs:
                case_data = {col: pd.Series(feature_df[col].values) for col in feature_df.columns}
                case_data["target_val"] = target_val
                sequence_dicts.append(case_data)
                
        return sequence_dicts

class VehicleDynamicsConverter(BaseConverter):
    def __init__(self, 
                    raw_dir: str, 
                    output_dir: str, 
                    target_column: str = "tireTemp_fr_degC",
                    sequence_length: int = 50, 
                    sequence_step: int = 1):
            super().__init__(raw_dir, output_dir)
            
            self.feature_columns = ['throttleCmd_percent', 
                                    'brake_fr_bar', 
                                    'axCG_mps2', 
                                    'ayCG_mps2', 
                                    'grade_rad', 
                                    'roadWheelAngle_rad',
                                    'yawRate_radps',
                                    'Vx_mps',
                                    'Vy_mps',
                                    'slipAngle_rad',
                                    'bank_rad',]
            self.target_column = target_column
            self.sequence_length = sequence_length     
            self.sequence_step = sequence_step
            self.resampling_interval = "1s"  # 1 Hz  

            self.test_files = {"VehicleDynamicsDataset_Nov2023_2023-11_11.csv",
                            "VehicleDynamicsDataset_Nov2023_2023-11_7.csv",
                            "VehicleDynamicsDataset_Oct2023_2023-10_1.csv",
                            "VehicleDynamicsDataset_Oct2023_2023-10_3.csv"}
        
    def load_raw(self) -> Tuple[List[Dict], Dict]:
        csv_paths = sorted(self.raw_dir.glob("**/*.csv"))
        all_cases = {"test": [], "train": []}

        for path in tqdm(csv_paths, desc=f"Processing Vehicle {self.raw_dir.name}"):
            sequences = self._process_single_csv(path)
            split = "test" if path.name in self.test_files else "train"
            all_cases[split].extend(sequences)
            

        logging.debug(f"Loaded {len(all_cases['test'])} test sequences, {len(all_cases['train'])} train sequences.")
        meta = {
            "feature_names": self.feature_columns,
            "target_name": self.target_column,
            "n_features": len(self.feature_columns)
        }
        return all_cases, meta

    def _process_single_csv(self, path: Path) -> List[Dict]:
        df = pd.read_csv(path, skiprows=2)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")

        # Keep valid columns
        valid_cols = [c for c in self.feature_columns + [self.target_column, "t_s"] if c in df.columns]
        df = df[valid_cols]

        df["t_s"] = pd.to_datetime(df["t_s"], unit="s", errors="coerce")
        df = df.set_index("t_s").resample(self.resampling_interval).mean()
        df = df.interpolate(limit_direction="both").reset_index(drop=True)

        if len(df) < self.sequence_length:
            return []

        sequences = self.split_into_sequences(df, self.sequence_length, self.target_column, step=self.sequence_step)
        
        sequence_dicts = []
        for feature_df, target_val in sequences:
            # Convert to sktime-compatible dict (column -> pd.Series)
            case_data = {col: pd.Series(feature_df[col].values) for col in feature_df.columns}
            case_data["target_val"] = target_val
            sequence_dicts.append(case_data)
            
        return sequence_dicts
    
class TennesseeEastmanProcessConverter(BaseConverter):
    X_DICT = {
        'XMEAS_1':'A_feed_stream',
        'XMEAS_2':'D_feed_stream',
        'XMEAS_3':'E_feed_stream',
        'XMEAS_4':'Total_fresh_feed_stripper',
        'XMEAS_5':'Recycle_flow_into_rxtr',
        'XMEAS_6':'Reactor_feed_rate',
        'XMEAS_7':'Reactor_pressure',
        'XMEAS_8':'Reactor_level',
        'XMEAS_9':'Reactor_temp',
        'XMEAS_10':'Purge_rate',
        'XMEAS_11':'Separator_temp',
        'XMEAS_12':'Separator_level',
        'XMEAS_13':'Separator_pressure',
        'XMEAS_14':'Separator_underflow',
        'XMEAS_15':'Stripper_level',
        'XMEAS_16':'Stripper_pressure',
        'XMEAS_17':'Stripper_underflow',
        'XMEAS_18':'Stripper_temperature',
        'XMEAS_19':'Stripper_steam_flow',
        'XMEAS_20':'Compressor_work',
        'XMEAS_21':'Reactor_cooling_water_outlet_temp',
        'XMEAS_22':'Condenser_cooling_water_outlet_temp',
        'XMV_1':'D_feed_flow_valve',
        'XMV_2':'E_feed_flow_valve',
        'XMV_3':'A_feed_flow_valve',
        'XMV_4':'Total_feed_flow_stripper_valve',
        'XMV_5':'Compressor_recycle_valve',
        'XMV_6':'Purge_valve',
        'XMV_7':'Separator_pot_liquid_flow_valve',
        'XMV_8':'Stripper_liquid_product_flow_valve',
        'XMV_9':'Stripper_steam_valve',
        'XMV_10':'Reactor_cooling_water_flow_valve',
        'XMV_11':'Condenser_cooling_water_flow_valve',
        'XMEAS_35':'Composition_of_G_purge'
    }
    def __init__(self, 
                 raw_dir: str, 
                 output_dir: str, 
                 target_column: str = "xmeas_35",
                 sequence_length: int = 20,
                 sequence_step: int = 1):
        super().__init__(raw_dir, output_dir)
        self.feature_columns = [
             'xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
             'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16',
             'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22',
             'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11'
        ]
        self.target_column = target_column
        self.id_columns = ["faultNumber", "simulationRun", "sample"]
        self.group_column = "simulationRun"
        self.train_file = "TEP_FaultFree_Training.RData"
        self.test_file  = "TEP_FaultFree_Testing.RData"
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        
    def load_raw(self) -> Tuple[Tuple[List, List], Dict]:
        train_path = self.raw_dir / self.train_file
        test_path  = self.raw_dir / self.test_file

        logging.debug("Loading Tennessee Eastman RData files...")
        train_cases = self._process_rdata(train_path, "Train")
        test_cases = self._process_rdata(test_path, "Test")

        meta = {
            "feature_names": [self.X_DICT.get(c.upper(), c) for c in self.feature_columns],
            "target_name": self.X_DICT.get(self.target_column.upper(), self.target_column),
            "n_features": len(self.feature_columns)
        }
        return {"train": train_cases, "test": test_cases}, meta

    def _process_rdata(self, path: Path, desc: str) -> List[Dict]:
        r_data = pyreadr.read_r(str(path))
        df = r_data[next(iter(r_data.keys()))]

        cols_to_keep = self.id_columns + self.feature_columns + [self.target_column]
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols]

        cases = []
        for _, group in tqdm(df.groupby(self.group_column), desc=f"Processing {desc} runs"):
            group = group.sort_values("sample")
            
            # Select only features + target for sequence generation
            seq_cols = [c for c in self.feature_columns + [self.target_column] if c in group.columns]
            group_for_seq = group[seq_cols]

            sequences = self.split_into_sequences(
                group_for_seq, 
                seq_len=self.sequence_length, 
                target_column=self.target_column, 
                step=self.sequence_step
            )
            
            for feature_df, target_val in sequences:
                case_data = {col: pd.Series(feature_df[col].values) for col in feature_df.columns}
                case_data["target_val"] = target_val
                cases.append(case_data)
        
        return cases
class Panasonic18650PFConverter(BaseConverter): 
    def __init__(self, 
                 raw_dir: str, 
                 output_dir: str, 
                 target_column: str = "SOC",
                 sequence_length: int = 120,
                 sequence_step: int = 1):
        super().__init__(raw_dir, output_dir)
        self.feature_columns = ["V", "I", "T", "V_0.5mHz", "I_0.5mHz", "V_5mHz", "I_5mHz"]
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        
    def load_raw(self) -> Tuple[Tuple[List, List], Dict]:
        mat_paths = sorted(self.raw_dir.glob("**/*.mat"))
        all_cases = {"train": [], "test": []}

        for path in tqdm(mat_paths, desc="Processing Panasonic MAT"):
            # Folder-based Split Logic
            folder_name = path.parent.name
            if folder_name == "Normalization": continue
            
            split = "test" if folder_name == "Test" else "train"
            
            # Load & Process
            try:
                sequences = self._process_single_mat(path)
                all_cases[split].extend(sequences)
            except Exception as e:
                logging.error(f"Failed to process {path.name}: {e}")
                raise e

        meta = {
            "feature_names": self.feature_columns,
            "target_name": self.target_column,
            "n_features": len(self.feature_columns)
        }
        return all_cases, meta

    def _process_single_mat(self, path: Path) -> List[Dict]:
        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)

        # Heuristic to find X and Y
        keys = list(data.keys())
        X = next((data[k] for k in keys if k.lower().startswith("x")), None)
        Y = next((data[k] for k in keys if k.lower().startswith("y") or k == "SOC"), None)

        datasets = self._unpack_mat_data(X, Y)
        sequences = []

        for name, ds in datasets.items():
            data_X, data_Y = ds["X"].T, ds["Y"]    
            df = pd.DataFrame(data_X)
            target_vals = np.array(data_Y).flatten()
            
            df[self.target_column] = target_vals

            # Create Sequences
            seqs = self.split_into_sequences(df, self.sequence_length, self.target_column, self.sequence_step)
            for feature_df, target_val in seqs:
                case_data = {col: pd.Series(feature_df[col].values) for col in feature_df.columns}
                case_data["target_val"] = target_val
                sequences.append(case_data)
                
        return sequences
    
    def _unpack_mat_data(self, X, Y) -> Dict[str, Dict]:
        """Handles aggregated Test.mat structure."""
        datasets = {}
        # Check if X is an object array containing multiple datasets
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == 2 and X.dtype == object:
            names = X[0, :]
            x_values = X[1, :]
            
            if isinstance(Y, np.ndarray) and Y.shape == X.shape:
                y_values = Y[1, :]
            else:
                y_values = [Y] * len(names)

            for name, x_arr, y_arr in zip(names, x_values, y_values):
                datasets[name] = {"X": x_arr, "Y": y_arr}
        else:
            datasets["Default"] = {"X": X, "Y": Y}
        return datasets
 
class BeijingPMQualityConverter(BaseConverter):
    """
    Cleaner/Converter for Beijing PM datasets. 
    Note: This class overrides the typical load_raw flow because it cleans existing .ts files
    rather than converting raw CSV/Mat/etc files.
    """
    def load_raw(self):
        return [], {}

    def run(self):
        """Override run to execute custom cleaning pipeline."""
        self.clean_data()
        
    def clean_data(self):
        train_file = self.raw_dir / f"{self.raw_dir.name}_TRAIN.ts"
        test_file  = self.raw_dir / f"{self.raw_dir.name}_TEST.ts"
        
        for file in [train_file, test_file]:
            if not file.exists():
                logging.warning(f"File not found: {file}")
                continue

            problem_name = "train" if "TRAIN" in file.name else "test"
            logging.debug(f"Cleaning {file.name} ...")

            X, y = load_from_tsfile_to_dataframe(str(file))
            
            # Count NaNs BEFORE cleaning
            total_nans_before = 0
            total_points_before = 0
            logging.debug("NaN counts BEFORE cleaning:")
            for col in X.columns:
                col_nans = X[col].apply(lambda s: s.isna().sum() if isinstance(s, pd.Series) else 0).sum()
                col_total = X[col].apply(lambda s: len(s) if isinstance(s, pd.Series) else 0).sum()
                total_nans_before += col_nans
                total_points_before += col_total
                logging.debug(f"  - {col:<25}: {col_nans} NaNs out of {col_total} points")

            # Interpolate and fill NaNs safely (feature-wise)
            logging.debug("Interpolating missing values in feature dimensions...")
            for col in X.columns:
                for i in range(len(X)):
                    val = X.iat[i, X.columns.get_loc(col)]
                    if isinstance(val, pd.Series):
                        X.iat[i, X.columns.get_loc(col)] = val.interpolate(method="linear").bfill().ffill()

            # Count NaNs AFTER cleaning
            total_nans_after = 0
            total_points_after = 0
            logging.debug("NaN counts AFTER cleaning:")
            for col in X.columns:
                col_nans = X[col].apply(lambda s: s.isna().sum() if isinstance(s, pd.Series) else 0).sum()
                col_total = X[col].apply(lambda s: len(s) if isinstance(s, pd.Series) else 0).sum()
                total_nans_after += col_nans
                total_points_after += col_total
                logging.debug(f"  - {col:<25}: {col_nans} NaNs out of {col_total} points")

            # Write output
            self.safe_write_tsfile(X, self.output_dir, problem_name, y)
            meta = {
                "feature_names": list(X.columns),
                "target_name": self.raw_dir.name.split("Beijing")[1].split("Quality")[0],
                "n_features": len(X.columns),
            }
            self.prepend_ts_metadata(f"{problem_name}.ts", meta)

            logging.debug(f"Cleaned file saved to {self.output_dir}")

class PPGDaliaConverter(BaseConverter):
    def __init__(self, 
                 raw_dir: str, 
                 output_dir: str, 
                 sequence_length: int = 512, 
                 sequence_step: int = 128):
        super().__init__(raw_dir, output_dir)
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        self.feature_names = ["wrist_BVP", "wrist_ACC_x", "wrist_ACC_y", "wrist_ACC_z", "wrist_EDA", "wrist_TEMP"]
        self.target_name = "heart_rate_bpm"
        self.test_subjects = [13, 4, 2]
        
        self._activities = None
        

    def load_raw(self) -> Tuple[Dict[str, List[Dict]], Dict]:
        all_cases = {"train": [], "test": []}
        activities = []

        # S1 to S15
        for subject_id in tqdm(range(1, 16), desc="Processing PPG-DaLiA"):
            path = self.raw_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
            if not path.exists():
                continue
                
            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            
            # Unwrap dict if nested
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            X, y, act = self._make_windows(data)

            # Convert to sktime format
            for k in range(len(y)):
                case = {
                    "wrist_BVP":   pd.Series(X[k, :, 0]),
                    "wrist_ACC_x": pd.Series(X[k, :, 1]),
                    "wrist_ACC_y": pd.Series(X[k, :, 2]),
                    "wrist_ACC_z": pd.Series(X[k, :, 3]),
                    "wrist_EDA":   pd.Series(X[k, :, 4]),
                    "wrist_TEMP":  pd.Series(X[k, :, 5]),
                    "target_val":  y[k],
                }
                split = "test" if subject_id in self.test_subjects else "train"
                all_cases[split].append(case)
                activities.append(int(act[k]))

        self._activities = np.asarray(activities, dtype=np.int16)
        
        meta = {
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "n_features": len(self.feature_names),
        }
        return all_cases, meta

    def _make_windows(self, data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Unpack signals
        wrist = data["signal"]["wrist"]
        bvp = np.asarray(wrist["BVP"]).squeeze()
        acc = np.asarray(wrist["ACC"])
        eda = np.asarray(wrist["EDA"]).squeeze()
        temp = np.asarray(wrist["TEMP"]).squeeze()
        y = np.asarray(data["label"]).squeeze()
        activity = np.asarray(data["activity"]).squeeze()

        # Resample logic
        acc_64 = resample_poly(acc, up=2, down=1, axis=0) # 32->64
        eda_64 = resample_poly(eda.reshape(-1, 1), up=16, down=1, axis=0) # 4->64
        
        # Repeat temp to avoid ringing (step function)
        temp_64 = np.repeat(temp, 16).reshape(-1, 1) # 4->64
        temp_64 = temp_64[:len(bvp)]
        if len(temp_64) < len(bvp):
            # Pad if shorter (match original logic)
            padding = len(bvp) - len(temp_64)
            temp_64 = np.pad(temp_64, ((0, padding), (0, 0)), mode="edge")
        
        # Generator for windows
        win_size, hop = self.sequence_length, self.sequence_step
        max_len = min(len(bvp), len(acc_64), len(eda_64), len(temp_64))

        X_list, y_list, act_list = [], [], []

        # Determine activity mapping type
        direct_map = (len(activity) == len(y))

        for i in range(len(y)):
            start, end = i * hop, i * hop + win_size
            if end > max_len: break

            window = np.concatenate([
                bvp[start:end].reshape(-1,1),
                acc_64[start:end],
                eda_64[start:end],
                temp_64[start:end]
            ], axis=1)

            X_list.append(window.astype(np.float32))
            y_list.append(float(y[i]))
            
            # Map activity
            if direct_map:
                act = activity[i]
            else:
                # repeat 4->64 logic roughly or take end of window
                idx = min(len(activity)-1, end // 16) 
                act = activity[idx]
            act_list.append(act)

        return np.stack(X_list), np.asarray(y_list), np.asarray(act_list)


class PPGDaliaConverterRandomSplit(PPGDaliaConverter):
    """Like PPGDaliaConverter but uses a random 80/20 train/test split
    instead of a fixed subject-based split."""

    def __init__(self,
                 raw_dir: str,
                 output_dir: str,
                 sequence_length: int = 512,
                 sequence_step: int = 128,
                 test_ratio: float = 0.2,
                 random_seed: int = 42):
        super().__init__(raw_dir, output_dir, sequence_length, sequence_step)
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def load_raw(self) -> Tuple[Dict[str, List[Dict]], Dict]:
        all_cases = []
        activities = []

        for subject_id in tqdm(range(1, 16), desc="Processing PPG-DaLiA"):
            path = self.raw_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
            if not path.exists():
                continue

            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            X, y, act = self._make_windows(data)

            for k in range(len(y)):
                case = {
                    "wrist_BVP":   pd.Series(X[k, :, 0]),
                    "wrist_ACC_x": pd.Series(X[k, :, 1]),
                    "wrist_ACC_y": pd.Series(X[k, :, 2]),
                    "wrist_ACC_z": pd.Series(X[k, :, 3]),
                    "wrist_EDA":   pd.Series(X[k, :, 4]),
                    "wrist_TEMP":  pd.Series(X[k, :, 5]),
                    "target_val":  y[k],
                }
                all_cases.append(case)
                activities.append(int(act[k]))

        # Random 80/20 split on windows
        rng = np.random.default_rng(self.random_seed)
        n = len(all_cases)
        indices = rng.permutation(n)
        n_test = int(np.ceil(n * self.test_ratio))
        test_idx = set(indices[:n_test])

        split_cases: Dict[str, List[Dict]] = {"train": [], "test": []}
        split_activities = []
        for i, case in enumerate(all_cases):
            split = "test" if i in test_idx else "train"
            split_cases[split].append(case)
            split_activities.append(activities[i])

        self._activities = np.asarray(split_activities, dtype=np.int16)

        meta = {
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "n_features": len(self.feature_names),
        }
        return split_cases, meta