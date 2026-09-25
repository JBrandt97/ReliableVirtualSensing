"""`MuViSDataset` loader for `.ts` files under a dataset directory.

Reused from the MuViS benchmark codebase.

Sources
-------
- https://arxiv.org/abs/2603.24602
- https://github.com/noah-puetz/MuViS
"""

import os
import numpy as np
from muvis_c._ts import load_from_tsfile

class MuViSDataset:
    def __init__(self, dataset_id, base_path="data/processed"):
        self.dataset_id = dataset_id
        self.base_path = base_path
        self.path = os.path.join(base_path, dataset_id)
        
        # Check if path exists, otherwise try with _ts suffix (common pattern in this workspace)
        if not os.path.exists(self.path):
            raise ValueError(f"path {self.path} does not exist")
        
        # Meta information (populated after loading data)
        self.meta = {"sequence_length": None, "n_channels": None, "target_name": None, "feature_names": None}
        
    def get_data(self, split='all'):
        """
        Load data from train.ts and test.ts.
        
        Parameters
        ----------
        flattened : bool
            If True, flattens the time dimension (N, T*C). 
            If False, returns (N, T, C).
        split : str
            'train', 'test', or 'all' (default).
            
        Returns
        -------
        X, y : np.ndarray
        """
        X_train, y_train = None, None
        X_test, y_test = None, None
        
        if split in ['train', 'all']:
            train_file = os.path.join(self.path, "train.ts")
            if os.path.exists(train_file):
                X_train, y_train = load_from_tsfile(train_file, return_data_type="numpy3d", return_y=True, y_dtype="float")
                # X is (N, C, T) from loader, transpose to (N, T, C)
                X_train = np.transpose(X_train, (0, 2, 1))
            else:
                pass 

        if split in ['test', 'all']:
            test_file = os.path.join(self.path, "test.ts")
            if os.path.exists(test_file):
                X_test, y_test = load_from_tsfile(test_file, return_data_type="numpy3d", return_y=True, y_dtype="float")
                X_test = np.transpose(X_test, (0, 2, 1))
            else:
                pass

        # Combine or return specific
        if split == 'train':
            if X_train is None: raise FileNotFoundError(f"train.ts not found in {self.path}")
            X, y = X_train, y_train
        elif split == 'test':
            if X_test is None: raise FileNotFoundError(f"test.ts not found in {self.path}")
            X, y = X_test, y_test
        elif split == 'all':
            if X_train is not None and X_test is not None:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
            elif X_train is not None:
                X, y = X_train, y_train
            elif X_test is not None:
                X, y = X_test, y_test
            else:
                raise FileNotFoundError(f"No .ts data found in {self.path}")
        else:
            raise ValueError("split must be 'train', 'test', or 'all'")

        # Update meta info based on loaded data
        # shape is (N, T, C) before flattening
        self.meta["sequence_length"] = X.shape[1]
        self.meta["n_channels"] = X.shape[2]
            
        return X, y
    
    def get_meta_info(self):
        # load meta.yaml if exists
        with open(os.path.join(self.path, "train.ts"), "r", encoding="utf-8") as f:
            original = f.read()
            meta_string = original[:original.find("@")]
            target, features, n_features = meta_string.splitlines()[:-1]
            self.meta["target_name"] = target.split(":")[-1].strip()  
            self.meta["feature_names"] = [feat.strip() for feat in features.split(":")[-1].strip().split(",")]

        return self.meta

    @staticmethod
    def get_dataset(dataset_id, base_path="data/processed"):
        return MuViSDataset(dataset_id, base_path)