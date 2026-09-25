"""PyTorch `Dataset` wrappers for sequential and flattened time-series inputs.

Reused from the MuViS benchmark codebase.

Sources
-------
- https://arxiv.org/abs/2603.24602
- https://github.com/noah-puetz/MuViS
"""

import torch
from torch.utils.data import Dataset

class SequentialDataset(Dataset):
	def __init__(self, X, y):
		self.feat_dim = X.shape[2]
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]
	
class FlattenedDataset(Dataset):
	def __init__(self, X, y):
		self.feat_dim = X.shape[2] * X.shape[1]
		num_samples = X.shape[0]
		self.X = torch.tensor(X.reshape(num_samples, -1), dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]