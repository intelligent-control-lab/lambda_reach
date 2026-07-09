import os
import pickle
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


class HJValueDataset(Dataset):
    """Loads unified dataset saved by data_processing.process_hj_dataset.

    Each item is a tuple:
      (x, x_next, l, sample_safety_value, invariant_flag, x_future_traj, l_future_traj)
    where future trajectories are variable-length numpy arrays starting at t+1.
    """

    def __init__(self, pkl_path: str, transform=None):
        assert os.path.exists(pkl_path), f"File not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Base arrays (fixed shapes)
        self.x = np.stack([d[0] for d in data], axis=0).astype(np.float32)
        self.x_next = np.stack([d[1] for d in data], axis=0).astype(np.float32)
        self.l = np.stack([d[2] for d in data], axis=0).astype(np.float32)
        self.sample_safety_value = np.stack([d[3] for d in data], axis=0).astype(np.float32)
        self.invariant = np.stack([d[4] for d in data], axis=0).astype(np.float32)

        # Variable-length futures stay as lists of numpy arrays
        self.x_future = [d[5] for d in data]
        self.l_future = [d[6] for d in data]

        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])
        x_next = torch.from_numpy(self.x_next[idx])
        l = torch.from_numpy(np.array(self.l[idx], dtype=np.float32)).reshape(-1)
        sample_safety_value = torch.from_numpy(np.array(self.sample_safety_value[idx], dtype=np.float32)).reshape(-1)
        invariant = torch.from_numpy(np.array(self.invariant[idx], dtype=np.float32)).reshape(-1)

        if self.transform is not None:
            x = self.transform(x)
            x_next = self.transform(x_next)

        # Each trajectory is already an array with shape [L, dim]; wrap once to keep it intact
        x_future = torch.as_tensor(self.x_future[idx], dtype=torch.float32)
        l_future = torch.as_tensor(self.l_future[idx], dtype=torch.float32)

        return x, x_next, l, sample_safety_value, invariant, x_future, l_future


def hj_collate(batch):
    """Collate unified samples, keeping future trajectories as lists."""
    (
        x_list,
        x_next_list,
        l_list,
        sample_safety_value_list,
        invariant_list,
        x_future_list,
        l_future_list,
    ) = zip(*batch)

    x = torch.stack(x_list, dim=0)
    x_next = torch.stack(x_next_list, dim=0)
    l = torch.stack(l_list, dim=0)
    sample_safety_value = torch.stack(sample_safety_value_list, dim=0)
    invariant = torch.stack(invariant_list, dim=0)

    # Future trajectories remain per-sample lists (no padding)
    x_future = list(x_future_list)
    l_future = list(l_future_list)

    return x, x_next, l, sample_safety_value, invariant, x_future, l_future
