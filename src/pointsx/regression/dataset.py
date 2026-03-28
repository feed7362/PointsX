"""Dataset class for circumference regression training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CircumferenceDataset(Dataset):
    """Dataset of (feature_vector, ground_truth_circumferences) pairs.

    Expected data format: a .npz file with:
        - "features": (N, 28) float32 array
        - "targets": (N, 6) float32 array [neck, waist, hip, thigh, calf, wrist]
    """

    def __init__(self, data_path: str | Path):
        data = np.load(str(data_path))
        self.features = torch.tensor(data["features"], dtype=torch.float32)
        self.targets = torch.tensor(data["targets"], dtype=torch.float32)

        assert self.features.shape[1] == 28, f"Expected 28 features, got {self.features.shape[1]}"
        assert self.targets.shape[1] == 6, f"Expected 6 targets, got {self.targets.shape[1]}"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
