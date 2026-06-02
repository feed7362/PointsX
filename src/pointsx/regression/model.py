"""MLP regression model for circumference prediction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CircumferenceRegressor(nn.Module):
    """Small MLP: 28 input features → 6 circumference predictions.

    Outputs: [neck, waist, hip, thigh, calf, wrist] circumferences in cm.
    """

    NUM_FEATURES = 28
    NUM_OUTPUTS = 6
    OUTPUT_NAMES = ["neck", "waist", "hip", "thigh", "calf", "wrist"]

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, self.NUM_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a single feature vector.

        Args:
            features: (28,) numpy array of input features.

        Returns:
            (6,) numpy array of predicted circumferences in cm.
        """
        self.eval()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        out = self.forward(x)
        return out.squeeze(0).numpy()
