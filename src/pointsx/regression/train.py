"""Training script for the circumference regression model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pointsx.regression.dataset import CircumferenceDataset
from pointsx.regression.model import CircumferenceRegressor

logger = logging.getLogger(__name__)


def train(
    data_path: str | Path,
    output_path: str | Path = "models/circumference_regressor.pt",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    """Train the circumference regression model."""
    torch.manual_seed(seed)

    dataset = CircumferenceDataset(data_path)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CircumferenceRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                preds = model(features)
                loss = criterion(preds, targets)
                val_loss += loss.item() * features.size(0)
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 20 == 0 or epoch == 1:
            rmse = val_loss ** 0.5
            logger.info(
                "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, val_RMSE: %.2f cm",
                epoch, epochs, train_loss, val_loss, rmse,
            )

    # Save best model
    if best_state is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, str(output_path))
        logger.info(
            "Saved best model to %s (val_RMSE: %.2f cm)",
            output_path, best_val_loss ** 0.5,
        )

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train circumference regression model")
    parser.add_argument("--data", required=True, help="Path to training data .npz file")
    parser.add_argument("--output", default="models/circumference_regressor.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train(args.data, args.output, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
