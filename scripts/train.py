"""
Training Script — Signature Verification System

Entry point for training the Siamese networks.

Usage:
    python scripts/train.py --branch spatial --data-dir ./data/cedar
    python scripts/train.py --branch temporal --data-dir ./data/svc2004
    python scripts/train.py --branch spatial --resume models/weights/checkpoint_spatial.pth
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    SPATIAL_EMBEDDING_DIM, TEMPORAL_EMBEDDING_DIM,
    CONTRASTIVE_MARGIN, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, WEIGHT_DECAY, MODEL_WEIGHTS_DIR,
    TEMPORAL_HIDDEN_SIZE, TEMPORAL_NUM_LAYERS,
    PRESSURE_DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
)
from src.models.siamese import SpatialSiamese, TemporalSiamese, ContrastiveLoss


class EarlyStopping:
    """
    Early Stopping monitor to halt training if validation loss stops improving.
    Prevents model overfitting by saving the best performing weight checkpoint.
    """

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Siamese signature verification models")
    parser.add_argument("--branch", choices=["spatial", "temporal"], required=True,
                        help="Which branch to train")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Path(s) to dataset directories")
    parser.add_argument("--val-dir", nargs="+", default=None,
                        help="Optional path(s) to validation holdout dataset directories")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """
    Train model for a single epoch.
    Returns average training loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x1, x2, labels) in enumerate(dataloader):
        # Transfer batch tensors to active computing device (GPU or CPU)
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()
        emb1, emb2 = model(x1, x2)
        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Train Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def validate_one_epoch(model, dataloader, loss_fn, device):
    """
    Evaluate validation loss on holdout set without computing gradients.
    Returns average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x1, x2, labels in dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            emb1, emb2 = model(x1, x2)
            loss = loss_fn(emb1, emb2, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save full training checkpoint for resuming later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


def save_weights(model, path):
    """Save model weights (state_dict only) for distribution and deployment."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Best model weights saved: {path}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    
    # Dynamic hardware selection: utilize CUDA GPU acceleration when available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Siamese backbone based on requested branch
    if args.branch == "spatial":
        model = SpatialSiamese(embedding_dim=SPATIAL_EMBEDDING_DIM).to(device)
        weights_filename = "spatial_cnn.pth"
    else:
        model = TemporalSiamese(
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=TEMPORAL_NUM_LAYERS,
            embedding_dim=TEMPORAL_EMBEDDING_DIM,
            pressure_dropout=PRESSURE_DROPOUT_RATE,
        ).to(device)
        weights_filename = "temporal_lstm.pth"

    loss_fn = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    weights_path = os.path.join(MODEL_WEIGHTS_DIR, weights_filename)

    print(f"Training {args.branch} branch")
    print(f"Data directories: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Hardware Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Training script initialized. Connect DataLoader pipelines to execute training with validation and early stopping.")


if __name__ == "__main__":
    main()

