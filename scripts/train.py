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
    PRESSURE_DROPOUT_RATE, HARD_NEGATIVE_RATIO,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    NUM_WORKERS, PIN_MEMORY,
)
from src.data.dataset import (
    SpatialSignatureDataset, TemporalSignatureDataset,
    collate_temporal_pairs, generate_skilled_pairs, writer_independent_split,
)
from src.data.parsers import parse_dataset, svc2004_has_pressure
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Siamese signature verification models")
    parser.add_argument("--branch", choices=["spatial", "temporal"], required=True,
                        help="Which branch to train")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Path(s) to dataset directories (cedar, bhsig260, utsig, svc2004)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--weights-dir", type=str, default=MODEL_WEIGHTS_DIR,
                        help="Directory to write weights and checkpoints into")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker processes (use 0 on Windows / in notebooks)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_writer_corpus(data_dirs):
    """Parse and merge every requested dataset into one writer-keyed corpus."""
    corpus = {}
    for directory in data_dirs:
        parsed = parse_dataset(directory)
        print(f"  {directory}: {len(parsed)} writers")
        corpus.update(parsed)
    if not corpus:
        raise ValueError(f"No writers found in {data_dirs}")
    return corpus


def build_loader(corpus, writer_ids, branch, batch_size, num_workers, shuffle, seed):
    """
    Build a DataLoader over skilled-forgery pairs drawn from `writer_ids` only.

    Pairs are generated *after* the writer split, so no writer's signatures can
    appear on both sides of the partition boundary.
    """
    subset = {w: corpus[w] for w in writer_ids}
    pairs, labels = generate_skilled_pairs(subset, seed=seed)

    if branch == "spatial":
        dataset = SpatialSignatureDataset(pairs, labels)
        collate = None
    else:
        dataset = TemporalSignatureDataset(pairs, labels)
        collate = collate_temporal_pairs

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=PIN_MEMORY,
                        collate_fn=collate)
    return loader, len(pairs)


def unpack_batch(batch, device):
    """
    Normalize a batch from either collate path into a uniform call signature.

    Spatial batches are (x1, x2, labels). Temporal batches carry the true
    sequence lengths the packing needs: (x1, len1, x2, len2, labels).
    """
    if len(batch) == 5:
        x1, lengths1, x2, lengths2, labels = batch
        return x1.to(device), x2.to(device), labels.to(device), (lengths1, lengths2)
    x1, x2, labels = batch
    return x1.to(device), x2.to(device), labels.to(device), None


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """
    Train model for a single epoch.
    Returns average training loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Transfer batch tensors to active computing device (GPU or CPU)
        x1, x2, labels, lengths = unpack_batch(batch, device)

        optimizer.zero_grad()
        emb1, emb2 = model(x1, x2, *lengths) if lengths else model(x1, x2)
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
        for batch in dataloader:
            x1, x2, labels, lengths = unpack_batch(batch, device)
            emb1, emb2 = model(x1, x2, *lengths) if lengths else model(x1, x2)
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


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)

    # Dynamic hardware selection: utilize CUDA GPU acceleration when available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {args.branch} branch")
    print(f"Hardware Device: {device}")
    print("Discovering writers:")
    corpus = build_writer_corpus(args.data_dir)

    # Writer-independent partitioning (ISO/IEC 23053) happens on writer IDs
    # before any pair is formed, so no signer can straddle two partitions.
    train_ids, val_ids, test_ids = writer_independent_split(sorted(corpus), seed=args.seed)
    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    if args.branch == "spatial":
        model = SpatialSiamese(embedding_dim=SPATIAL_EMBEDDING_DIM).to(device)
        weights_filename = "spatial_cnn.pth"
    else:
        # Pressure-channel dropout only teaches robustness if real pressure exists
        # to drop. A Task1 (4-column) corpus has none, so the branch would train
        # permanently pressure-blind — surface that rather than let it pass silently.
        pressure_dropout = PRESSURE_DROPOUT_RATE
        has_pressure = True
        for directory in args.data_dir:
            if os.path.basename(os.path.normpath(directory)).lower().startswith("svc2004"):
                if svc2004_has_pressure(directory):
                    print(f"  {directory}: Task2 (7-column) — real pressure, "
                          f"dropout active at {pressure_dropout:.0%}")
                else:
                    print(f"  WARNING: {directory} is SVC2004 Task1 (4-column, no pressure).")
                    print("           P is imputed to 1.0 for every sample, so pressure "
                          "dropout is a no-op and is being disabled.")
                    print("           This run is the pressure-blind ablation baseline (Run A).")
                    pressure_dropout = 0.0
                    has_pressure = False

        model = TemporalSiamese(
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=TEMPORAL_NUM_LAYERS,
            embedding_dim=TEMPORAL_EMBEDDING_DIM,
            pressure_dropout=pressure_dropout,
        ).to(device)

        # Distinct filenames so the pressure-blind ablation can never silently
        # overwrite the shipped pressure-aware weights (AGENTS.md pressure ablation).
        weights_filename = ("temporal_lstm.pth" if has_pressure
                            else "temporal_lstm_task1.pth")

    train_loader, num_train = build_loader(corpus, train_ids, args.branch, args.batch_size,
                                           args.num_workers, True, args.seed)
    val_loader, num_val = build_loader(corpus, val_ids, args.branch, args.batch_size,
                                       args.num_workers, False, args.seed)
    print(f"  Pairs: {num_train} train / {num_val} val (skilled-forgery negatives)")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    loss_fn = ContrastiveLoss(margin=CONTRASTIVE_MARGIN,
                              hard_negative_ratio=HARD_NEGATIVE_RATIO)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                   min_delta=EARLY_STOPPING_MIN_DELTA)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    weights_path = os.path.join(args.weights_dir, weights_filename)
    checkpoint_path = os.path.join(
        args.weights_dir, f"checkpoint_{os.path.splitext(weights_filename)[0]}.pth")

    history = {
        "train_loss": [],
        "val_loss": [],
        "split": {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)},
    }
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Only the best-validation weights ship — the final epoch is not
        # necessarily the best one once early stopping has been counting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(model, weights_path)

        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    print(f"Weights: {weights_path}")
    return history


if __name__ == "__main__":
    main()

