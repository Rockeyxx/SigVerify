"""
Training Script — Signature Verification System

Entry point for training the Siamese networks.

Usage:
    OPENCV_LOG_LEVEL=ERROR python scripts/train.py --branch spatial --data-dir data/cedar data/bhsig260 data/utsig --epochs 100 --batch-size 32 --num-workers 4 --seed 42
    OPENCV_LOG_LEVEL=ERROR python scripts/train.py --branch temporal --data-dir ./data/svc2004
    OPENCV_LOG_LEVEL=ERROR python scripts/train.py --branch spatial --resume models/weights/checkpoint_spatial.pth
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    SPATIAL_EMBEDDING_DIM, TEMPORAL_EMBEDDING_DIM,
    CONTRASTIVE_MARGIN, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, WEIGHT_DECAY, MODEL_WEIGHTS_DIR,
    TEMPORAL_HIDDEN_SIZE, TEMPORAL_NUM_LAYERS,
    PRESSURE_DROPOUT_RATE, HARD_NEGATIVE_RATIO, SPATIAL_HEAD_DROPOUT,
    SPATIAL_TRUNK, SPATIAL_CONV_DROPOUT, SPATIAL_FC_HIDDEN,
    SPATIAL_GRAD_CHECKPOINT,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    NUM_WORKERS, PIN_MEMORY,
    SPATIAL_SPLIT_RATIOS, TEMPORAL_SPLIT_RATIOS,
    TEMPORAL_PAIRS_GENUINE_PER_WRITER, TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER,
    TEMPORAL_AUGMENT, SPATIAL_AUGMENT,
)
from src.data.dataset import (
    SpatialSignatureDataset, TemporalSignatureDataset,
    collate_temporal_pairs, generate_skilled_pairs, writer_independent_split,
)
from src.data.parsers import merge_datasets, parse_dataset, svc2004_has_pressure
from src.models.siamese import (
    SPATIAL_TRUNKS, SpatialSiamese, TemporalSiamese, ContrastiveLoss,
)


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
            tqdm.write(f"  Early stopping counter: {self.counter}/{self.patience} "
                       f"(Best Val Loss: {self.best_loss:.4f})")
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
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Path to a .pth to initialize model weights from before "
                             "training, for fine-tuning a pre-trained backbone. Unlike "
                             "--resume this loads weights only, not optimizer state")
    parser.add_argument("--weights-dir", type=str, default=MODEL_WEIGHTS_DIR,
                        help="Directory to write weights and checkpoints into")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker processes (use 0 on Windows / in notebooks)")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Early-stopping patience. Small datasets give a noisy "
                             "validation curve, so a low value can halt a run whose "
                             "training loss is still descending")
    parser.add_argument("--seed", type=int, default=42)

    # Spatial architecture overrides. These exist as flags rather than config
    # edits so an ablation series runs back-to-back from one shell, and so the
    # command line in the log states which architecture produced the weights.
    parser.add_argument("--trunk", choices=sorted(SPATIAL_TRUNKS), default=SPATIAL_TRUNK,
                        help="Spatial convolutional trunk (spatial branch only)")
    parser.add_argument("--conv-dropout", type=float, default=SPATIAL_CONV_DROPOUT,
                        help="Dropout2d rate after the trunk stages that declare it")
    parser.add_argument("--fc-hidden", type=int, default=SPATIAL_FC_HIDDEN,
                        help="Width of an FC layer inserted before the embedding "
                             "projection, or 0 for the single linear head")
    parser.add_argument("--head-dropout", type=float, default=SPATIAL_HEAD_DROPOUT,
                        help="Dropout rate on the pooled vector feeding the head")
    parser.add_argument("--grad-checkpoint", type=int, default=SPATIAL_GRAD_CHECKPOINT,
                        metavar="SEGMENTS",
                        help="Recompute the trunk in this many segments during the "
                             "backward pass instead of storing its activations. The "
                             "arithmetic is unchanged, roughly 15%% slower, and it is "
                             "what lets the wide trunk train at the same batch size "
                             "as the compact one on a 4 GB card. 0 disables it")
    parser.add_argument("--weights-name", type=str, default=None,
                        help="Filename for the saved weights, so an ablation run "
                             "cannot overwrite the shipped spatial_cnn.pth")
    parser.add_argument("--augment", dest="augment", action="store_true",
                        default=SPATIAL_AUGMENT,
                        help="Enable training-time augmentation (spatial branch)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable training-time augmentation")
    parser.add_argument("--lr-schedule", choices=["none", "plateau"], default="none",
                        help="'plateau' halves the learning rate after 2 epochs "
                             "without validation improvement. The spatial validation "
                             "curve is non-monotone, so decaying on a plateau fits it "
                             "better than a fixed step schedule")
    parser.add_argument("--temporal-channels", choices=["xyp", "tilt", "dt"], default="xyp",
                        help="Extra SVC2004 channels to feed the temporal branch. "
                             "'xyp' is the shipped 3-channel contract; 'tilt' adds "
                             "azimuth+altitude (SVC2004 only, and the XPPen reports "
                             "neither); 'dt' adds real inter-sample time")
    parser.add_argument("--pressure-dropout", type=float, default=None,
                        help="Override PRESSURE_DROPOUT_RATE for this run (temporal "
                             "branch only). A Task1 dataset still forces 0, since "
                             "there is no real pressure to drop")
    return parser.parse_args(argv)


def build_writer_samples(data_dirs, extra_channels=None):
    """Parse and merge every requested dataset into one writer-keyed mapping."""
    writer_samples = {}
    for directory in data_dirs:
        # Only SVC2004 declares extra channels; asking any other parser for them
        # would raise, which is why this is conditional rather than always passed.
        kwargs = ({"extra_channels": extra_channels}
                  if extra_channels and "svc2004" in os.path.basename(
                      os.path.normpath(directory)).lower() else {})
        parsed = parse_dataset(directory, **kwargs)
        print(f"  {directory}: {len(parsed)} writers")
        merge_datasets(writer_samples, parsed)
    if not writer_samples:
        raise ValueError(f"No writers found in {data_dirs}")
    return writer_samples


def build_loader(writer_samples, writer_ids, branch, batch_size, num_workers, shuffle, seed,
                 augment=SPATIAL_AUGMENT):
    """
    Build a DataLoader over skilled-forgery pairs drawn from `writer_ids` only.

    Pairs are generated *after* the writer split, so no writer's signatures can
    appear on both sides of the partition boundary.

    The temporal branch draws more pairs per writer than the spatial one, since
    SVC2004 offers a fraction of the writers the image datasets do.
    """
    subset = {w: writer_samples[w] for w in writer_ids}

    if branch == "spatial":
        pairs, labels = generate_skilled_pairs(subset, seed=seed)
        # `shuffle` is true only for the training loader, so validation and test
        # are never augmented — an augmented val set would move the target the
        # early-stopping rule is measuring against.
        dataset = SpatialSignatureDataset(pairs, labels,
                                          augment=shuffle and augment)
        collate = None
    else:
        pairs, labels = generate_skilled_pairs(
            subset, TEMPORAL_PAIRS_GENUINE_PER_WRITER,
            TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER, seed=seed)
        # `shuffle` is true only for the training loader, which is also the only
        # split that may be augmented.
        dataset = TemporalSignatureDataset(pairs, labels,
                                           augment=shuffle and TEMPORAL_AUGMENT)
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


def epoch_label(epoch, total_epochs=None):
    """Human-facing 1-based epoch tag for progress bars ('Epoch 003/100')."""
    if total_epochs is None:
        return f"Epoch {epoch + 1:03d}"
    return f"Epoch {epoch + 1:03d}/{total_epochs:03d}"


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, total_epochs=None):
    """
    Train model for a single epoch.
    Returns average training loss across all batches.

    Progress reports through `tqdm.auto`, which renders a widget under
    Colab/Jupyter and a plain bar in a terminal, so the notebook and the CLI both
    get live feedback from this one call site.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    bar = tqdm(dataloader, desc=f"{epoch_label(epoch, total_epochs)} [train]", leave=False)
    for batch in bar:
        # Transfer batch tensors to active computing device (GPU or CPU)
        x1, x2, labels, lengths = unpack_batch(batch, device)

        optimizer.zero_grad()
        emb1, emb2 = model(x1, x2, *lengths) if lengths else model(x1, x2)
        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def validate_one_epoch(model, dataloader, loss_fn, device, epoch=None, total_epochs=None):
    """
    Evaluate validation loss on holdout set without computing gradients.
    Returns average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    desc = "validate" if epoch is None else f"{epoch_label(epoch, total_epochs)} [val]  "
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
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
    tqdm.write(f"  Checkpoint saved: {path}")


def save_weights(model, path):
    """Save model weights (state_dict only) for distribution and deployment."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    tqdm.write(f"  Best model weights saved: {path}")


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs,
        weights_path, checkpoint_path, start_epoch=0, early_stopping=None,
        scheduler=None):
    """
    Run the full training loop: per-epoch checkpoint, best-validation weight
    save, and early stopping.

    Kept as a standalone function so the notebook can drive the same loop the
    CLI does instead of carrying its own copy.

    Returns a history dict of "train_loss" / "val_loss" lists, which is
    everything a loss curve needs, so callers can plot after the run without
    training again. The best validation loss is `min(history["val_loss"])`.
    """
    if early_stopping is None:
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                       min_delta=EARLY_STOPPING_MIN_DELTA)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     device, epoch, epochs)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device, epoch, epochs)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        tqdm.write(f"{epoch_label(epoch, epochs)} | train {train_loss:.4f} | val {val_loss:.4f}")

        # Stepped on validation, not training, loss: the point is to decay when
        # generalization stalls, which is the only signal that matters here.
        if scheduler is not None:
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < previous_lr:
                tqdm.write(f"  LR decayed: {previous_lr:.2e} -> {current_lr:.2e}")

        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Only the best-validation weights ship — the final epoch is not
        # necessarily the best one once early stopping has been counting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(model, weights_path)

        if early_stopping(val_loss):
            tqdm.write(f"Early stopping triggered at {epoch_label(epoch, epochs)}")
            break

    return history


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)

    # Dynamic hardware selection: utilize CUDA GPU acceleration when available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {args.branch} branch")
    print(f"Hardware Device: {device}")
    print("Discovering writers:")
    writer_samples = build_writer_samples(
        args.data_dir,
        extra_channels=None if args.temporal_channels == "xyp" else args.temporal_channels)

    # Writer-independent partitioning (ISO/IEC 23053) happens on writer IDs
    # before any pair is formed, so no signer can straddle two partitions.
    ratios = (TEMPORAL_SPLIT_RATIOS if args.branch == "temporal"
              else SPATIAL_SPLIT_RATIOS)
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(writer_samples), *ratios, seed=args.seed)
    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    if args.branch == "spatial":
        model = SpatialSiamese(embedding_dim=SPATIAL_EMBEDDING_DIM,
                               head_dropout=args.head_dropout,
                               trunk=args.trunk,
                               conv_dropout=args.conv_dropout,
                               fc_hidden=args.fc_hidden,
                               grad_checkpoint=args.grad_checkpoint).to(device)
        print(f"  Trunk: {args.trunk}, conv dropout {args.conv_dropout}, "
              f"head dropout {args.head_dropout}, "
              f"FC hidden {args.fc_hidden or 'none'}, "
              f"grad checkpoint {args.grad_checkpoint or 'off'}")
        weights_filename = "spatial_cnn.pth"
    else:
        # Pressure-channel dropout only teaches robustness if real pressure exists
        # to drop. A Task1 (4-column) dataset has none, so the branch would train
        # permanently pressure-blind — surface that rather than let it pass silently.
        pressure_dropout = (args.pressure_dropout if args.pressure_dropout is not None
                            else PRESSURE_DROPOUT_RATE)
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

        channel_count = {"xyp": 3, "dt": 4, "tilt": 5}[args.temporal_channels]
        if channel_count != 3:
            print(f"  Temporal channels: {args.temporal_channels} "
                  f"({channel_count}ch) — NOT the shipped [L, 3] capture contract")
        model = TemporalSiamese(
            input_features=channel_count,
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=TEMPORAL_NUM_LAYERS,
            embedding_dim=TEMPORAL_EMBEDDING_DIM,
            pressure_dropout=pressure_dropout,
        ).to(device)

        # Distinct filenames so the pressure-blind ablation can never silently
        # overwrite the shipped pressure-aware weights (AGENTS.md pressure ablation).
        weights_filename = ("temporal_lstm.pth" if has_pressure
                            else "temporal_lstm_task1.pth")

    if args.weights_name:
        weights_filename = args.weights_name

    train_loader, num_train = build_loader(writer_samples, train_ids, args.branch, args.batch_size,
                                           args.num_workers, True, args.seed,
                                           augment=args.augment)
    val_loader, num_val = build_loader(writer_samples, val_ids, args.branch, args.batch_size,
                                       args.num_workers, False, args.seed,
                                       augment=args.augment)
    print(f"  Pairs: {num_train} train / {num_val} val (skilled-forgery negatives)")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    loss_fn = ContrastiveLoss(margin=CONTRASTIVE_MARGIN,
                              hard_negative_ratio=HARD_NEGATIVE_RATIO)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=args.patience,
                                   min_delta=EARLY_STOPPING_MIN_DELTA)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    if args.init_weights:
        # Fine-tuning, not resuming: take the weights and nothing else. The
        # optimizer state and epoch counter belong to the pre-training dataset,
        # and carrying them over would start this run mid-schedule against data
        # the optimizer has never seen.
        model.load_state_dict(torch.load(args.init_weights, map_location=device))
        print(f"Initialized from pre-trained weights: {args.init_weights}")

    weights_path = os.path.join(args.weights_dir, weights_filename)
    checkpoint_path = os.path.join(
        args.weights_dir, f"checkpoint_{os.path.splitext(weights_filename)[0]}.pth")

    # Decay patience sits below early-stopping patience on purpose: the run gets
    # at least one decayed-LR epoch to prove itself before stopping is allowed.
    scheduler = None
    if args.lr_schedule == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)
        print(f"  LR schedule: plateau (halve after 2 stalled epochs, "
              f"early-stopping patience {args.patience})")

    history = fit(model, train_loader, val_loader, loss_fn, optimizer, device,
                  args.epochs, weights_path, checkpoint_path,
                  start_epoch=start_epoch, early_stopping=early_stopping,
                  scheduler=scheduler)

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    print(f"Weights: {weights_path}")
    return {**history,
            "split": {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}}


if __name__ == "__main__":
    main()

