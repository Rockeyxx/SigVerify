"""Training pipeline for signature verification models.

Entry point script for training spatial (CNN) and temporal (LSTM) Siamese networks.
"""


import argparse
import os

# cuBLAS picks a non-deterministic reduction unless told this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from config.settings import (
    SEED,
    TEMPORAL_CHANNELS, TEMPORAL_CHANNEL_WIDTHS, TEMPORAL_INPUT_FEATURES,
    TEMPORAL_TILT_HIDE_PROB, TILT_CHANNEL_MODES,
    SPATIAL_EMBEDDING_DIM, TEMPORAL_EMBEDDING_DIM,
    CONTRASTIVE_MARGIN, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, WEIGHT_DECAY, MODEL_WEIGHTS_DIR,
    TEMPORAL_HIDDEN_SIZE, TEMPORAL_NUM_LAYERS,
    PRESSURE_DROPOUT_RATE, HARD_NEGATIVE_RATIO, SPATIAL_HEAD_DROPOUT,
    SPATIAL_TRUNK, SPATIAL_CONV_DROPOUT, SPATIAL_FC_HIDDEN,
    SPATIAL_GRAD_CHECKPOINT, SPATIAL_INPUT_SIZE,
    PAIRS_GENUINE_PER_WRITER, PAIRS_IMPOSTOR_PER_WRITER,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    NUM_WORKERS, PIN_MEMORY,
    SPATIAL_SPLIT_RATIOS, TEMPORAL_SPLIT_RATIOS,
    TEMPORAL_PAIRS_GENUINE_PER_WRITER, TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER,
    TEMPORAL_AUGMENT, SPATIAL_AUGMENT, RANDOM_IMPOSTOR_RATIO, TEMPORAL_TRAIN_FLOAT32_PARSE,
)
from src.data.dataset import (
    SpatialSignatureDataset, TemporalSignatureDataset,
    collate_temporal_pairs, generate_skilled_pairs, pin_holdout,
    writer_independent_split,
    save_split_record, split_record_path)
from src.data.parsers import (
    EXTRA_CHANNEL_SUPPORT, MAX_GAP_MS, is_known_dataset, is_temporal_dataset, merge_datasets,
    parse_dataset, supports_extra_channel,
    svc2004_has_pressure,
)
from src.models.siamese import (
    SPATIAL_TRUNKS, SpatialSiamese, TemporalSiamese, ContrastiveLoss,
)


class EarlyStopping:
    """
    Early stopping monitor tracking validation loss to halt training when convergence plateaus.
    Saves state parameters corresponding to lowest validation error.
    """

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def state_dict(self):
        """The two fields a resume needs; patience and min_delta come from the flags."""
        return {"counter": self.counter, "best_loss": self.best_loss}

    def load_state(self, state):
        """Restore a saved counter, or start fresh for a checkpoint predating the field."""
        if state:
            self.counter = state["counter"]
            self.best_loss = state["best_loss"]

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



# Options for closed experiments were removed on 18 November 2026. Each one was measured
# worse and closed in docs/ENG_REPORT.md:
#   --triplet-weight, --triplet-margin, --cross-session-positives   time-gap arms (F32 to F37)
#   --conv-filters, --conv-kernel                                    conv before the LSTM (C15)
#   --utsig-forgery-types, --utsig-hold-pairs                        UTSig forgery types (D11)
#   --temporal-features derived                                      derived time functions
# They are all in commit 3c80324. To rerun one of those arms:
#   git show 3c80324:scripts/train.py > /tmp/train_3c80324.py   (and the matching
#   src/data/dataset.py and src/models/siamese.py from the same commit)
# TemporalSiamese still loads a checkpoint that has a conv stage, and benchmark.py
# still scores a derived-features checkpoint, so the archived weights stay usable.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Siamese signature verification models")
    parser.add_argument("--branch", choices=["spatial", "temporal"], required=True,
                        help="Which branch to train")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Path(s) to dataset directories (cedar, bhsig260, utsig, svc2004)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--augment-scale", type=float, default=1.0,
                        help="Multiplier on every temporal augmentation range "
                             "(rotation, jitter, speed change). 1.0 is the default "
                             "strength, which measured 2.54 EER points worse than no "
                             "augmentation at all — so strength is the untested axis")
    parser.add_argument("--hidden-size", type=int, default=TEMPORAL_HIDDEN_SIZE,
                        help="Temporal LSTM width. 256x2 layers is ~2.2M parameters on "
                             "62 training writers; TA-RNN uses 46+46->23 memory blocks "
                             "and DsDTW a GRU-128, so smaller is worth measuring")
    parser.add_argument("--num-layers", type=int, default=TEMPORAL_NUM_LAYERS,
                        help="Temporal LSTM depth. Inter-layer dropout only exists when "
                             "this is above 1, so dropping to 1 removes that too")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Adam L2 penalty. The temporal branch best-validates at "
                             "epoch 1 on 62 writers, which is the shape of an "
                             "over-capacity model, so this is a lever worth sweeping "
                             "there. Note the opposite finding on spatial: at 301 "
                             "writers regularizing removed capacity the model needed")
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
    parser.add_argument("--seed", type=int, default=SEED)

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
    parser.add_argument("--pool-grid", type=int, default=4,
                        help="N for the N x N avg+max pooling grid before the head "
                             "(spatial branch only). Recoverable from the checkpoint.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Embedding width; defaults to the branch's config value. "
                             "Recoverable from the checkpoint.")
    parser.add_argument("--aug-rotation", type=float, default=None, metavar="DEG",
                        help="Spatial augmentation: max rotation (default SPATIAL_AUG_MAX_ROTATION_DEG)")
    parser.add_argument("--aug-shear", type=float, default=None,
                        help="Spatial augmentation: max shear (default SPATIAL_AUG_MAX_SHEAR)")
    parser.add_argument("--aug-scale", type=float, default=None,
                        help="Spatial augmentation: max scale change (default SPATIAL_AUG_MAX_SCALE)")
    parser.add_argument("--aug-translate", type=float, default=None,
                        help="Spatial augmentation: max translation (default SPATIAL_AUG_MAX_TRANSLATE)")
    parser.add_argument("--aug-stroke-prob", type=float, default=None,
                        help="Spatial augmentation: stroke dilate/erode probability (default SPATIAL_AUG_STROKE_PROB)")
    parser.add_argument("--aug-blur", type=float, default=None, metavar="SIGMA",
                        help="Spatial augmentation: max Gaussian blur sigma in pixels (default SPATIAL_AUG_BLUR_SIGMA, off)")
    parser.add_argument("--aug-ink-gain-min", type=float, default=None,
                        help="Spatial augmentation: ink intensity scaled by U(min, 1) (default SPATIAL_AUG_INK_GAIN_MIN, off)")
    parser.add_argument("--aug-noise", type=float, default=None, metavar="STD",
                        help="Spatial augmentation: max additive noise sigma (default SPATIAL_AUG_NOISE_STD, off)")
    parser.add_argument("--device-augment", action="store_true",
                        help="Temporal branch: re-record each training trajectory as another tablet "
                             "inside the capture spec would deliver it (TEMPORAL_DEVICE_AUG_*)")
    parser.add_argument("--unidirectional", action="store_true",
                        help="Temporal branch: forward-only LSTM instead of bidirectional. "
                             "Recoverable from the checkpoint.")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "rmsprop"], default="adam",
                        help="Optimiser; adam is the shipped recipe")
    parser.add_argument("--min-delta", type=float, default=EARLY_STOPPING_MIN_DELTA,
                        help="Early-stopping improvement threshold on validation loss")
    parser.add_argument("--pairs-per-writer", type=int, default=None,
                        help="Pairs drawn per writer per class for train, val and "
                             "test alike, on whichever branch is training (default "
                             "PAIRS_*_PER_WRITER for spatial, TEMPORAL_PAIRS_*_PER_WRITER "
                             "for temporal). A reduced budget for screening runs; the "
                             "shipped recipe uses the config value.")
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
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Override the branch's writer-split ratios. The temporal "
                             "default is 50/25/25 only because SVC2004's 40 writers "
                             "would otherwise leave 6 test writers and an EER too "
                             "noisy to compare; a corpus in the hundreds can afford "
                             "the spatial branch's 70/15/15 and buy training writers "
                             "back. Pass the same value to benchmark.py or the split "
                             "will not match the one the model trained on.")
    parser.add_argument("--augment", dest="augment", action="store_true",
                        default=None,
                        help="Enable training-time augmentation. Defaults to the "
                             "branch's config constant (SPATIAL_AUGMENT / "
                             "TEMPORAL_AUGMENT); passing this overrides it")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable training-time augmentation")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for the writer split only. Defaults to --seed. "
                             "Hold it fixed while varying --seed to re-run a config "
                             "on the same held-out writers, which is the only way to "
                             "measure training variance without split variance on top")
    parser.add_argument("--margin", type=float, default=CONTRASTIVE_MARGIN,
                        help="Contrastive margin. Never swept — it carries its "
                             "original guessed value. Embeddings are NOT "
                             "L2-normalised, so this is not scale-free: at 1.0 it "
                             "sits just above the impostor mean distance of 0.9393, "
                             "meaning most impostor pairs contribute zero loss")
    parser.add_argument("--hard-negative-ratio", type=float, default=HARD_NEGATIVE_RATIO,
                        help="Fraction of impostor terms kept by mining. Also never "
                             "swept. 1.0 disables mining")
    parser.add_argument("--holdout-writers", nargs="+", metavar="ID", default=None,
                        help="Writer IDs forced into the test partition. Adding a "
                             "corpus otherwise reshuffles every writer, moving the "
                             "ones an earlier result was scored on — some into "
                             "training. Pin them to keep a comparison honest, and "
                             "pass the same IDs to benchmark.py --writers")
    parser.add_argument("--random-impostor-ratio", type=float,
                        default=RANDOM_IMPOSTOR_RATIO, metavar="R",
                        help="Fraction of the training negative budget drawn from "
                             "other writers' genuine signatures instead of this "
                             "writer's skilled forgeries. Training only — validation "
                             "keeps pure skilled-forgery pairs. At 0 the encoder "
                             "never learns that two different people should be far "
                             "apart, which measurement showed it does not infer")
    parser.add_argument("--lr-schedule", choices=["none", "plateau"], default="none",
                        help="'plateau' halves the learning rate after 2 epochs "
                             "without validation improvement. The spatial validation "
                             "curve is non-monotone, so decaying on a plateau fits it "
                             "better than a fixed step schedule")
    parser.add_argument("--temporal-channels", choices=sorted(TEMPORAL_CHANNEL_WIDTHS),
                        default=TEMPORAL_CHANNELS,
                        help=f"Extra channels to feed the temporal branch. "
                             f"'{TEMPORAL_CHANNELS}' ({TEMPORAL_INPUT_FEATURES}ch) ships; "
                             f"'xyp' is the legacy 3-channel contract; 'tilt' adds "
                             "azimuth+altitude (SVC2004 only, and the XPPen reports "
                             "neither); 'dt' adds inter-sample time, measured from "
                             "SVC2004's clock and reconstructed on SigComp2011 from "
                             "the pen-up runs it drops, so both corpora can supply it; "
                             "dt_tilt0 / dt_tilt1 / dt_tilt_flag (research only, see "
                             "AGENTS.md)")
    parser.add_argument("--pressure-dropout", type=float, default=None,
                        help="Override PRESSURE_DROPOUT_RATE for this run (temporal "
                             "branch only). A Task1 dataset still forces 0, since "
                             "there is no real pressure to drop")
    parser.add_argument("--train-only-data-dir", nargs="+", default=None, metavar="DIR",
                        help="Spatial only, research. Image corpora whose listed "
                             "writers (--train-only-writers) join training after the "
                             "split, so the --data-dir partition does not move")
    parser.add_argument("--train-only-writers", nargs="+", default=None, metavar="ID",
                        help="Writer IDs of --train-only-data-dir to train on")
    args = parser.parse_args(argv)
    if args.train_only_data_dir and (args.branch != "spatial" or not args.train_only_writers):
        parser.error("--train-only-data-dir is spatial only and needs --train-only-writers")
    if args.split_seed is None:
        args.split_seed = args.seed
    return args


def build_writer_samples(data_dirs, extra_channels=None, float32_parse=False):
    """Parse and merge every requested dataset into one writer-keyed mapping.

    float32_parse reads SVC2004 and DeepSignDB as training does
    (TEMPORAL_TRAIN_FLOAT32_PARSE); every scorer leaves it off.
    """
    writer_samples = {}
    for directory in data_dirs:
        # Refusing here is the point. Every corpus in one run must return the
        # same width, so a corpus that cannot supply the channel has to stop the
        # run rather than quietly contribute narrower samples to the same merge.
        kwargs = {}
        # Only the temporal corpora are asked for the channel. A fused run lists
        # image and trajectory corpora together, and demanding `dt` from CEDAR
        # aborted the whole run — `--branch fused` was unrunnable under the
        # shipped default the moment `dt` became it.
        if extra_channels and is_temporal_dataset(directory):
            if is_known_dataset(directory) and not supports_extra_channel(
                    directory, extra_channels):
                raise ValueError(
                    f"'{directory}' cannot supply the '{extra_channels}' channel. "
                    f"Corpora that can: "
                    f"{', '.join(sorted(c for c, ch in EXTRA_CHANNEL_SUPPORT.items() if extra_channels in ch))}."
                )
            kwargs = {"extra_channels": extra_channels}
        parsed = parse_dataset(directory, float32_parse=float32_parse, **kwargs)
        print(f"  {directory}: {len(parsed)} writers")
        merge_datasets(writer_samples, parsed)
    if not writer_samples:
        raise ValueError(f"No writers found in {data_dirs}")
    return writer_samples


def add_train_only_writers(writer_samples, train_ids, data_dirs, writers):
    """
    Add the listed writers of `data_dirs` to training, after the split.

    The split is drawn on --data-dir alone, so validation, test and the base
    training writers stay exactly where they were. writer_samples gains the
    listed writers in place; the returned list is train_ids plus them.
    """
    extra = {}
    for directory in data_dirs:
        merge_datasets(extra, parse_dataset(directory))
    missing = sorted(set(writers) - set(extra))
    if missing:
        raise ValueError(f"Train-only writers absent from {data_dirs}: {', '.join(missing)}")
    clash = sorted(set(writers) & set(writer_samples))
    if clash:
        raise ValueError(f"Train-only writers already in --data-dir: {', '.join(clash)}")
    for writer in sorted(writers):
        writer_samples[writer] = extra[writer]
    return list(train_ids) + sorted(writers)


def _seed_worker(worker_id):
    """
    Give each DataLoader worker its own augmentation RNG stream.

    Workers are forked, so without this they inherit one already-initialized
    generator and replay the *same* rotation/shear/stroke-width draws in
    lockstep — the augmentation still runs, but its diversity silently collapses
    toward a single worker's worth. Deriving each stream from the dataset's own
    seed keeps the run reproducible from `--seed`.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    base = getattr(info.dataset, "_seed", None)
    if base is None:
        return
    info.dataset._rng = np.random.default_rng(base + worker_id)


def build_loader(writer_samples, writer_ids, branch, batch_size, num_workers, shuffle, seed,
                 augment=SPATIAL_AUGMENT, random_impostor_ratio=RANDOM_IMPOSTOR_RATIO,
                 augment_scale=1.0, pairs_per_writer=None, augment_params=None,
                 device_augment=False, tilt_hide_prob=0.0):
    """
    Build a DataLoader over skilled-forgery pairs drawn from `writer_ids` only.

    Pairs are generated *after* the writer split, so no writer's signatures can
    appear on both sides of the partition boundary.

    The temporal branch draws more pairs per writer than the spatial one, since
    SVC2004 offers a fraction of the writers the image datasets do.
    """
    subset = {w: writer_samples[w] for w in writer_ids}

    # Random impostors are a training-time supervision signal, never an
    # evaluation one: `shuffle` is true only for the training loader, so val and
    # test keep the pure skilled-forgery pairing every reported metric uses.
    ratio = random_impostor_ratio if shuffle else 0.0

    if branch == "spatial":
        genuine_n = pairs_per_writer or PAIRS_GENUINE_PER_WRITER
        forged_n = pairs_per_writer or PAIRS_IMPOSTOR_PER_WRITER
        pairs, labels = generate_skilled_pairs(subset, genuine_n, forged_n, seed=seed,
                                               random_impostor_ratio=ratio)
        # `shuffle` is true only for the training loader, so validation and test
        # are never augmented — an augmented val set would move the target the
        # early-stopping rule is measuring against.
        dataset = SpatialSignatureDataset(pairs, labels,
                                          augment=shuffle and augment,
                                          seed=seed,
                                          augment_params=augment_params)
        collate = None
    else:
        pairs, labels = generate_skilled_pairs(
            subset, pairs_per_writer or TEMPORAL_PAIRS_GENUINE_PER_WRITER,
            pairs_per_writer or TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER, seed=seed,
            random_impostor_ratio=ratio)
        # `shuffle` is true only for the training loader, which is also the only
        # split that may be augmented.
        #
        # This read `shuffle and TEMPORAL_AUGMENT`, which pinned the branch to the
        # config constant and made `--augment` a no-op for temporal while the
        # spatial path above honoured it. `augment_trajectory` had therefore never
        # run. The symptom that exposed it: temporal best-validates at epoch 1,
        # which is the same data-variety starvation augmentation fixed on spatial
        # (best epoch 1 -> 6, 11.78% -> 11.19%). Config still supplies the default.
        dataset = TemporalSignatureDataset(pairs, labels,
                                           augment=shuffle and augment,
                                           augment_scale=augment_scale,
                                           seed=seed,
                                           device_augment=shuffle and device_augment,
                                           tilt_hide_prob=tilt_hide_prob if shuffle else 0.0)
        collate = collate_temporal_pairs

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=PIN_MEMORY,
                        collate_fn=collate, worker_init_fn=_seed_worker)
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


def rng_state(dataset=None):
    """Every random stream an epoch draws from: torch on CPU and CUDA (shuffle order, dropout,
    worker seeds) and the training set's own augmentation generator. Reading it draws nothing,
    so saving it leaves an uninterrupted run exactly as it was."""
    generator = getattr(dataset, "_rng", None)
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "dataset": generator.bit_generator.state if generator is not None else None}


def set_rng_state(state, dataset=None):
    """Put back what rng_state() saved, so the next epoch draws what it would have drawn."""
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["dataset"] is not None and getattr(dataset, "_rng", None) is not None:
        dataset._rng.bit_generator.state = state["dataset"]


def save_checkpoint(model, optimizer, epoch, loss, path, best_val_loss=None,
                    early_stopping=None, scheduler=None, rng=None):
    """
    Save full training checkpoint for resuming later.

    `loss` is this epoch's validation loss; `best_val_loss` is the best seen
    across the whole run. Both are stored because they answer different
    questions, and resuming needs the second one. Restoring from `loss` alone
    re-baselines the run against whatever the final epoch happened to score, so
    an epoch worse than the true best still reads as an improvement and
    overwrites the good weights.

    `scheduler` and `rng` make a resumed run end with the same weights as one
    that never stopped (tests/test_scripts.py::TestResumeIsExact).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # skylos: ignore[SKY-D215] operator-supplied --weights-dir
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_val_loss": best_val_loss,
        # Save early stopping state so resuming training preserves the patience counter.
        "early_stopping": early_stopping.state_dict() if early_stopping else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": rng,
    }, path)
    tqdm.write(f"  Checkpoint saved: {path}")


def save_weights(model, path):
    """Save model weights (state_dict only) for distribution and deployment."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # skylos: ignore[SKY-D215] operator-supplied --weights-dir
    torch.save(model.state_dict(), path)
    tqdm.write(f"  Best model weights saved: {path}")


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs,
        weights_path, checkpoint_path, start_epoch=0, early_stopping=None,
        scheduler=None, resume_best=None, resume_early_stopping=None,
        resume_scheduler=None, resume_rng=None):
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
    early_stopping.load_state(resume_early_stopping)
    train_dataset = getattr(train_loader, "dataset", None)
    if scheduler is not None and resume_scheduler is not None:
        scheduler.load_state_dict(resume_scheduler)
    if resume_rng is not None:
        set_rng_state(resume_rng, train_dataset)

    history = {"train_loss": [], "val_loss": []}
    # Resuming must inherit the best validation loss reached so far. Starting
    # from infinity makes the first resumed epoch "best" by definition, so a
    # *worse* model overwrites good weights and early stopping restarts its
    # count against an inflated baseline. Measured: a run stopped at val 0.1153
    # and resumed recorded the next epoch's 0.1202 as its new best.
    best_val_loss = float("inf") if resume_best is None else resume_best

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

        # Count this epoch before saving: a checkpoint written first carries the
        # previous epoch's counter, and a resumed run falls one stall behind.
        stop = early_stopping(val_loss)
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path,
                        best_val_loss=min(best_val_loss, val_loss),
                        early_stopping=early_stopping, scheduler=scheduler,
                        rng=rng_state(train_dataset))

        # Only the best-validation weights ship — the final epoch is not
        # necessarily the best one once early stopping has been counting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(model, weights_path)

        if stop:
            tqdm.write(f"Early stopping triggered at {epoch_label(epoch, epochs)}")
            break

    return history


def build_model(args, device):
    """The model the flags describe, and the file name its weights are saved under."""
    if args.branch == "spatial":
        model = SpatialSiamese(embedding_dim=args.embedding_dim or SPATIAL_EMBEDDING_DIM,
                               head_dropout=args.head_dropout,
                               trunk=args.trunk,
                               conv_dropout=args.conv_dropout,
                               fc_hidden=args.fc_hidden,
                               pool_grid=args.pool_grid,
                               grad_checkpoint=args.grad_checkpoint).to(device)
        print(f"  Trunk: {args.trunk}, conv dropout {args.conv_dropout}, "
              f"head dropout {args.head_dropout}, "
              f"FC hidden {args.fc_hidden or 'none'}, "
              f"pool grid {args.pool_grid}x{args.pool_grid}, "
              f"embedding {model.embedding_dim}, "
              f"grad checkpoint {args.grad_checkpoint or 'off'}")
        # The size is not in the weights, so it is stated here and in the record.
        print(f"  Input size: {SPATIAL_INPUT_SIZE[1]}x{SPATIAL_INPUT_SIZE[0]} (HxW)"
              + (" from SIGVER_SPATIAL_INPUT_SIZE" if os.environ.get("SIGVER_SPATIAL_INPUT_SIZE") else ""))
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

        channel_count = TEMPORAL_CHANNEL_WIDTHS[args.temporal_channels]
        if args.temporal_channels != TEMPORAL_CHANNELS:
            print(f"  Temporal channels: {args.temporal_channels} "
                  f"({channel_count}ch) — NOT the shipped '{TEMPORAL_CHANNELS}' "
                  f"contract ([L, {TEMPORAL_INPUT_FEATURES}])")
        model = TemporalSiamese(
            input_features=channel_count,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim or TEMPORAL_EMBEDDING_DIM,
            bidirectional=not args.unidirectional,
            pressure_dropout=pressure_dropout,
        ).to(device)

        # Distinct filenames so pressure-blind runs do not overwrite
        # the shipped pressure-aware weights.
        weights_filename = ("temporal_lstm.pth" if has_pressure
                            else "temporal_lstm_task1.pth")
    return model, weights_filename


def main(argv=None):
    args = parse_args(argv)

    # Seed Python, NumPy, and PyTorch RNGs and enforce deterministic algorithms
    # to guarantee exact run reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Dynamic hardware selection: utilize CUDA GPU acceleration when available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {args.branch} branch")
    print(f"Hardware Device: {device}")
    print("Discovering writers:")
    writer_samples = build_writer_samples(
        args.data_dir,
        # The flag is temporal-only. A spatial run reaches image corpora that
        # have no trajectory at all, so forwarding it there would ask CEDAR for
        # a timing channel and abort a perfectly valid run.
        extra_channels=(None if args.branch != "temporal"
                        or args.temporal_channels == "xyp"
                        else args.temporal_channels),
        # The recipe's only training noise; see TEMPORAL_TRAIN_FLOAT32_PARSE.
        float32_parse=TEMPORAL_TRAIN_FLOAT32_PARSE)

    # Each branch keeps its own documented default; --augment/--no-augment override.
    if args.augment is None:
        args.augment = SPATIAL_AUGMENT if args.branch == "spatial" else TEMPORAL_AUGMENT

    # Writer-independent partitioning (ISO/IEC 23053) happens on writer IDs
    # before any pair is formed, so no signer can straddle two partitions.
    ratios = tuple(args.split_ratios) if args.split_ratios else (
        TEMPORAL_SPLIT_RATIOS if args.branch == "temporal" else SPATIAL_SPLIT_RATIOS)
    # The split seed is separate from the training seed so a config can be
    # re-run on the *same* held-out writers. Without that, changing --seed
    # reshuffles the partition and the two runs score different people, which
    # measures split variance and training variance together — useless for
    # deciding whether one arm actually beat another.
    if args.holdout_writers:
        train_ids, val_ids, test_ids = pin_holdout(
            sorted(writer_samples), args.holdout_writers, *ratios,
            seed=args.split_seed)
        print(f"  Holding out {len(test_ids)} pinned test writers so this run stays "
              f"comparable to results scored on them")
    else:
        train_ids, val_ids, test_ids = writer_independent_split(
            sorted(writer_samples), *ratios, seed=args.split_seed)
    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")
    if args.train_only_data_dir:
        train_ids = add_train_only_writers(writer_samples, train_ids,
                                           args.train_only_data_dir, args.train_only_writers)
        print(f"  Train-only: +{len(args.train_only_writers)} writers from "
              f"{', '.join(args.train_only_data_dir)} ({len(train_ids)} train)")

    model, weights_filename = build_model(args, device)

    if args.weights_name:
        weights_filename = args.weights_name

    augment_params = {k: v for k, v in (("max_rotation_deg", args.aug_rotation),
                                        ("max_shear", args.aug_shear),
                                        ("max_scale", args.aug_scale),
                                        ("max_translate", args.aug_translate),
                                        ("stroke_prob", args.aug_stroke_prob),
                                        ("blur_sigma_max", args.aug_blur),
                                        ("ink_gain_min", args.aug_ink_gain_min),
                                        ("noise_std_max", args.aug_noise)) if v is not None}
    if augment_params:
        print(f"  Augmentation overrides: {augment_params}")
    if args.branch == "temporal":
        print(f"  LSTM: {'forward-only' if args.unidirectional else 'bidirectional'}, "
              f"embedding {model.embedding_dim}, dt clip {MAX_GAP_MS:g} ms"
              + (" from SIGVER_MAX_GAP_MS" if os.environ.get("SIGVER_MAX_GAP_MS") else ""))

    train_loader, num_train = build_loader(writer_samples, train_ids, args.branch, args.batch_size,
                                           args.num_workers, True, args.seed,
                                           augment=args.augment,
                                           augment_scale=args.augment_scale,
                                           random_impostor_ratio=args.random_impostor_ratio,
                                           pairs_per_writer=args.pairs_per_writer,
                                           augment_params=augment_params,
                                           device_augment=args.device_augment,
                                           tilt_hide_prob=(TEMPORAL_TILT_HIDE_PROB
                                                           if args.temporal_channels == "dt_tilt_flag"
                                                           else 0.0))
    val_loader, num_val = build_loader(writer_samples, val_ids, args.branch, args.batch_size,
                                       args.num_workers, False, args.seed,
                                       augment=args.augment,
                                       pairs_per_writer=args.pairs_per_writer)
    negatives = ("skilled-forgery negatives" if not args.random_impostor_ratio else
                 f"{1 - args.random_impostor_ratio:.0%} skilled / "
                 f"{args.random_impostor_ratio:.0%} random-impostor negatives in train")
    print(f"  Pairs: {num_train} train / {num_val} val ({negatives})")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    if args.margin != CONTRASTIVE_MARGIN or args.hard_negative_ratio != HARD_NEGATIVE_RATIO:
        print(f"  Loss: margin {args.margin}, hard-negative ratio {args.hard_negative_ratio}")
    loss_fn = ContrastiveLoss(margin=args.margin,
                              hard_negative_ratio=args.hard_negative_ratio)
    optimizers = {"adam": optim.Adam, "adamw": optim.AdamW, "rmsprop": optim.RMSprop}
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr,
                                           weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    if args.optimizer != "adam" or args.min_delta != EARLY_STOPPING_MIN_DELTA:
        print(f"  Optimiser: {args.optimizer}, early-stopping min delta {args.min_delta:g}")
    start_epoch = 0
    resume_best = None
    resume_early_stopping = None
    resume_scheduler = None
    resume_rng = None

    if args.resume:
        # Every line that reports a resume used to live inside an `os.path.exists`
        # guard, so a typo'd path made the whole block a no-op: the run started from
        # epoch 0 with no message, discarded hours of progress, and could then
        # overwrite the very checkpoint it was meant to continue.
        if not os.path.exists(args.resume):
            raise FileNotFoundError(
                f"--resume {args.resume!r} does not exist. Refusing to silently "
                f"start a fresh run: that discards the checkpoint you meant to "
                f"continue, and the first save would overwrite it.")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # Prefer the run's best validation loss over the checkpointed epoch's
        # own loss. Older checkpoints predate the `best_val_loss` field and fall
        # back to `loss`, which re-baselines against the final epoch rather than
        # the best one — so a resumed run can still overwrite better weights.
        # When that matters, copy the current best weights aside first.
        resume_best = checkpoint.get("best_val_loss")
        if resume_best is None:
            resume_best = checkpoint.get("loss")
        resume_early_stopping = checkpoint.get("early_stopping")
        resume_scheduler = checkpoint.get("scheduler_state_dict")
        resume_rng = checkpoint.get("rng_state")
        if resume_rng is None:
            print("  Note: this checkpoint predates the saved random-number state, so the "
                  "resumed run will not match an uninterrupted one.")
        print(f"Resumed training from epoch {start_epoch}"
              + (f", best val loss {resume_best:.4f}" if resume_best is not None else ""))

    if args.init_weights:
        # Fine-tuning, not resuming: take the weights and nothing else. The
        # optimizer state and epoch counter belong to the pre-training dataset,
        # and carrying them over would start this run mid-schedule against data
        # the optimizer has never seen.
        model.load_state_dict(
            torch.load(args.init_weights, map_location=device, weights_only=True))
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
                  scheduler=scheduler, resume_best=resume_best,
                  resume_early_stopping=resume_early_stopping,
                  resume_scheduler=resume_scheduler, resume_rng=resume_rng)

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    print(f"Weights: {weights_path}")
    # Save the writer split record so evaluation can verify partition integrity.
    save_split_record(weights_path, train_ids, val_ids, test_ids,
                      data_dirs=args.data_dir, split_seed=args.split_seed,
                      holdout_writers=args.holdout_writers,
                      train_only_dirs=args.train_only_data_dir,
                      input_size=SPATIAL_INPUT_SIZE if args.branch == "spatial" else None,
                      max_gap_ms=MAX_GAP_MS if args.branch == "temporal" else None,
                      temporal_channels=args.temporal_channels if args.branch == "temporal" else None)
    print(f"Split record: {split_record_path(weights_path)}")
    return {**history,
            "split": {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}}


if __name__ == "__main__":
    main()

