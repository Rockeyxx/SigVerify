"""
Evaluation Script — Signature Verification System

Standalone evaluation and calibration script. Imports from src.models
and runs analysis independently of the training pipeline.

Usage:
    python scripts/evaluate.py --branch spatial --weights models/weights/spatial_cnn.pth --data-dir ./data
    python scripts/evaluate.py --alpha-sweep --spatial-weights ... --temporal-weights ... --data-dir ./data
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, confusion_matrix
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    ALPHA_GRID_STEP, TARGET_FAR, TARGET_FRR,
    TEMPORAL_EMBEDDING_DIM,
    TEMPORAL_HIDDEN_SIZE, TEMPORAL_NUM_LAYERS,
    BATCH_SIZE, NUM_WORKERS, RESULTS_DIR,
    SPATIAL_SPLIT_RATIOS, TEMPORAL_SPLIT_RATIOS,
    ENROLLMENT_TRIALS, NUM_ENROLLMENT_REFERENCES,
)
from src.data.dataset import (
    SpatialSignatureDataset, TemporalSignatureDataset,
    collate_temporal_pairs, generate_skilled_pairs, preprocess_image,
    writer_independent_split,
)
from src.data.parsers import merge_datasets, parse_dataset
from src.models.siamese import SpatialSiamese, TemporalSiamese


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER) from genuine and impostor similarity scores.

    EER occurs at the operating threshold where False Acceptance Rate (FAR)
    equals False Rejection Rate (FRR).

    Returns: (eer: float, threshold_at_eer: float)
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Identify the threshold index where FPR and FNR intersect
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR) at a given threshold."""
    far = np.mean(impostor_scores >= threshold)
    frr = np.mean(genuine_scores < threshold)
    return float(far), float(frr)


def _finish_plot(fig, output_path, kind):
    """
    Save-and-close for the CLI path.

    The plot helpers below take an optional `ax`. When a caller supplies one,
    say the notebook drawing into its own subplot grid, the figure belongs to
    that caller, so it is neither saved nor closed here. That is the
    `fig is None` case, where this returns immediately.
    """
    if fig is None:
        return
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    tqdm.write(f"{kind} saved: {output_path}")


def plot_loss_curve(history, title="Training vs Validation Loss",
                    output_path=None, ax=None):
    """
    Plot per-epoch train and validation loss from a `train.fit` history dict.

    Marks the best-validation epoch, since that is the one whose weights were
    saved. Under early stopping the final epoch is rarely the best, and a curve
    that doesn't say so invites reading the wrong number off it.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(8, 5))

    # Markers make a short run readable and a long one an unreadable smear.
    marker = "o" if len(train_loss) <= 30 else None
    ax.plot(epochs, train_loss, label="Train Loss", marker=marker, linewidth=2)
    ax.plot(epochs, val_loss, label="Validation Loss", marker=marker,
            linewidth=2, linestyle="--")

    if val_loss:
        best = int(np.argmin(val_loss))
        ax.axvline(best + 1, color="green", linestyle=":", alpha=0.7)
        ax.scatter([best + 1], [val_loss[best]], color="green", s=60, zorder=5,
                   label=f"Best epoch {best + 1} (val {val_loss[best]:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish_plot(fig, output_path, "Loss curve")
    return ax


def plot_roc(genuine_scores, impostor_scores, title, output_path=None, ax=None):
    """
    Plot a Receiver Operating Characteristic (ROC) curve.

    Draws into `ax` if given (nothing is saved); otherwise creates its own
    figure and writes it to `output_path`.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finish_plot(fig, output_path, "ROC curve")
    return ax


def plot_det(genuine_scores, impostor_scores, title, output_path=None, ax=None):
    """
    Plot a Detection Error Tradeoff (DET) curve.

    Draws into `ax` if given (nothing is saved); otherwise creates its own
    figure and writes it to `output_path`.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, fnr, _ = det_curve(labels, scores)

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(8, 6))
    ax.plot(fpr * 100, fnr * 100, linewidth=2)
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finish_plot(fig, output_path, "DET curve")
    return ax


def plot_confusion_matrix(genuine_scores, impostor_scores, threshold, title,
                          output_path=None, ax=None):
    """
    Plot a confusion matrix at a calibrated decision threshold.

    Draws into `ax` if given (nothing is saved); otherwise creates its own
    figure and writes it to `output_path`.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Forged", "Genuine"])
    ax.set_yticklabels(["Forged", "Genuine"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

    ax.figure.colorbar(im, ax=ax)
    _finish_plot(fig, output_path, "Confusion matrix")
    return ax


def alpha_grid_search(spatial_genuine, spatial_impostor,
                      temporal_genuine, temporal_impostor,
                      step=ALPHA_GRID_STEP, output_path=None):
    """
    Perform grid search over spatial weight α ∈ [0.0, 1.0] to minimize validation EER.
    Returns: (best_alpha, best_eer, results: list of (alpha, eer))
    """
    results = []
    for alpha_int in range(0, int(1.0 / step) + 1):
        alpha = alpha_int * step
        fused_genuine = alpha * spatial_genuine + (1 - alpha) * temporal_genuine
        fused_impostor = alpha * spatial_impostor + (1 - alpha) * temporal_impostor
        eer, _ = compute_eer(fused_genuine, fused_impostor)
        results.append((alpha, eer))

    best_alpha, best_eer = min(results, key=lambda x: x[1])

    if output_path:
        alphas = [r[0] for r in results]
        eers = [r[1] for r in results]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, eers, 'o-', linewidth=2, markersize=4)
        ax.axvline(x=best_alpha, color='r', linestyle='--', label=f"Best α={best_alpha:.2f}")
        ax.set_xlabel("α (spatial weight)")
        ax.set_ylabel("EER")
        ax.set_title("α vs EER — Fusion Weight Calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"α sweep plot saved: {output_path}")

    return best_alpha, best_eer, results


def tri_point_report(spatial_eer, temporal_eer, fused_eer, fused_is_provisional=True):
    """Print tri-point EER report summarizing unimodal and multimodal branch metrics."""
    print("\n" + "=" * 50)
    print("TRI-POINT EER REPORT")
    print("=" * 50)
    print(f"  Spatial-only EER:   {spatial_eer*100:.2f}%  (target: <{TARGET_FAR*100:.0f}%)")
    print(f"  Temporal-only EER:  {temporal_eer*100:.2f}%  (target: <{TARGET_FRR*100:.0f}%)")
    label = "PROVISIONAL" if fused_is_provisional else "FINAL"
    print(f"  Fused EER:          {fused_eer*100:.2f}%  [{label}]")
    print("=" * 50)


def similarity_from_distance(distance):
    """
    Convert an L2 embedding distance into a bounded similarity score.

        S = 1 / (1 + d)

    Maps d ∈ [0, ∞) onto S ∈ (0, 1], so identical signatures approach 1 and
    wildly different ones approach 0. Every metric below consumes S, not d.
    """
    return 1.0 / (1.0 + distance)


def build_eval_loader(writer_samples, writer_ids, branch, batch_size, num_workers, seed):
    """Build a shuffle-free loader of skilled-forgery pairs over the given writers."""
    subset = {w: writer_samples[w] for w in writer_ids}
    pairs, labels = generate_skilled_pairs(subset, seed=seed)

    if branch == "spatial":
        dataset = SpatialSignatureDataset(pairs, labels)
        collate = None
    else:
        dataset = TemporalSignatureDataset(pairs, labels)
        collate = collate_temporal_pairs

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate)


def score_pairs(model, dataloader, device, desc="scoring"):
    """
    Run inference over a loader and return similarity scores in loader order.

    Order is only meaningful for shuffle-free loaders, which is every loader
    `build_eval_loader` hands out. Where it holds, the scores line up with
    `dataloader.dataset.pairs`, letting a caller show which specific pair got
    which prediction.

    Returns: (scores, labels) as numpy arrays.
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if len(batch) == 5:
                x1, lengths1, x2, lengths2, labels = batch
                emb1, emb2 = model(x1.to(device), x2.to(device), lengths1, lengths2)
            else:
                x1, x2, labels = batch
                emb1, emb2 = model(x1.to(device), x2.to(device))

            scores = similarity_from_distance(
                torch.nn.functional.pairwise_distance(emb1, emb2, p=2)
            ).cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

    if not all_scores:
        return np.array([]), np.array([])
    return np.concatenate(all_scores), np.concatenate(all_labels)


def embed_samples(model, samples, branch, device, batch_size=BATCH_SIZE):
    """
    Embed a flat list of samples — not pairs — returning [N, D] float32.

    The enrollment protocol scores every probe against every reference, so
    embedding each sample once and comparing in NumPy costs O(N) forward passes
    instead of the O(N x K) a pair loader would run.

    Spatial samples are image paths; temporal samples are [L, 3] trajectories,
    padded per batch and packed so a padded batch yields the same embedding the
    sample would produce alone.
    """
    if not samples:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)

    model.eval()
    out = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]

            if branch == "spatial":
                arrays = [s if isinstance(s, np.ndarray) else preprocess_image(s)
                          for s in chunk]
                batch = torch.tensor(np.stack(arrays), dtype=torch.float32).to(device)
                embeddings = model.encode(batch)
            else:
                seqs = [torch.tensor(np.asarray(s, dtype=np.float32)) for s in chunk]
                lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
                padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
                embeddings = model.encode(padded, lengths)

            out.append(embeddings.cpu().numpy())

    return np.vstack(out).astype(np.float32)


def embed_writers(model, writer_samples, writer_ids, branch, device, batch_size=BATCH_SIZE,
                  desc="embedding"):
    """
    Embed every genuine and forged sample for each writer, once.

    Reference draws are then re-sampled from these arrays rather than re-running
    the encoder, so averaging over ten enrollment trials costs one forward pass
    over the dataset instead of ten.

    Returns: {writer_id: {"genuine": [N, D], "forged": [M, D]}}
    """
    embedded = {}

    for writer in tqdm(sorted(writer_ids), desc=desc, leave=False):
        record = writer_samples[writer]
        embedded[writer] = {
            "genuine": embed_samples(model, record.get("genuine", []), branch,
                                     device, batch_size),
            "forged": embed_samples(model, record.get("forged", []), branch,
                                    device, batch_size),
        }

    return embedded


def enrolled_trial_scores(embedded, num_references=NUM_ENROLLMENT_REFERENCES,
                          seed=42):
    """
    Score one enrollment draw: K genuine references per writer, everything else a probe.

    This is the protocol the field uses and the one `src/verification.py` runs.
    SVC2004 enrolled exactly 5 genuine signatures per user and never showed a
    skilled forgery during enrollment — forgeries appear only as probes, which
    is why `references` is drawn from `genuine` alone.

    Scoring is nearest-reference with `S = 1 / (1 + d)`, identical to
    `src.verification._nearest_reference_score`. A person's signature varies, so
    the closest enrolled sample is the fairest comparison.

    A writer needs `num_references` + 1 genuines and at least one forgery, or it
    cannot contribute both arms and is skipped rather than half-counted.

    Returns: (scores, labels) as numpy arrays, genuine probes labelled 1.
    """
    rng = np.random.default_rng(seed)
    all_scores, all_labels = [], []

    for writer in sorted(embedded):
        genuine = embedded[writer]["genuine"]
        forged = embedded[writer]["forged"]

        if len(genuine) < num_references + 1 or not len(forged):
            continue

        order = rng.permutation(len(genuine))
        references = genuine[order[:num_references]]
        genuine_probes = genuine[order[num_references:]]

        for probes, label in ((genuine_probes, 1), (forged, 0)):
            if not len(probes):
                continue
            distances = np.linalg.norm(
                probes[:, None, :] - references[None, :, :], axis=2).min(axis=1)
            all_scores.append(1.0 / (1.0 + distances))
            all_labels.append(np.full(len(probes), label))

    if not all_scores:
        return np.array([]), np.array([])

    return np.concatenate(all_scores), np.concatenate(all_labels)


def fit_writer_classifier(references, negatives, C=1.0):
    """
    Fit one enrolled user's decision boundary.

    Positives are that user's enrolled genuine signatures. Negatives are genuine
    signatures belonging to **other** users — "random forgeries" in the
    literature's terminology.

    This is the stage published systems have and this project did not. On
    identical features SigNet-F reports 1.78% EER with a single global threshold
    against 0.4% with per-user boundaries, and every "99%+ on CEDAR" result is
    writer-dependent. One global distance threshold asks every signer to be
    equally consistent; a per-user boundary does not.

    Negatives deliberately exclude this user's own skilled forgeries. Those are
    the attack being measured, so fitting on them leaks the evaluation — and the
    deployed system never holds a labelled forgery of a real customer, so a
    result that needed them could not ship.

    `class_weight="balanced"` because negatives outnumber positives heavily:
    5 enrolled signatures against a pool drawn from every other writer.

    The classifier reads embeddings only. It never touches the Siamese encoder.
    """
    x = np.vstack([references, negatives])
    y = np.concatenate([np.ones(len(references)), np.zeros(len(negatives))])
    return LinearSVC(C=C, class_weight="balanced").fit(x, y)


def writer_dependent_scores(embedded, num_references=NUM_ENROLLMENT_REFERENCES,
                            seed=42, max_negatives=200):
    """
    Score every probe with its own writer's fitted classifier.

    Same enrolment draw as `enrolled_trial_scores` — K genuine references, the
    rest of the genuines plus every forgery as probes — but the decision comes
    from a per-user boundary instead of a distance to the nearest reference.

    Returns SVM decision-function margins rather than similarities. They are
    higher-is-genuine, which is all the downstream EER code requires, but they
    are **not** on the `S = 1/(1+d)` scale, so a threshold tuned against one is
    meaningless against the other.

    A writer needs `num_references` + 1 genuines, at least one forgery, and at
    least one other writer to draw negatives from, or it is skipped.
    """
    rng = np.random.default_rng(seed)
    writers = sorted(embedded)
    all_scores, all_labels = [], []

    for writer in writers:
        genuine = embedded[writer]["genuine"]
        forged = embedded[writer]["forged"]
        others = [w for w in writers if w != writer and len(embedded[w]["genuine"])]

        if len(genuine) < num_references + 1 or not len(forged) or not others:
            continue

        order = rng.permutation(len(genuine))
        references = genuine[order[:num_references]]
        genuine_probes = genuine[order[num_references:]]

        # Other writers' genuines only — never this writer's forgeries.
        pool = np.vstack([embedded[w]["genuine"] for w in others])
        if len(pool) > max_negatives:
            pool = pool[rng.choice(len(pool), max_negatives, replace=False)]

        classifier = fit_writer_classifier(references, pool)

        for probes, label in ((genuine_probes, 1), (forged, 0)):
            if not len(probes):
                continue
            all_scores.append(classifier.decision_function(probes))
            all_labels.append(np.full(len(probes), label))

    if not all_scores:
        return np.array([]), np.array([])

    return np.concatenate(all_scores), np.concatenate(all_labels)


def enrolled_writer_count(embedded, num_references=NUM_ENROLLMENT_REFERENCES):
    """How many writers can supply K references plus a probe of each class."""
    return sum(1 for w in embedded
               if len(embedded[w]["genuine"]) >= num_references + 1
               and len(embedded[w]["forged"]))


def score_loader(model, dataloader, device, desc="scoring"):
    """
    Run inference over a loader and split the similarity scores by true label.

    Returns: (genuine_scores, impostor_scores) as numpy arrays.
    """
    scores, labels = score_pairs(model, dataloader, device, desc=desc)
    return scores[labels == 1], scores[labels == 0]


def load_branch_model(branch, weights_path, device):
    """
    Instantiate the requested backbone and load its trained weights.

    The spatial architecture is read back out of the checkpoint rather than out
    of `config/`, so an archived run trained with a different trunk or embedding
    width still benchmarks correctly while config points at whatever is being
    trained now. Scoring a checkpoint under the wrong architecture would either
    raise a shape error or, worse, load a subset of the weights and report a
    number that means nothing.
    """
    state_dict = torch.load(weights_path, map_location="cpu")
    if branch == "spatial":
        model = SpatialSiamese.from_state_dict(state_dict)
    else:
        model = TemporalSiamese(
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=TEMPORAL_NUM_LAYERS,
            embedding_dim=TEMPORAL_EMBEDDING_DIM,
        )
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate signature verification models")
    parser.add_argument("--branch", choices=["spatial", "temporal", "fused"], default="spatial",
                        help="Which branch to evaluate")
    parser.add_argument("--weights", type=str, help="Path to model weights (.pth)")
    parser.add_argument("--spatial-weights", type=str, help="Spatial model weights for fused eval")
    parser.add_argument("--temporal-weights", type=str, help="Temporal model weights for fused eval")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Dataset directories — must match the training run")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                        help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed to reproduce the same split")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run α grid search")
    return parser.parse_args(argv)


def dataset_key(writer_id):
    """
    Group a namespaced writer ID back to its source writer_samples.

    A writer_samples that splits its writers by script or language keeps that middle
    token — `bhsig_Bengali`, `bhsig_Hindi`, `sigcomp_Dutch` — because the
    cross-script comparison is the interesting one; datasets with a bare numeric
    writer ID collapse to the dataset name.
    """
    parts = writer_id.split("_")
    if len(parts) > 2 and not parts[1].isdigit():
        return "_".join(parts[:2])
    return parts[0]


def per_dataset_report(branch, model, writer_samples, test_ids, threshold, args, device):
    """
    Report EER per source writer_samples at the validation-tuned threshold.

    A pooled EER can hide a dataset the model handles badly by averaging it
    against one it handles well. Training merges CEDAR (Latin), BHSig260
    (Bengali/Devanagari) and UTSig (Persian) into one model — this is the
    breakdown that shows whether it actually generalizes across scripts or
    just learned the majority script.
    """
    groups = {}
    for writer in test_ids:
        groups.setdefault(dataset_key(writer), []).append(writer)

    if len(groups) < 2:
        return {}

    print("\n" + "=" * 50)
    print(f"PER-DATASET BREAKDOWN — {branch} (τ={threshold:.4f} from validation)")
    print("=" * 50)

    breakdown = {}
    for name, writers in sorted(groups.items()):
        loader = build_eval_loader(writer_samples, writers, branch, args.batch_size,
                                   args.num_workers, args.seed)
        genuine, impostor = score_loader(model, loader, device, desc=f"{branch} · {name}")
        if len(genuine) == 0 or len(impostor) == 0:
            continue

        eer, _ = compute_eer(genuine, impostor)
        far, frr = compute_far_frr(genuine, impostor, threshold)
        breakdown[name] = {"eer": eer, "far": far, "frr": frr, "writers": len(writers)}
        print(f"  {name:<16} {len(writers):>3} writers   "
              f"EER {eer*100:>6.2f}%   FAR {far*100:>6.2f}%   FRR {frr*100:>6.2f}%")

    print("=" * 50)
    return breakdown


def evaluate_branch(branch, weights_path, writer_samples, val_ids, test_ids, args, device):
    """
    Score one branch: tune τ on validation, then report on the held-out test set.

    Returns (report dict, test_genuine, test_impostor).
    """
    model = load_branch_model(branch, weights_path, device)

    val_loader = build_eval_loader(writer_samples, val_ids, branch, args.batch_size,
                                   args.num_workers, args.seed)
    test_loader = build_eval_loader(writer_samples, test_ids, branch, args.batch_size,
                                    args.num_workers, args.seed)

    # τ is tuned on validation ONLY (AGENTS.md). Picking it on test would be
    # reporting the best-case threshold rather than an honest operating point.
    val_genuine, val_impostor = score_loader(model, val_loader, device,
                                             desc=f"{branch} · val (tune τ)")
    _, threshold = compute_eer(val_genuine, val_impostor)

    test_genuine, test_impostor = score_loader(model, test_loader, device,
                                               desc=f"{branch} · test")
    eer, _ = compute_eer(test_genuine, test_impostor)
    far, frr = compute_far_frr(test_genuine, test_impostor, threshold)

    plot_roc(test_genuine, test_impostor, f"ROC — {branch}",
             os.path.join(args.output_dir, f"roc_{branch}.png"))
    plot_det(test_genuine, test_impostor, f"DET — {branch}",
             os.path.join(args.output_dir, f"det_{branch}.png"))
    plot_confusion_matrix(test_genuine, test_impostor, threshold, f"Confusion — {branch}",
                          os.path.join(args.output_dir, f"confusion_{branch}.png"))

    report = {
        "branch": branch,
        "eer": eer,
        "far": far,
        "frr": frr,
        "threshold": threshold,
        "threshold_tuned_on": "validation",
        "reported_on": "test",
        "num_genuine": len(test_genuine),
        "num_impostor": len(test_impostor),
    }

    print(f"\n[{branch}]  EER {eer*100:.2f}%  |  at τ={threshold:.4f}: "
          f"FAR {far*100:.2f}%  FRR {frr*100:.2f}%")
    if far > TARGET_FAR:
        # AGENTS.md: when the FAR target is missed, keeping FAR low outranks FRR.
        print(f"  FAR target ({TARGET_FAR*100:.0f}%) MISSED — raise τ to trade FRR for FAR.")

    report["per_dataset"] = per_dataset_report(branch, model, writer_samples, test_ids,
                                               threshold, args, device)

    return report, test_genuine, test_impostor


def main(argv=None):
    # Headless backend is selected here, not at import time: importing this
    # module from a notebook must not clobber the interactive backend and kill
    # every inline plot. Only the CLI path runs without a display.
    matplotlib.use("Agg")

    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dynamic hardware selection for evaluation inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluation mode: {args.branch}")
    print(f"Hardware Device: {device}")
    print(f"Output directory: {args.output_dir}")

    writer_samples = {}
    for directory in args.data_dir:
        merge_datasets(writer_samples, parse_dataset(directory))

    # Re-derives the exact partition training used — same writer list, same seed.
    ratios = (TEMPORAL_SPLIT_RATIOS if args.branch == "temporal"
              else SPATIAL_SPLIT_RATIOS)
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(writer_samples), *ratios, seed=args.seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    if args.branch in ("spatial", "temporal"):
        weights = args.weights or (args.spatial_weights if args.branch == "spatial"
                                   else args.temporal_weights)
        if not weights:
            raise ValueError(f"--weights is required for --branch {args.branch}")
        report, _, _ = evaluate_branch(args.branch, weights, writer_samples,
                                       val_ids, test_ids, args, device)
        return report

    # ─── Fused ────────────────────────────────────────────────────────
    if not (args.spatial_weights and args.temporal_weights):
        raise ValueError("--branch fused requires --spatial-weights and --temporal-weights")

    spatial_report, spatial_genuine, spatial_impostor = evaluate_branch(
        "spatial", args.spatial_weights, writer_samples, val_ids, test_ids, args, device)
    temporal_report, temporal_genuine, temporal_impostor = evaluate_branch(
        "temporal", args.temporal_weights, writer_samples, val_ids, test_ids, args, device)

    # No writer has both an image and a trajectory yet, so the two score vectors
    # describe different pairs. Truncating to a common length lets α be swept,
    # but the resulting EER is an estimate — hence PROVISIONAL, per AGENTS.md.
    n_genuine = min(len(spatial_genuine), len(temporal_genuine))
    n_impostor = min(len(spatial_impostor), len(temporal_impostor))

    best_alpha, best_eer, _ = alpha_grid_search(
        spatial_genuine[:n_genuine], spatial_impostor[:n_impostor],
        temporal_genuine[:n_genuine], temporal_impostor[:n_impostor],
        output_path=(os.path.join(args.output_dir, "alpha_sweep.png")
                     if args.alpha_sweep else None),
    )

    tri_point_report(spatial_report["eer"], temporal_report["eer"], best_eer,
                     fused_is_provisional=True)

    return {
        "branch": "fused",
        "spatial": spatial_report,
        "temporal": temporal_report,
        "alpha": best_alpha,
        "eer": best_eer,
        "provisional": True,
        "threshold_tuned_on": "validation",
        "reported_on": "test",
    }


if __name__ == "__main__":
    main()

