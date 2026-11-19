"""
Evaluation & Calibration — Signature Verification System

Independent evaluation and threshold calibration script. Loads saved weights,
computes EER/FAR/FRR metrics, and generates ROC/DET curves and confusion matrices.

CLI Examples:
    python scripts/evaluate.py --branch spatial --weights models/weights/spatial_cnn.pth --data-dir ./data
    python scripts/evaluate.py --alpha-sweep --spatial-weights ... --temporal-weights ... --data-dir ./data
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, confusion_matrix
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm


from config.settings import (
    SEED,
    ALPHA_GRID_STEP, TARGET_ACCURACY, TARGET_FAR, TARGET_FRR,
    BATCH_SIZE, NUM_WORKERS, RESULTS_DIR,
    SPATIAL_SPLIT_RATIOS, TEMPORAL_SPLIT_RATIOS,
    ENROLLMENT_TRIALS, NUM_ENROLLMENT_REFERENCES,
    TEMPORAL_CHANNELS, TEMPORAL_CHANNEL_WIDTHS,
)

from src.data.parsers import MAX_GAP_MS
from src.data.dataset import (
    SpatialSignatureDataset, TemporalSignatureDataset,
    collate_temporal_pairs, generate_skilled_pairs, pin_holdout, preprocess_image,
    writer_independent_split,
    check_split_against_record)
from src.models.siamese import SpatialSiamese, TemporalSiamese
from src.verification import nearest_reference_distances, similarity_from_distance


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER) from genuine and forgery score distributions.

    Finds the decision threshold where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).

    Returns: (eer: float, threshold_at_eer: float)
    """

    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Compute minimal absolute difference index where FPR equals FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR) at specified decision threshold."""
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


def _draw(samples, indices):
    """
    Index a sample collection that may be a stacked matrix or a ragged list.

    Embeddings arrive as `[N, D]` and index directly. Raw trajectories do not —
    they are variable-length `[L, C]` arrays that cannot stack — so the
    enrollment draw has to work on a plain list too.
    """
    if isinstance(samples, np.ndarray):
        return samples[indices]
    return [samples[i] for i in indices]


def enrollment_draw(rng, num_genuine, num_references):
    """
    One enrollment draw: which genuine samples are references, which are probes.

    Returns `(reference_indices, probe_indices)`. The writer's forgeries are not
    involved — they are always probes, never references.

    This existed eight times across `scripts/` and `tools/` in two different
    idioms: `rng.permutation(n)` sliced in two here, and `rng.choice` plus
    `np.setdiff1d` in the tools. Both draw a uniformly random K-subset, so they
    were statistically equivalent but consumed the RNG stream differently, which
    meant "the same seed" named two different draws depending on which file you
    were reading. The permutation form is canonical because it is what the
    enrolment protocol in `src/verification.py` mirrors.
    """
    order = rng.permutation(num_genuine)
    return order[:num_references], order[num_references:]


def enrolled_trial_scores(embedded, num_references=NUM_ENROLLMENT_REFERENCES,
                          seed=42, distance=None):
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

    `distance` swaps the metric without touching the protocol. It defaults to
    None, meaning the vectorized L2 over stacked embeddings that every model
    path uses; passing a callable `(probe, reference) -> float` routes scoring
    through it instead. This exists so the DTW baseline
    (`tools/research/dtw_baseline.py`) can be measured on the same writers, the same K
    references and the same RNG sequence as the model it is compared against —
    a second copy of this loop is how two measurements drift apart. The
    callable path costs O(probes x references) Python calls, so it is for
    diagnostics, not for scoring a model.

    Returns: (scores, labels) as numpy arrays, genuine probes labelled 1.
    """
    rng = np.random.default_rng(seed)
    all_scores, all_labels = [], []

    for writer in sorted(embedded):
        genuine = embedded[writer]["genuine"]
        forged = embedded[writer]["forged"]

        if len(genuine) < num_references + 1 or not len(forged):
            continue

        ref_idx, probe_idx = enrollment_draw(rng, len(genuine), num_references)
        references = _draw(genuine, ref_idx)
        genuine_probes = _draw(genuine, probe_idx)

        for probes, label in ((genuine_probes, 1), (forged, 0)):
            if not len(probes):
                continue
            if distance is None:
                distances = nearest_reference_distances(probes, references)
            else:
                distances = np.array(
                    [min(distance(probe, reference) for reference in references)
                     for probe in probes])
            all_scores.append(similarity_from_distance(distances))
            all_labels.append(np.full(len(probes), label))

    if not all_scores:
        return np.array([]), np.array([])

    return np.concatenate(all_scores), np.concatenate(all_labels)


def znorm_trial_scores(embedded, num_references=NUM_ENROLLMENT_REFERENCES, seed=42):
    """
    Enrollment scoring with the probe distance standardized per writer.

    Identical draw to `enrolled_trial_scores` — same RNG, same K references, same
    probes — so the two are directly comparable. Only the metric differs.

    **The problem this addresses.** `S = 1/(1+d)` is compared against one global
    threshold for every writer in every corpus, which asks every signer to be
    equally self-consistent. They are not: this branch reads 4.24% EER on
    sigcomp_Dutch and 16.32% on svc2004, and a single threshold cannot sit well
    for both. A naturally variable signer's genuine probe can land further away
    than a precise signer's skilled forgery, and no threshold fixes that.

    **The fix.** The enrolled references already measure that writer's own scale,
    for free and without any extra capture: the pairwise distances of the K
    genuine references *to each other*. Standardizing the probe distance by their
    mean and spread is the Z-norm the biometrics literature has used for decades.

    This learns no decision boundary, only a scale — which is why the earlier
    per-user SVM result (EER 2-3x worse) does not transfer. That failed because
    random-forgery negatives cannot teach a skilled-forgery boundary; Z-norm
    never looks at a negative.

    Scores are **negated z-scores**, not `S = 1/(1+d)`: higher still means more
    genuine and EER is rank-based, so it is directly comparable, but a threshold
    tuned on one scoring does not carry to the other. Same caveat as
    `writer_dependent_scores`.

    Returns: (scores, labels) as numpy arrays, genuine probes labelled 1.
    """
    rng = np.random.default_rng(seed)
    all_scores, all_labels = [], []

    for writer in sorted(embedded):
        genuine = embedded[writer]["genuine"]
        forged = embedded[writer]["forged"]

        if len(genuine) < num_references + 1 or not len(forged):
            continue

        ref_idx, probe_idx = enrollment_draw(rng, len(genuine), num_references)
        references = _draw(genuine, ref_idx)
        genuine_probes = _draw(genuine, probe_idx)

        # The writer's own scale: how far their enrolled references sit from one
        # another. Upper triangle only — the diagonal is zero by construction and
        # would drag the mean toward it.
        intra = np.linalg.norm(references[:, None, :] - references[None, :, :], axis=2)
        upper = intra[np.triu_indices(len(references), k=1)]
        centre = float(upper.mean()) if upper.size else 0.0
        # A writer who signed identically twice has zero spread. Floor it rather
        # than clean up the inf afterwards.
        spread = max(float(upper.std()), 1e-6) if upper.size else 1.0

        for probes, label in ((genuine_probes, 1), (forged, 0)):
            if not len(probes):
                continue
            distances = nearest_reference_distances(probes, references)
            # Negated: the convention everywhere else is that a larger score
            # means more genuine, and a z-score runs the other way.
            all_scores.append(-(distances - centre) / spread)
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

        ref_idx, probe_idx = enrollment_draw(rng, len(genuine), num_references)
        references = genuine[ref_idx]
        genuine_probes = genuine[probe_idx]

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
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if branch == "spatial":
        model = SpatialSiamese.from_state_dict(state_dict)
    else:
        model = TemporalSiamese.from_state_dict(state_dict)
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
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Must match the training seed to reproduce the same split")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Must match the --split-ratios the weights were trained "
                             "with. The split is derived from the ratios, so a "
                             "mismatch scores the model on writers it trained on.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for the writer split only; defaults to --seed, "
                             "matching scripts/train.py. Training holds this fixed "
                             "while varying --seed to measure training variance on "
                             "one held-out set — which is how the six ensemble seeds "
                             "were produced. Scoring such a checkpoint without "
                             "repeating the value re-derives a different partition "
                             "and reports EER over writers the model trained on.")
    parser.add_argument("--holdout-writers", nargs="+", metavar="ID", default=None,
                        help="Must match the --holdout-writers the weights were "
                             "trained with. Pinned writers are forced into test; "
                             "omitting them here reshuffles the partition and can "
                             "move a training writer into the scored split.")
    parser.add_argument("--temporal-channels", default=TEMPORAL_CHANNELS,
                        choices=sorted(TEMPORAL_CHANNEL_WIDTHS),
                        help="Temporal channel mode: 'xyp' (3ch), 'dt' (4ch), or 'tilt' (5ch); "
                             "dt_tilt0 / dt_tilt1 / dt_tilt_flag (research only, see AGENTS.md)")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run α grid search")
    args = parser.parse_args(argv)
    if args.split_seed is None:
        args.split_seed = args.seed
    return args



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


def per_dataset_report(branch, model, writer_samples, test_ids, threshold, args, device,
                       embedded=None):
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
        # Same enrolment protocol τ was tuned under, so each dataset's FAR/FRR is
        # comparable to the pooled figure and to the deployed operating point.
        try:
            genuine, impostor = enrolled_split_scores(
                model, branch, writer_samples, writers, args, device, desc=name,
                embedded=embedded)
        except ValueError:
            continue
        if len(genuine) == 0 or len(impostor) == 0:
            continue

        eer, _ = compute_eer(genuine, impostor)
        far, frr = compute_far_frr(genuine, impostor, threshold)
        breakdown[name] = {"eer": eer, "far": far, "frr": frr, "writers": len(writers)}
        print(f"  {name:<16} {len(writers):>3} writers   "
              f"EER {eer*100:>6.2f}%   FAR {far*100:>6.2f}%   FRR {frr*100:>6.2f}%")

    print("=" * 50)
    return breakdown


def enrolled_split_scores(model, branch, writer_samples, writer_ids, args, device,
                          desc="split",
                          num_references=NUM_ENROLLMENT_REFERENCES,
                          trials=ENROLLMENT_TRIALS, embedded=None):
    """
    Scores for one writer set under the enrolment protocol.

    Which K genuines a writer happens to enrol moves the scores, so a single draw
    would fix τ against one lucky enrolment. This averages by taking the
    **median-EER** draw of `trials`, the same representative-draw rule
    `scripts/benchmark.enrolled_scores` uses.

    `embedded` (from embed_writers) is reused when given, so writers already run
    through the model are not run again.

    Returns (genuine_scores, impostor_scores).
    """
    if embedded is None:
        embedded = embed_writers(model, writer_samples, writer_ids, branch, device,
                                 args.batch_size, desc=f"{branch} · {desc}")
    else:
        embedded = {w: embedded[w] for w in writer_ids}

    # A writer needs K+1 genuines and a forgery to contribute both arms, so a
    # small validation split may not support the deployed K. Step K down rather
    # than falling back to pair-scoring: a lower K is still the right *protocol*
    # and its τ is close, whereas pair-scoring is the wrong protocol and its τ is
    # the bug this function exists to fix. Announce it, because a τ calibrated at
    # a K the app does not use is an approximation the reader should see.
    for k in range(num_references, 0, -1):
        draws = []
        for trial in range(trials):
            scores, labels = enrolled_trial_scores(embedded, k, seed=args.seed + trial)
            if not len(scores) or labels.min() == labels.max():
                continue
            eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
            draws.append((eer, scores, labels))

        if draws:
            if k != num_references:
                print(f"  Enrolled at K={k}, not {num_references}: this writer "
                      f"set has no writer with {num_references + 1} genuines and a forgery.")
            draws.sort(key=lambda d: d[0])
            _, scores, labels = draws[len(draws) // 2]
            return scores[labels == 1], scores[labels == 0]

    raise ValueError(
        "No writer in this set has 2 genuines and a forgery, so no enrolment is "
        "possible at any K. Widen the split.")


def evaluate_branch(branch, weights_path, writer_samples, val_ids, test_ids, args, device):
    """
    Score one branch: tune τ on validation, then report on the held-out test set.

    Returns (report dict, test_genuine, test_impostor).
    """
    model = load_branch_model(branch, weights_path, device)

    # τ is tuned on validation only. Picking it on test would report
    # an overly optimistic threshold rather than a realistic operating point.
    # It is tuned using the enrollment protocol to match production deployment.
    val_genuine, val_impostor = enrolled_split_scores(
        model, branch, writer_samples, val_ids, args, device, desc="val (tune τ)")
    _, threshold = compute_eer(val_genuine, val_impostor)

    # Test is scored under the same protocol τ was tuned on. Mixing them — an
    # enrolment τ applied to pair-scored test scores — would report a FAR/FRR
    # pair that describes no real operating point at all.
    test_embedded = embed_writers(model, writer_samples, test_ids, branch, device,
                                  args.batch_size, desc=f"{branch} · test")
    test_genuine, test_impostor = enrolled_split_scores(
        model, branch, writer_samples, test_ids, args, device, embedded=test_embedded)
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
    if frr > TARGET_FRR:
        print(f"  FRR target ({TARGET_FRR*100:.0f}%) MISSED — genuine users are being "
              f"rejected at this τ.")
    # Accuracy at the EER operating point is 1 - EER by definition, so this is a
    # restatement of the EER against the Term-1 requirement rather than a fourth
    # measurement. It is printed because the requirement is written in accuracy.
    if 1.0 - eer < TARGET_ACCURACY:
        print(f"  Accuracy target ({TARGET_ACCURACY*100:.0f}%) MISSED — "
              f"{(1.0 - eer)*100:.2f}% at the EER point.")

    report["per_dataset"] = per_dataset_report(branch, model, writer_samples, test_ids,
                                               threshold, args, device, embedded=test_embedded)

    return report, test_genuine, test_impostor


def main(argv=None):
    # Headless backend is selected here, not at import time: importing this
    # module from a notebook must not clobber the interactive backend and kill
    # every inline plot. Only the CLI path runs without a display.
    matplotlib.use("Agg")

    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)  # skylos: ignore[SKY-D215] operator-supplied --output-dir

    # Dynamic hardware selection for evaluation inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluation mode: {args.branch}")
    print(f"Hardware Device: {device}")
    print(f"Output directory: {args.output_dir}")

    from scripts.train import build_writer_samples
    channels = getattr(args, "temporal_channels", "xyp") if args.branch in ("temporal", "fused") else None
    writer_samples = build_writer_samples(
        args.data_dir, None if channels in (None, "xyp") else channels)

    # Re-derive the exact partition training used: same writer list, same seed,
    # same ratios, and same holdout set.
    ratios = tuple(args.split_ratios) if args.split_ratios else (
        TEMPORAL_SPLIT_RATIOS if args.branch == "temporal" else SPATIAL_SPLIT_RATIOS)
    if args.holdout_writers:
        train_ids, val_ids, test_ids = pin_holdout(
            sorted(writer_samples), args.holdout_writers, *ratios,
            seed=args.split_seed)
    else:
        train_ids, val_ids, test_ids = writer_independent_split(
            sorted(writer_samples), *ratios, seed=args.split_seed)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    # Ensure evaluation partition matches the checkpoint's split record.
    record = check_split_against_record(args.weights, train_ids, test_ids, max_gap_ms=MAX_GAP_MS)
    print("Split record: matches the checkpoint" if record else
          "Split record: none (checkpoint predates 2026-09-19; provenance unverified)")
    if record and record.get("input_size"):
        print(f"Input size: {record['input_size'][1]}x{record['input_size'][0]} (HxW), from the record")
    if record and record.get("max_gap_ms"):
        print(f"dt clip: {record['max_gap_ms']:g} ms, from the record")

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

