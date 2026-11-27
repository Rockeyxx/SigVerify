"""Dynamic Time Warping (DTW) baseline for temporal trajectories."""

import argparse

import numpy as np
import torch


from config.settings import (ENROLLMENT_TRIALS, NUM_ENROLLMENT_REFERENCES,
                             TEMPORAL_SPLIT_RATIOS)
from scripts.benchmark import dataset_of, separation_report
from scripts.evaluate import compute_eer, enrolled_trial_scores
from src.data.dataset import writer_independent_split
from src.data.features import derive_time_functions
from src.data.parsers import merge_datasets, parse_dataset


def dtw_distance(a, b):
    """Length-normalized DTW Euclidean distance between two `[L, C]` sequences.

    Minimizes unnormalized DP cost with the standard three-way step, then
    divides by optimal path length so longer signatures do not score worse by size.
    Returns 0.0 for identical or purely time-warped sequences.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"DTW needs [L, C] sequences, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"channel counts differ: {a.shape[1]} against {b.shape[1]}. "
            "Both sequences must carry the same feature set."
        )

    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW is undefined for an empty sequence")

    # Precompute pairwise Euclidean costs across channels.
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    # Track accumulated cost and step count for path-length normalization.
    acc = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        row_steps, prev_steps = steps[i], steps[i - 1]
        cost_row = cost[i - 1]
        for j in range(1, m + 1):
            # Break ties toward the diagonal to keep the shortest path.
            diagonal, up, left = prev[j - 1], prev[j], row[j - 1]
            if diagonal <= up and diagonal <= left:
                best, best_steps = diagonal, prev_steps[j - 1]
            elif up <= left:
                best, best_steps = up, prev_steps[j]
            else:
                best, best_steps = left, row_steps[j - 1]
            row[j] = cost_row[j - 1] + best
            row_steps[j] = best_steps + 1

    return float(acc[n, m] / steps[n, m])


def _wavefront(cost, lengths_a, lengths_b):
    """Batched DTW computed one anti-diagonal (`i + j = k`) at a time.

    Each diagonal depends only on `k - 1` and `k - 2`, reducing sequential
    depth to `N + M` without masking because padded cells cannot feed earlier end cells.
    Returns `(total_cost [B], path_steps [B])`.
    """
    batch, n, m = cost.shape
    device = cost.device

    acc = torch.full((batch, n + 1, m + 1), float("inf"), device=device, dtype=cost.dtype)
    steps = torch.zeros((batch, n + 1, m + 1), device=device, dtype=cost.dtype)
    acc[:, 0, 0] = 0.0

    for k in range(2, n + m + 1):
        i = torch.arange(max(1, k - m), min(n, k - 1) + 1, device=device)
        j = k - i

        candidates = torch.stack([
            acc[:, i - 1, j - 1],   # match
            acc[:, i - 1, j],       # deletion
            acc[:, i, j - 1],       # insertion
        ])
        best, choice = candidates.min(dim=0)

        prior_steps = torch.stack([
            steps[:, i - 1, j - 1],
            steps[:, i - 1, j],
            steps[:, i, j - 1],
        ]).gather(0, choice.unsqueeze(0)).squeeze(0)

        acc[:, i, j] = cost[:, i - 1, j - 1] + best
        steps[:, i, j] = prior_steps + 1.0

    rows = torch.arange(batch, device=device)
    return acc[rows, lengths_a, lengths_b], steps[rows, lengths_a, lengths_b]


def _length_sorted_batches(pairs, cell_budget, batch_size):
    """Sort pairs by shape and split under `cell_budget` to bound padding and DP memory."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), len(pairs[i][1])))

    batch, max_a, max_b = [], 0, 0
    for index in order:
        a, b = pairs[index]
        next_a, next_b = max(max_a, len(a)), max(max_b, len(b))
        cells = (len(batch) + 1) * (next_a + 1) * (next_b + 1)
        if batch and (cells > cell_budget or len(batch) >= batch_size):
            yield batch, [pairs[i] for i in batch]
            batch, max_a, max_b = [], 0, 0
            next_a, next_b = len(a), len(b)
        batch.append(index)
        max_a, max_b = next_a, next_b

    if batch:
        yield batch, [pairs[i] for i in batch]


def dtw_distance_batch(pairs, device=None, batch_size=1024, cell_budget=12_000_000):
    """Batched `dtw_distance` over ragged `(a, b)` pairs on CUDA or CPU."""
    if not pairs:
        return np.zeros(0, dtype=np.float64)

    channels = np.asarray(pairs[0][0]).shape[1]
    for a, b in pairs:
        if np.asarray(a).shape[1] != channels or np.asarray(b).shape[1] != channels:
            raise ValueError(
                f"channel counts differ across the batch (expected {channels}). "
                "Every sequence must carry the same feature set."
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    out = np.empty(len(pairs), dtype=np.float64)

    for indices, chunk in _length_sorted_batches(pairs, cell_budget, batch_size):
        lengths_a = torch.tensor([len(a) for a, _ in chunk], device=device)
        lengths_b = torch.tensor([len(b) for _, b in chunk], device=device)
        max_a, max_b = int(lengths_a.max()), int(lengths_b.max())

        # Pad to batch max with zeros; padded cells do not affect any pair's endpoint.
        padded_a = torch.zeros((len(chunk), max_a, channels), device=device)
        padded_b = torch.zeros((len(chunk), max_b, channels), device=device)
        for index, (a, b) in enumerate(chunk):
            padded_a[index, :len(a)] = torch.as_tensor(np.asarray(a, dtype=np.float32))
            padded_b[index, :len(b)] = torch.as_tensor(np.asarray(b, dtype=np.float32))

        # Disable matrix-multiply expansion so identical frames stay at exact zero in float32.
        cost = torch.cdist(padded_a, padded_b,
                           compute_mode="donot_use_mm_for_euclid_dist")
        total, path_steps = _wavefront(cost, lengths_a, lengths_b)
        out[indices] = (total / path_steps).double().cpu().numpy()

    return out


def standardized_time_functions(trajectory):
    """Derive 16 time functions and z-score each channel per signature."""
    features = derive_time_functions(trajectory)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return ((features - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)


def trajectory_writers(writer_samples, writer_ids, transform=None):
    """Build `{writer: {"genuine": [...], "forged": [...]}}` directly from trajectories."""
    apply = transform if transform is not None else (lambda sample: sample)
    return {
        writer: {
            "genuine": [apply(s) for s in writer_samples[writer].get("genuine", [])],
            "forged": [apply(s) for s in writer_samples[writer].get("forged", [])],
        }
        for writer in sorted(writer_ids)
    }


def precomputed_distances(writers, device=None, batch_size=64):
    """Batch-compute all `(probe, reference)` DTW distances per writer and return a lookup."""
    pairs, keys = [], []
    for record in writers.values():
        genuine, forged = record["genuine"], record["forged"]
        for probe in list(genuine) + list(forged):
            for reference in genuine:
                pairs.append((probe, reference))
                keys.append((id(probe), id(reference)))

    distances = dtw_distance_batch(pairs, device=device, batch_size=batch_size)
    table = dict(zip(keys, distances))
    return lambda probe, reference: table[(id(probe), id(reference))]


def dtw_enrolled_eer(writers, num_references, trials, seed,
                     device=None, batch_size=64):
    """Run enrollment protocol over `trials` draws; return `(scores, labels, spread)`."""
    distance = precomputed_distances(writers, device=device, batch_size=batch_size)

    results = []
    for trial in range(trials):
        scores, labels = enrolled_trial_scores(writers, num_references,
                                               seed=seed + trial,
                                               distance=distance)
        if not len(scores) or labels.min() == labels.max():
            continue
        eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
        results.append((eer, scores, labels))

    if not results:
        return np.array([]), np.array([]), np.array([])

    results.sort(key=lambda r: r[0])
    spread = np.array([r[0] for r in results])
    _, scores, labels = results[len(results) // 2]
    return scores, labels, spread


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Learning-free DTW baseline under the enrollment protocol. "
                    "Diagnostic only — DTW is not shipped or fused (AGENTS.md)."
    )
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Temporal dataset directories — must match the "
                             "temporal training run whose EER this is compared against")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed to reproduce the same writer split")
    parser.add_argument("--enroll", type=int, default=NUM_ENROLLMENT_REFERENCES, metavar="K")
    parser.add_argument("--enroll-trials", type=int, default=ENROLLMENT_TRIALS, metavar="T")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Defaults to TEMPORAL_SPLIT_RATIOS, matching the branch")
    parser.add_argument("--features", choices=["raw", "derived"], default="raw",
                        help="raw: the (X, Y, P) the LSTM is fed today. derived: the "
                             "16 z-scored time functions from src/data/features.py. "
                             "The gap between these two answers whether the temporal "
                             "branch's problem is features or alignment")
    parser.add_argument("--device", default=None,
                        help="torch device for the DTW dynamic program. Defaults to "
                             "CUDA when present. This is a diagnostic, not inference, "
                             "so the no-GPU-at-inference rule does not apply")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Pairs per batch. Memory is O(batch x max_n x max_m); "
                             "turn this down on a small card")
    parser.add_argument("--writers", nargs="+", default=None,
                        help="Score only these writer IDs. DTW trains on nothing, so "
                             "there is no leakage to guard against here — this exists "
                             "to match a model run's held-out list exactly")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    writer_samples = {}
    for directory in args.data_dir:
        merge_datasets(writer_samples, parse_dataset(directory))

    # Derive the same held-out writer split used by the LSTM benchmark.
    ratios = tuple(args.split_ratios) if args.split_ratios else TEMPORAL_SPLIT_RATIOS
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(writer_samples), *ratios, seed=args.seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    if args.writers:
        test_ids = [w for w in args.writers if w in set(writer_samples)]
        print(f"Restricted to {len(test_ids)} explicitly requested writers")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = standardized_time_functions if args.features == "derived" else None
    print(f"Enrollment protocol: {args.enroll} references x {args.enroll_trials} draws, "
          f"nearest-reference DTW distance, features {args.features}, device {device}")

    groups = {}
    for writer in test_ids:
        groups.setdefault(dataset_of(writer), []).append(writer)

    report = {}
    for name, writers in sorted(groups.items()) + [("pooled", list(test_ids))]:
        scores, labels, spread = dtw_enrolled_eer(
            trajectory_writers(writer_samples, writers, transform=transform),
            args.enroll, args.enroll_trials, args.seed,
            device=device, batch_size=args.batch_size)
        if not len(scores):
            continue
        r = separation_report(scores, labels)
        report[name] = {**r, "spread_mean": float(spread.mean()),
                        "spread_std": float(spread.std())}
        print(f"  {name:<18} {len(writers):>3} writers   "
              f"EER {spread.mean() * 100:>6.2f}% +/- {spread.std() * 100:.2f}   "
              f"median-draw EER {r['eer'] * 100:>6.2f}%   Cohen's d {r['cohens_d']:>6.3f}")

    return report


if __name__ == "__main__":
    main()
