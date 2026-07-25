"""
EER Regression Harness — Signature Verification System

Judges a model on threshold-independent separation quality (EER, Cohen's d)
over held-out test writers, instead of FAR/FRR at a fixed decision threshold.
Reading FAR/FRR at a fixed tau is what masked the P-001 accuracy failure
(docs/PROBLEMS_FACED.md): a 98.58% FAR looked like a broken model, but the
real defect was tau=0.5 sitting exactly on the contrastive margin — a
calibration bug, not evidence about the embedding quality itself. EER and
Cohen's d slide the threshold out of the picture entirely, so this harness is
the regression gate for representation changes.

All distance/pairing/EER math is imported from scripts.evaluate — this module
only adds effect size and margin diagnostics on top of it.

Usage:
    python scripts/benchmark.py --branch spatial \\
        --weights models/weights/spatial_cnn.pth \\
        --data-dir data/cedar data/bhsig260 data/utsig --seed 42

    # Fast, reproducible run over a fixed-size sample of the test pairs:
    python scripts/benchmark.py --branch spatial \\
        --weights models/weights/spatial_cnn.pth \\
        --data-dir data/cedar data/bhsig260 data/utsig \\
        --seed 42 --limit-pairs 800
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    BATCH_SIZE, CONTRASTIVE_MARGIN, NUM_WORKERS,
    SPATIAL_SPLIT_RATIOS, TEMPORAL_SPLIT_RATIOS,
)
from scripts.evaluate import build_eval_loader, compute_eer, load_branch_model, score_pairs
from src.data.dataset import writer_independent_split
from src.data.parsers import parse_dataset


def cohens_d(impostor_distances, genuine_distances):
    """
    Effect size between the impostor and genuine distance distributions.

    d = (mean(impostor) - mean(genuine)) / pooled_std, pooled across BOTH arms
    (not just genuine) — a wide impostor arm should read as less separated
    even if its mean is just as far from genuine, because more of it overlaps.

    Returns NaN if either arm is empty (nothing to compare). Constant-valued
    arms give a zero pooled std; that is guarded against as a divide-by-zero,
    not surfaced as NaN, since the means may still be cleanly separated.
    """
    impostor = np.asarray(impostor_distances, dtype=float)
    genuine = np.asarray(genuine_distances, dtype=float)
    if impostor.size == 0 or genuine.size == 0:
        return float("nan")

    n1, n2 = impostor.size, genuine.size
    var1 = impostor.var(ddof=1) if n1 > 1 else 0.0
    var2 = genuine.var(ddof=1) if n2 > 1 else 0.0
    dof = n1 + n2 - 2
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / dof if dof > 0 else 0.0
    # Epsilon floor instead of a raw division: keeps the sign and magnitude
    # meaningful for near-constant arms rather than producing inf/nan.
    pooled_std = max(np.sqrt(pooled_var), 1e-12)

    return float((impostor.mean() - genuine.mean()) / pooled_std)


def separation_report(scores, labels, margin=CONTRASTIVE_MARGIN):
    """
    Bundle the threshold-independent separation diagnostics for one score set.

    `scores` are similarities S = 1/(1+d); this inverts that mapping (d = 1/S
    - 1) so genuine/impostor distances are directly comparable to `margin`,
    the same quantity the contrastive loss optimizes against.

    Returns a dict with: eer, threshold, cohens_d, genuine_mean_distance,
    impostor_mean_distance, impostors_inside_margin, num_genuine, num_impostor.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    genuine_d = 1.0 / genuine_scores - 1.0
    impostor_d = 1.0 / impostor_scores - 1.0

    # A degenerate (single-class) split can't define an ROC curve — sklearn
    # only warns and returns garbage in that case, so skip it explicitly
    # rather than let a downstream nanargmin blow up mid-benchmark.
    if genuine_scores.size and impostor_scores.size:
        eer, threshold = compute_eer(genuine_scores, impostor_scores)
    else:
        eer, threshold = float("nan"), float("nan")

    return {
        "eer": eer,
        "threshold": threshold,
        "cohens_d": cohens_d(impostor_d, genuine_d),
        "genuine_mean_distance": float(genuine_d.mean()) if genuine_d.size else float("nan"),
        "impostor_mean_distance": float(impostor_d.mean()) if impostor_d.size else float("nan"),
        "impostors_inside_margin": float(np.mean(impostor_d < margin)) if impostor_d.size else float("nan"),
        "num_genuine": int(genuine_scores.size),
        "num_impostor": int(impostor_scores.size),
    }


def _limit_pairs(loader, limit_pairs, seed):
    """
    Subsample a shuffle-free loader down to `limit_pairs` pairs.

    Lets the CLI reproduce a fixed measurement (e.g. 800 pairs) without
    scoring the full held-out split every time. Sampling uses a fixed-seed
    RNG so the subsample — and therefore the reported EER/Cohen's d — is
    reproducible run to run. Reuses `torch.utils.data.Subset` rather than
    touching the pairing logic in `src.data.dataset`.
    """
    if limit_pairs is None or limit_pairs >= len(loader.dataset):
        return loader

    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(loader.dataset), size=limit_pairs, replace=False))
    subset = Subset(loader.dataset, indices.tolist())
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False,
                      num_workers=loader.num_workers, collate_fn=loader.collate_fn)


def print_separation_report(branch, report):
    """Print the separation report in the tri-point-report style used elsewhere in the CLI."""
    print("\n" + "=" * 50)
    print(f"SEPARATION REPORT — {branch}")
    print("=" * 50)
    print(f"  EER:                      {report['eer']*100:.2f}%")
    print(f"  EER threshold (S):        {report['threshold']:.4f}")
    print(f"  Cohen's d:                {report['cohens_d']:.3f}")
    print(f"  Genuine mean distance:    {report['genuine_mean_distance']:.4f}")
    print(f"  Impostor mean distance:   {report['impostor_mean_distance']:.4f}")
    print(f"  Impostors inside margin:  {report['impostors_inside_margin']*100:.2f}%")
    print(f"  N genuine / impostor:     {report['num_genuine']} / {report['num_impostor']}")
    print("=" * 50)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="EER/Cohen's d separation-quality regression harness (held-out test writers)"
    )
    parser.add_argument("--branch", choices=["spatial", "temporal"], default="spatial",
                        help="Which branch to benchmark")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="Dataset directories — must match the training run")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed to reproduce the same writer "
                             "split and pairing; also seeds --limit-pairs subsampling")
    parser.add_argument("--margin", type=float, default=CONTRASTIVE_MARGIN,
                        help="Distance margin impostors are compared against")
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="Subsample the test pairs to this many for a fast, reproducible "
                             "run instead of scoring the full held-out split")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # GPU may be busy with a training run; CUDA_VISIBLE_DEVICES="" forces CPU
    # from outside without needing a flag here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark branch: {args.branch}")
    print(f"Hardware device: {device}")

    corpus = {}
    for directory in args.data_dir:
        corpus.update(parse_dataset(directory))

    # Re-derives the exact partition training used — same writer list, same seed.
    ratios = (TEMPORAL_SPLIT_RATIOS if args.branch == "temporal"
              else SPATIAL_SPLIT_RATIOS)
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(corpus), *ratios, seed=args.seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    model = load_branch_model(args.branch, args.weights, device)
    loader = build_eval_loader(corpus, test_ids, args.branch, args.batch_size,
                               args.num_workers, args.seed)
    loader = _limit_pairs(loader, args.limit_pairs, args.seed)
    print(f"Scoring {len(loader.dataset)} pairs from {len(test_ids)} held-out test writers")

    scores, labels = score_pairs(model, loader, device, desc=f"{args.branch} · benchmark")
    report = separation_report(scores, labels, margin=args.margin)
    print_separation_report(args.branch, report)
    return report


if __name__ == "__main__":
    main()
