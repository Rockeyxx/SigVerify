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
    ENROLLMENT_TRIALS, NUM_ENROLLMENT_REFERENCES,
)
from tqdm.auto import tqdm

from scripts.evaluate import (build_eval_loader, compute_eer, embed_writers,
                              enrolled_trial_scores, enrolled_writer_count,
                              load_branch_model, score_pairs,
                              writer_dependent_scores)
from src.data.dataset import writer_independent_split
from src.data.parsers import merge_datasets, parse_dataset


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
    parser.add_argument("--enroll", type=int, nargs="?", const=NUM_ENROLLMENT_REFERENCES,
                        default=None, metavar="K",
                        help="Score under the enrollment protocol the field uses and "
                             "src/verification.py runs: K genuine references per writer, "
                             "probe matched against the nearest. SVC2004 used K=5. "
                             "Without this the harness scores isolated pairs, which is a "
                             "different and harder task")
    parser.add_argument("--enroll-trials", type=int, default=ENROLLMENT_TRIALS,
                        metavar="T",
                        help="Independent reference draws to average, reported as "
                             "mean +/- std. Which K signatures a user happens to enroll "
                             "moves the result, so a single draw is not the protocol")
    parser.add_argument("--wd-classifier", action="store_true",
                        help="Fit a per-user linear SVM on the enrolled references, "
                             "with other writers' genuines as negatives, instead of "
                             "thresholding a distance to the nearest reference. This "
                             "is the writer-dependent stage published systems use. "
                             "Requires --enroll. Scores are SVM margins, not "
                             "S = 1/(1+d), so thresholds do not carry across")
    parser.add_argument("--writers", nargs="+", default=None,
                        help="Score only these writer IDs instead of the whole held-out "
                             "split. For comparing two models trained on different dataset "
                             "lists, whose splits therefore differ: pass the writers both "
                             "of them held out. Every ID given must be outside this run's "
                             "own training set, or the run is rejected")
    args = parser.parse_args(argv)

    if args.wd_classifier and not args.enroll:
        parser.error("--wd-classifier requires --enroll: the per-user classifier is "
                     "fitted on the enrolled reference set, which single-pair scoring "
                     "does not have")

    return args


def select_test_writers(requested, train_ids, val_ids, test_ids):
    """
    Narrow the held-out split to an explicit writer list.

    Two temporal models trained on different dataset lists get different splits,
    because the partition is derived from whichever writers were loaded. Their
    reported EERs are then not comparable — SVC2004 scored 29.62% on one run's
    10 test writers and 16.92% on another's 13, and only 5 writers were common
    to both. This exists to score both models on those 5.

    A requested writer that this run trained on is a hard error, not a warning.
    Reporting a leaked number as a comparison is the failure being prevented.
    """
    requested = list(dict.fromkeys(requested))
    known = set(train_ids) | set(val_ids) | set(test_ids)

    unknown = [w for w in requested if w not in known]
    if unknown:
        raise ValueError(f"Unknown writer IDs, not present in this dataset: {unknown}")

    leaked = [w for w in requested if w in set(train_ids)]
    if leaked:
        raise ValueError(
            f"Refusing to score writers this run trained on: {leaked}. "
            f"Restrict --writers to writers held out by every model being compared."
        )

    return [w for w in requested if w in set(test_ids) or w in set(val_ids)]


def main(argv=None):
    args = parse_args(argv)

    # GPU may be busy with a training run; CUDA_VISIBLE_DEVICES="" forces CPU
    # from outside without needing a flag here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark branch: {args.branch}")
    print(f"Hardware device: {device}")

    writer_samples = {}
    for directory in args.data_dir:
        merge_datasets(writer_samples, parse_dataset(directory))

    # Re-derives the exact partition training used — same writer list, same seed.
    ratios = (TEMPORAL_SPLIT_RATIOS if args.branch == "temporal"
              else SPATIAL_SPLIT_RATIOS)
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(writer_samples), *ratios, seed=args.seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test writers")

    if args.writers:
        test_ids = select_test_writers(args.writers, train_ids, val_ids, test_ids)
        print(f"Restricted to {len(test_ids)} explicitly requested held-out writers")

    model = load_branch_model(args.branch, args.weights, device)

    if args.enroll:
        scores, labels, spread = enrolled_scores(args, writer_samples, test_ids, model, device)
    else:
        loader = build_eval_loader(writer_samples, test_ids, args.branch, args.batch_size,
                                   args.num_workers, args.seed)
        loader = _limit_pairs(loader, args.limit_pairs, args.seed)
        print(f"Scoring {len(loader.dataset)} pairs from {len(test_ids)} held-out test writers")
        scores, labels = score_pairs(model, loader, device, desc=f"{args.branch} · benchmark")
        spread = None
    report = separation_report(scores, labels, margin=args.margin)
    print_separation_report(args.branch, report)

    per_dataset = per_dataset_report(args, writer_samples, test_ids, model, device)
    return {**report, "per_dataset": per_dataset}


def enrolled_scores(args, writer_samples, writer_ids, model, device, desc="benchmark"):
    """
    Score `writer_ids` under the enrollment protocol, averaged over reference draws.

    Every sample is embedded once, then each trial re-draws which K genuines act
    as the enrolled references. That makes T trials cost one forward pass over
    the dataset rather than T, and it is the same averaging the SVC2004
    competition ran ("multiple sets of five genuine signatures each").

    The returned (scores, labels) come from the **median-EER** trial, so the
    downstream separation report describes a representative draw rather than a
    lucky one. `spread` carries the EER of every trial, because the spread is
    the interesting part: it says how much of a result is enrollment luck.

    Returns: (scores, labels, spread) where spread is a numpy array of per-trial EERs.
    """
    embedded = embed_writers(model, writer_samples, writer_ids, args.branch, device,
                             args.batch_size, desc=f"{args.branch} · {desc} · embedding")

    scoreable = enrolled_writer_count(embedded, args.enroll)
    decision = "per-user SVM" if args.wd_classifier else "nearest-reference distance"
    print(f"Enrollment protocol: {args.enroll} references x {args.enroll_trials} draws, "
          f"{decision}, {scoreable} of {len(writer_ids)} writers scoreable")

    scorer = writer_dependent_scores if args.wd_classifier else enrolled_trial_scores

    trials = []
    for trial in range(args.enroll_trials):
        scores, labels = scorer(embedded, args.enroll, seed=args.seed + trial)
        if not len(scores) or labels.min() == labels.max():
            continue
        eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
        trials.append((eer, scores, labels))

    if not trials:
        return np.array([]), np.array([]), np.array([])

    trials.sort(key=lambda t: t[0])
    spread = np.array([t[0] for t in trials])
    _, scores, labels = trials[len(trials) // 2]

    print(f"  EER over {len(trials)} draws: mean {spread.mean() * 100:.2f}%  "
          f"std {spread.std() * 100:.2f}%  "
          f"min {spread.min() * 100:.2f}%  max {spread.max() * 100:.2f}%")

    return scores, labels, spread


def dataset_of(writer_id):
    """
    Map a namespaced writer ID back to the dataset it came from.

    A writer_samples that splits its writers by script or language keeps that middle
    token — `bhsig_Bengali`, `sigcomp_Dutch` — because the cross-script gap is
    the interesting comparison and pooling it away is exactly the mistake the
    per-dataset breakdown exists to prevent. Datasets with a bare numeric writer
    ID (`svc_11`, `cedar_7`) collapse to the prefix.
    """
    parts = writer_id.split("_")
    if len(parts) > 2 and not parts[1].isdigit():
        return "_".join(parts[:2])
    return parts[0]


def per_dataset_report(args, writer_samples, test_ids, model, device):
    """
    Break the pooled EER down by source writer_samples.

    A single averaged number is the wrong summary when the goal is one model
    that works across capture devices and scripts: it hides a dataset the model
    handles badly behind one it handles well. On the spatial branch a pooled
    19% concealed CEDAR at 3.4% alongside UTSig at 36%.
    """
    groups = {}
    for writer in test_ids:
        groups.setdefault(dataset_of(writer), []).append(writer)

    if len(groups) < 2:
        return {}

    print(f"\n{'=' * 50}")
    print(f"PER-DATASET BREAKDOWN — {args.branch}")
    print("=" * 50)

    breakdown = {}
    for name, writers in sorted(groups.items()):
        if args.enroll:
            scores, labels, _ = enrolled_scores(args, writer_samples, writers, model, device,
                                                desc=name)
        else:
            loader = build_eval_loader(writer_samples, writers, args.branch, args.batch_size,
                                       args.num_workers, args.seed)
            loader = _limit_pairs(loader, args.limit_pairs, args.seed)
            scores, labels = score_pairs(model, loader, device,
                                         desc=f"{args.branch} · {name}")
        if not len(scores) or labels.min() == labels.max():
            continue

        r = separation_report(scores, labels, margin=args.margin)
        breakdown[name] = r
        tqdm.write(f"  {name:<16} {len(writers):>3} writers   "
                   f"EER {r['eer'] * 100:>6.2f}%   Cohen's d {r['cohens_d']:>6.3f}   "
                   f"{r['num_genuine'] + r['num_impostor']:>5} pairs")

    print("=" * 50)
    return breakdown


if __name__ == "__main__":
    main()
