"""SynSig2Vec training pipeline: synthesis, ranking loss, and verification.

Integrates kinematic synthesis (tools/research/sigma_lognormal.py) with ranking
representation learning (tools/research/sig2vec.py) for writer-independent verification.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn


from config.settings import (ENROLLMENT_TRIALS, NUM_ENROLLMENT_REFERENCES,
                             TEMPORAL_SPLIT_RATIOS)
from scripts.benchmark import dataset_of, separation_report
from scripts.evaluate import compute_eer, enrolled_trial_scores
from src.data.dataset import writer_independent_split
from src.data.parsers import merge_datasets, parse_dataset
from tools.research.sig2vec import (FEATURES, Sig2Vec, average_precision_loss,
                           template_similarity)
from tools.research.sigma_lognormal import decompose, distort, reconstruct


def _decompose_one(sample):
    """Worker: one signature to its stroke parameters. Top-level so it pickles."""
    return decompose(sample)


def build_parameter_cache(writer_samples, writer_ids, workers, cache_path):
    """
    Extract SigmaLambda parameters for every genuine signature, once.

    Cached because extraction dominates runtime by orders of magnitude and is
    deterministic — the same signature always yields the same strokes, so
    recomputing it every epoch would burn hours for nothing.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading SigmaLambda cache: {cache_path}")
        # `weights_only=False` is required and safe here: the cache holds pickled
        # SignatureParameters dataclasses, not tensors, and this process wrote it
        # itself under ~/.cache. It is not a distributed artifact.
        return torch.load(  # skylos: ignore[SKY-D265] self-written local dataclass cache
            cache_path, weights_only=False)

    flat, owner = [], []
    for writer in sorted(writer_ids):
        for sample in writer_samples[writer].get("genuine", []):
            flat.append(sample)
            owner.append(writer)

    print(f"Decomposing {len(flat)} genuine signatures on {workers} workers "
          f"(~10-18 s each, once)")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        params = list(pool.map(_decompose_one, flat, chunksize=4))

    cache = {}
    for writer, p in zip(owner, params):
        cache.setdefault(writer, []).append(p)

    if cache_path:
        torch.save(cache, cache_path)
        print(f"Cached to {cache_path}")
    return cache


def _to_tensor(trajectories, device, feature_fn, channels):
    """Time functions, padded into `[B, C, L]` with their true lengths."""
    feats = [feature_fn(t) for t in trajectories]
    lengths = torch.tensor([max(len(f), 1) for f in feats], dtype=torch.long)
    width = max(int(lengths.max()), 1)
    batch = torch.zeros(len(feats), channels, width)
    for i, f in enumerate(feats):
        if len(f):
            batch[i, :, :len(f)] = torch.tensor(f.T)
    return batch.to(device), lengths.to(device)


def _with_pressure(synthetic, source):
    """
    Give a synthetic trajectory its template's own pressure profile.

    SigmaLambda models the velocity of the pen, not how hard it pressed, so
    `reconstruct` returns `[N, 2]`. The 12-function input wants a pressure
    channel, and leaving it at zero would train the network on samples whose
    seventh channel is flat while every real signature it is later scored on has
    a live one — a train/test mismatch dressed up as a feature.

    `distort` preserves `params.length`, so the synthetic and its source are the
    same length and the profile transfers without resampling.
    """
    if source.shape[1] < 3 or len(synthetic) != len(source):
        return synthetic
    return np.hstack([synthetic[:, :2], source[:, 2:3]])


def save_weights(model, args, channels, path, epoch):
    """
    Single writer for both the periodic checkpoint and the final save.

    Written to a temporary file and renamed, because `torch.save` is not atomic
    and every checkpoint overwrites the same path: a crash partway through the
    write would leave a truncated file *and* destroy the last good checkpoint,
    which is exactly the failure this is here to survive. `os.replace` is atomic
    on POSIX.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # skylos: ignore[SKY-D215] operator-supplied --weights-dir
    tmp = path + ".tmp"
    torch.save({"state_dict": model.state_dict(), "features": args.features,
                "channels": channels, "seed": args.seed,
                "split_seed": args.split_seed, "epochs": args.epochs,
                "epoch": epoch}, tmp)
    os.replace(tmp, path)


def train(model, cache, writer_samples, feature_fn, channels, args, device):
    """
    Algorithm 1 of the paper: draw a genuine, synthesise G1 and G2, rank them.

    The classification head supplies the cross-entropy term that protects
    against *random* forgeries, while the AP term on the ranking head is what
    protects against skilled ones. Neither ever sees a real forgery.
    """
    writers = sorted(cache)
    index = {w: i for i, w in enumerate(writers)}
    classifier = nn.Linear(model.embedding_dim, len(writers)).to(device)
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    cross_entropy = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    model.train()
    starved = set()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(writers))
        losses = []
        for w in order:
            writer = writers[w]
            pool = cache[writer]
            if not pool:
                continue
            choice = int(rng.integers(len(pool)))
            params = pool[choice]
            if not params.strokes:
                continue

            # The cache was built by walking each writer's genuine list in order,
            # so index `choice` addresses the very signature these strokes came
            # from — that is where the pressure profile comes from.
            genuine = writer_samples[writer].get("genuine", [])
            source = genuine[choice] if choice < len(genuine) else None

            def _rebuild(p):
                out = reconstruct(p)
                return out if source is None else _with_pressure(out, source)

            template = _rebuild(params)
            g1 = [_rebuild(distort(params, 1, rng)) for _ in range(args.n_low)]
            if args.oracle_supervised:
                forged = writer_samples[writer].get("forged", [])
                if len(forged) < 2:
                    # Each writer gets exactly one step per epoch, and the
                    # forged count never changes, so this drops the writer from
                    # the run entirely. Report it — the whole point of the
                    # oracle is a like-for-like ceiling on a known population.
                    starved.add(writer)
                    continue
                picks = rng.choice(len(forged),
                                   size=min(args.n_median, len(forged)),
                                   replace=False)
                g2 = [forged[int(i)] for i in picks]
            else:
                g2 = [_rebuild(distort(params, 2, rng))
                      for _ in range(args.n_median)]

            batch, lengths = _to_tensor([template] + g1 + g2, device,
                                        feature_fn, channels)
            embeddings = model(batch, lengths)

            similarity = template_similarity(embeddings[0], embeddings[1:])
            loss = average_precision_loss(similarity[:len(g1)], similarity[len(g1):])
            # Every sample in this batch belongs to the same writer, synthetic or
            # not — that is exactly the invariance the classifier should learn.
            target = torch.full((len(embeddings),), index[writer],
                                dtype=torch.long, device=device)
            loss = loss + args.ce_weight * cross_entropy(classifier(embeddings), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())

        if losses:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}",
                  flush=True)
        # A 1000-epoch run is ~2.7 h. One that dies at epoch 90 with the weights
        # only written after the final eval leaves nothing behind — that has
        # already happened once.
        if args.save and args.checkpoint_every \
                and epoch % args.checkpoint_every == 0:
            save_weights(model, args, channels, args.save + ".ckpt", epoch)

    if starved:
        print(f"  WARNING: {len(starved)}/{len(writers)} writers never trained "
              f"(fewer than 2 forgeries): {', '.join(sorted(starved)[:5])}"
              f"{' ...' if len(starved) > 5 else ''}")
    return model


@torch.no_grad()
def embed_writers_sig2vec(model, writer_samples, writer_ids, device,
                          feature_fn, channels):
    """Embed every sample, L2-normalised, in the shape `enrolled_trial_scores` wants."""
    embedded = {}
    for writer in sorted(writer_ids):
        record = writer_samples[writer]
        out = {}
        for label in ("genuine", "forged"):
            samples = record.get(label, [])
            if not samples:
                out[label] = np.zeros((0, model.embedding_dim), dtype=np.float32)
                continue
            batch, lengths = _to_tensor(samples, device, feature_fn, channels)
            vectors = model(batch, lengths)
            # The verifier normalises before comparing (equation 23), so the
            # distance thresholded here is the one the loss ranked.
            out[label] = torch.nn.functional.normalize(vectors, dim=1).cpu().numpy()
        embedded[writer] = out
    return embedded


def evaluate(model, writer_samples, writer_ids, args, device, feature_fn,
             channels, quiet=False):
    """
    Score under the project's enrolment protocol, not the paper's.

    `enrolled_trial_scores` is reused rather than reimplemented so this number
    sits on the same axis as every other temporal EER in the repository — K
    references, nearest-reference distance, averaged over independent draws.
    """
    model.eval()
    embedded = embed_writers_sig2vec(model, writer_samples, writer_ids, device,
                                     feature_fn, channels)

    trials = []
    for trial in range(args.enroll_trials):
        scores, labels = enrolled_trial_scores(embedded, args.enroll,
                                               seed=args.seed + trial)
        if not len(scores) or labels.min() == labels.max():
            continue
        eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
        trials.append((eer, scores, labels))

    if not trials:
        return None
    trials.sort(key=lambda t: t[0])
    spread = np.array([t[0] for t in trials])
    _, scores, labels = trials[len(trials) // 2]
    if not quiet:
        print(f"  EER over {len(trials)} draws: mean {spread.mean() * 100:.2f}%  "
              f"std {spread.std() * 100:.2f}%")
    return separation_report(scores, labels)


def per_writer_eer(model, writer_samples, writer_ids, args, device, feature_fn,
                   channels):
    """
    User-threshold EER: the EER *inside* each writer, then averaged.

    This is the number the SVC2004 literature reports, and it is not the same
    measurement as the pooled EER used everywhere else in this repository. Pooled
    scoring puts every writer on one global threshold, so a writer whose
    embeddings sit far from the origin drags the whole operating point; per-writer
    scoring gives each signer their own threshold, which is what a deployed system
    with per-user enrolment actually has. The competition's own 2.89% was measured
    this way, so comparing our pooled figure against it compares two protocols and
    concludes the model is bad.

    Reported alongside pooled, never instead of it — pooled is the honest number
    for a system that must ship one threshold.
    """
    embedded = embed_writers_sig2vec(model, writer_samples, writer_ids, device,
                                     feature_fn, channels)
    per = []
    for writer in sorted(writer_ids):
        draws = []
        for trial in range(args.enroll_trials):
            scores, labels = enrolled_trial_scores(
                {writer: embedded[writer]}, args.enroll, seed=args.seed + trial)
            if not len(scores) or labels.min() == labels.max():
                continue
            eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
            draws.append(eer)
        if draws:
            per.append(float(np.mean(draws)))
    return per


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SynSig2Vec (clean-room) — tools-only experiment")
    p.add_argument("--data-dir", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Seed for the writer split only. Defaults to --seed.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--n-low", type=int, default=5, help="|G1| per step")
    p.add_argument("--n-median", type=int, default=10, help="|G2| per step")
    p.add_argument("--features", choices=sorted(FEATURES), default="velocity",
                   help="'velocity' is the AAAI paper's 3 channels (vx, vy, |v|); "
                        "'full' is the 12 time functions the same authors use in "
                        "their later work, pressure included")
    p.add_argument("--deepsign", action="store_true",
                   help="Merge DeepSignDB Development/stylus (498 writers). Weights "
                        "trained with this are DONT_SHIP under licence clause 4.4")
    p.add_argument("--save", type=str, default=None,
                   help="Persist the trained weights, so a new metric never "
                        "costs another training run")
    p.add_argument("--oracle-supervised", action="store_true",
                   help="DIAGNOSTIC ONLY. Train with real skilled forgeries as "
                        "negatives to establish the backbone's ceiling. Breaks the "
                        "forgery-free premise; never ship a model trained this way")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Write --save + '.ckpt' every N epochs. 0 disables it.")
    p.add_argument("--enroll", type=int, default=NUM_ENROLLMENT_REFERENCES)
    p.add_argument("--enroll-trials", type=int, default=ENROLLMENT_TRIALS)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--cache", type=str, default=None,
                   help="Path for the SigmaLambda parameter cache. Delete it "
                        "whenever the extractor changes, or it silently serves "
                        "strokes from the old one")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")

    writer_samples = {}
    for directory in args.data_dir:
        merge_datasets(writer_samples, parse_dataset(directory))
    if args.deepsign:
        from src.data.parsers import parse_deepsigndb
        merge_datasets(writer_samples, parse_deepsigndb())

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    args.split_seed = split_seed
    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(writer_samples), *TEMPORAL_SPLIT_RATIOS, seed=split_seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    cache = build_parameter_cache(writer_samples, train_ids, args.workers, args.cache)
    feature_fn, channels = FEATURES[args.features]
    model = Sig2Vec(in_channels=channels).to(device)
    print(f"Features: {args.features} ({channels} channels)")
    print(f"Sig2Vec parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, cache, writer_samples, feature_fn, channels, args, device)

    print("\nTEST — project enrolment protocol")
    report = evaluate(model, writer_samples, test_ids, args, device,
                      feature_fn, channels)
    if report:
        print(f"  pooled EER {report['eer'] * 100:.2f}%   "
              f"Cohen's d {report['cohens_d']:.3f}")
        groups = {}
        for writer in test_ids:
            groups.setdefault(dataset_of(writer), []).append(writer)
        if len(groups) > 1:
            print(f"  {'corpus':<18} {'wr':>3}   {'pooled':>8}  {'per-writer':>10}")
            for name, writers in sorted(groups.items()):
                sub = evaluate(model, writer_samples, writers, args, device,
                               feature_fn, channels, quiet=True)
                # Both protocols side by side: pooled is what this repo ships on,
                # per-writer is what the SVC2004 literature quotes.
                per = per_writer_eer(model, writer_samples, writers, args, device,
                                     feature_fn, channels)
                if sub:
                    user = f"{np.mean(per) * 100:>9.2f}%" if per else "        -"
                    print(f"  {name:<18} {len(writers):>3}   "
                          f"{sub['eer'] * 100:>7.2f}%  {user}")

    everyone = per_writer_eer(model, writer_samples, test_ids, args, device,
                              feature_fn, channels)
    if everyone:
        print(f"  ALL per-writer (user threshold): {np.mean(everyone) * 100:.2f}%  "
              f"median {np.median(everyone) * 100:.2f}%  "
              f"over {len(everyone)} writers")

    if args.save:
        save_weights(model, args, channels, args.save, args.epochs)
        print(f"  saved {args.save}")
    return report


if __name__ == "__main__":
    main()
