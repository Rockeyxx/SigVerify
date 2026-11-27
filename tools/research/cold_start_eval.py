"""Cold-start evaluation on unseen corpora across shipped, balanced, and FAR-constrained thresholds."""
import argparse
import os
import re

import numpy as np


from config.settings import (
    MODEL_WEIGHTS_DIR,
    NUM_ENROLLMENT_REFERENCES,
    PROJECT_ROOT,
    SIGCOMP2009_DIR,
    SPATIAL_DECISION_THRESHOLD,
    TARGET_FAR,
    TEMPORAL_CHANNELS,
    TEMPORAL_DECISION_THRESHOLD,
)
from scripts.evaluate import (compute_eer, compute_far_frr, enrollment_draw,
                              similarity_from_distance)
from src.data.dataset import quality_gate, trajectory_gate
from src.data.parsers import parse_deepsigndb, parse_sigcomp2011_offline
from src.verification import (bytes_to_embedding, embed, embedding_to_bytes,
                              load_model, nearest_reference_distances, verify)

# Mapping of unseen evaluation corpora by modality branch.
UNSEEN_CORPUS = {
    "spatial": "SigComp2011 offline (Dutch + Chinese scans) — TRAINING DATA for the shipped weight",
    "temporal": "DeepSignDB Evaluation / stylus",
}


def parse_sign_data(root, allowed):
    """Parse screened sign_data folders, namespaced by split so reused IDs stay separate."""
    writers = {}
    for split in ("train", "test"):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            continue
        for folder in sorted(os.listdir(base)):
            stem = re.sub(r"_forg$", "", folder)
            if stem not in allowed:
                continue
            directory = os.path.join(base, folder)
            if not os.path.isdir(directory):
                continue
            label = "forged" if folder.endswith("_forg") else "genuine"
            writer_id = f"signdata_{split}_{stem}"
            entry = writers.setdefault(writer_id, {"genuine": [], "forged": []})
            entry[label].extend(
                os.path.join(directory, f) for f in sorted(os.listdir(directory))
                if f.lower().endswith((".png", ".jpg")))
    return writers


def parse_nfi2009(root, allowed):
    """Parse screened SigComp2009/NFI offline images into {writer: {genuine, forged}}."""
    pattern = re.compile(r"^NFI-(\d{3})(\d{2})(\d{3})\.png$", re.IGNORECASE)
    writers = {}
    for kind, folder in (("genuine", "genuines"), ("forged", "forgeries")):
        directory = os.path.join(root, folder)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            match = pattern.match(name)
            if not match:
                continue
            author, target = match.group(1), match.group(3)
            if (author == target) != (kind == "genuine"):
                continue
            writer = f"nfi_{target}"
            if writer not in allowed:
                continue
            writers.setdefault(writer, {"genuine": [], "forged": []})[kind].append(
                os.path.join(directory, name))
    return writers


def load_corpus(branch, corpus="default"):
    """Load unseen evaluation writers with >K genuines and >=1 forgery."""
    if corpus == "default" and branch == "spatial":
        # Default sigcomp2011_offline overlaps training data for the shipped spatial weight.
        raise ValueError(
            "sigcomp2011_offline is training data for the spatial weight shipped "
            "on 2026-09-06, so the default corpus would "
            "measure memorisation.\n"
            "  Use --corpus nfi2009 (screened against every corpus on disk), or\n"
            "  pass a pre-2026-09-06 checkpoint via --weights, e.g.\n"
            "    --weights models/weights/archive_spatial_cnn_m20_8.59eer.pth")

    if corpus == "nfi2009":
        with open(os.path.join(PROJECT_ROOT, "data", "sigcomp2009_clean",
                               "clean_writers.txt")) as handle:
            allowed = {line.strip() for line in handle if line.strip()}
        writers = parse_nfi2009(
            SIGCOMP2009_DIR, allowed)
    elif corpus == "sign_data":
        listing = os.path.join(PROJECT_ROOT, "data", "sign_data_clean", "clean_writers.txt")
        with open(listing) as handle:
            allowed = {line.strip() for line in handle if line.strip()}
        writers = parse_sign_data(
            os.path.join(os.path.dirname(listing), "..", "sign_data_dl", "sign_data"),
            allowed)
    elif branch == "spatial":
        writers = parse_sigcomp2011_offline()
    else:
        writers = parse_deepsigndb(
            split="Evaluation", modality="stylus",
            extra_channels=None if TEMPORAL_CHANNELS == "xyp" else TEMPORAL_CHANNELS)

    usable = {w: s for w, s in sorted(writers.items())
              if len(s["genuine"]) > NUM_ENROLLMENT_REFERENCES and s["forged"]}
    return usable, len(writers) - len(usable)


def embed_corpus(models, writers, branch):
    """Gate and embed all writer samples once across every model in `models`."""
    gate = quality_gate if branch == "spatial" else trajectory_gate
    embedded, rejected = {}, {"genuine": 0, "forged": 0}

    for writer, samples in writers.items():
        kept = {}
        for label in ("genuine", "forged"):
            raw = []
            for sample in samples[label]:
                passed, _ = gate(sample)
                if not passed:
                    rejected[label] += 1
                    continue
                raw.append(sample)
            kept[label] = (raw, [np.vstack([embed(m, s, branch) for s in raw])
                                 if raw else np.empty((0, 0)) for m in models])

        if len(kept["genuine"][0]) > NUM_ENROLLMENT_REFERENCES and kept["forged"][0]:
            embedded[writer] = kept

    return embedded, rejected


def scores_per_writer(embedded, num_references, seed):
    """Score one shared reference draw per writer, averaged across models."""
    rng = np.random.default_rng(seed)
    per = {}
    for writer in sorted(embedded):
        genuine_banks = embedded[writer]["genuine"][1]
        forged_banks = embedded[writer]["forged"][1]
        refs, probes = enrollment_draw(rng, len(genuine_banks[0]), num_references)

        def nearest(candidates, bank):
            return similarity_from_distance(nearest_reference_distances(candidates, bank))

        # Share one reference draw across ensemble members so draw variance does not shrink artificially.
        genuine = np.mean([nearest(g[probes], g[refs])
                           for g in genuine_banks], axis=0)
        forged = np.mean([nearest(f, g[refs])
                          for f, g in zip(forged_banks, genuine_banks)], axis=0)
        per[writer] = (genuine, forged)
    return per


def pool(per, writers):
    return (np.concatenate([per[w][0] for w in writers]),
            np.concatenate([per[w][1] for w in writers]))


def threshold_for_far(impostor_scores, target_far):
    return float(np.quantile(impostor_scores, 1.0 - target_far))


def writer_disjoint(embedded, num_references, repeats, base_seed, far_target,
                    shipped_tau):
    """Fit balanced and FAR-constrained taus on half the writers and score on the other half."""
    rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(base_seed + repeat)
        per = scores_per_writer(embedded, num_references, seed=base_seed + repeat)
        writers = sorted(per)
        if len(writers) < 4:
            raise ValueError(
                f"Only {len(writers)} scoreable writers -- a writer-disjoint split "
                "needs at least 4 to put two on each side. Widen the corpus.")

        order = rng.permutation(len(writers))
        half = len(writers) // 2
        fit = [writers[i] for i in order[:half]]
        test = [writers[i] for i in order[half:]]

        fit_genuine, fit_impostor = pool(per, fit)
        _, balanced_tau = compute_eer(fit_genuine, fit_impostor)
        far_tau = threshold_for_far(fit_impostor, far_target)

        test_genuine, test_impostor = pool(per, test)
        rows.append((compute_far_frr(test_genuine, test_impostor, shipped_tau),
                     compute_far_frr(test_genuine, test_impostor, balanced_tau),
                     compute_far_frr(test_genuine, test_impostor, far_tau)))
    return rows


def shipped_operating_point(model, embedded, branch, num_references, seed):
    """Score probes through verify() using serialized/deserialized embeddings as references."""
    rng = np.random.default_rng(seed)
    scores, truths, accepted = [], [], []
    for samples in embedded.values():
        refs, probes = enrollment_draw(rng, len(samples["genuine"][0]),
                                       num_references)
        stored = [bytes_to_embedding(embedding_to_bytes(samples["genuine"][1][0][i]))
                  for i in refs]

        bank = [(samples["genuine"][0][i], 1) for i in probes]
        bank += [(forgery, 0) for forgery in samples["forged"][0]]
        for sample, truth in bank:
            result = verify(**{branch: (model, stored, sample)})
            scores.append(result["confidence"] / 100.0)
            truths.append(truth)
            accepted.append(result["decision"] == "Genuine")

    return np.array(scores), np.array(truths), np.array(accepted)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cold-start evaluation of the "
                                                 "shipped weights on an unseen corpus")
    parser.add_argument("--branch", choices=("spatial", "temporal"), required=True)
    parser.add_argument("--weights", nargs="+", default=None,
                        help="One checkpoint, or several to score as a "
                             "score-averaged ensemble")
    parser.add_argument("--enroll", type=int, default=NUM_ENROLLMENT_REFERENCES)
    parser.add_argument("--repeats", type=int, default=25,
                        help="Writer-disjoint calibration splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", choices=("default", "sign_data", "nfi2009"),
                        default="default")
    parser.add_argument("--far-target", type=float, default=TARGET_FAR,
                        help="FAR the constrained operating point holds to")
    args = parser.parse_args(argv)

    weights = args.weights or [os.path.join(
        MODEL_WEIGHTS_DIR,
        "spatial_cnn.pth" if args.branch == "spatial" else "temporal_lstm.pth")]
    shipped_tau = (SPATIAL_DECISION_THRESHOLD if args.branch == "spatial"
                   else TEMPORAL_DECISION_THRESHOLD)

    print(f"Weights ..... {len(weights)} checkpoint(s)")
    for w in weights:
        print(f"              {os.path.basename(w)}")
    models = [load_model(args.branch, w) for w in weights]
    writers, dropped = load_corpus(args.branch, args.corpus)
    labels = {"sign_data": "sign_data (screened)",
              "nfi2009": "SigComp2009 / NFI offline (screened vs every corpus on disk)"}
    print(f"Corpus ...... {labels.get(args.corpus, UNSEEN_CORPUS[args.branch])}")
    print(f"              {len(writers)} usable writers, {dropped} dropped")

    embedded, rejected = embed_corpus(models, writers, args.branch)
    print(f"Embedded .... {sum(len(v['genuine'][0]) for v in embedded.values())} genuine"
          f" + {sum(len(v['forged'][0]) for v in embedded.values())} skilled forgeries")
    if sum(rejected.values()):
        print(f"Input gate .. rejected {rejected['genuine']} genuine, "
              f"{rejected['forged']} forged before scoring")

    # verify() accepts a single model only; skip when evaluating an ensemble.
    if len(models) == 1:
        scores, truths, accepted = shipped_operating_point(
            models[0], embedded, args.branch, args.enroll, args.seed)
        far = float(np.mean(accepted[truths == 0]))
        frr = float(np.mean(~accepted[truths == 1]))
        eer, _ = compute_eer(scores[truths == 1], scores[truths == 0])
        print(f"\nSHIPPED OPERATING POINT -- {len(scores)} verify() calls, "
              f"{int(np.sum(truths == 1))} genuine / {int(np.sum(truths == 0))} forged")
        print(f"  shipped tau {shipped_tau:<22.4f} FAR {far * 100:>6.2f}%   "
              f"FRR {frr * 100:>6.2f}%   HTER {(far + frr) / 2 * 100:>6.2f}%")
        print(f"  EER (threshold-free floor)         {eer * 100:>6.2f}%")
    else:
        print(f"\nSHIPPED OPERATING POINT -- skipped: verify() serves one model, "
              f"not a {len(models)}-member ensemble")

    print(f"\nWRITER-DISJOINT RECALIBRATION -- fit tau on half the writers, "
          f"score the other half, {args.repeats} splits")
    rows = writer_disjoint(embedded, args.enroll, args.repeats, args.seed,
                           args.far_target, shipped_tau)
    shipped = np.array([r[0] for r in rows])
    balanced = np.array([r[1] for r in rows])
    constrained = np.array([r[2] for r in rows])

    for label, arm in (("shipped tau (frozen constant)", shipped),
                       ("domain-calibrated tau", balanced),
                       (f"FAR-constrained tau (<={args.far_target * 100:.0f}%)",
                        constrained)):
        hter = (arm[:, 0] + arm[:, 1]) / 2
        print(f"  {label:<34} FAR {arm[:, 0].mean() * 100:>6.2f}% +/- {arm[:, 0].std() * 100:<5.2f} "
              f"FRR {arm[:, 1].mean() * 100:>6.2f}% +/- {arm[:, 1].std() * 100:<5.2f} "
              f"HTER {hter.mean() * 100:>6.2f}% +/- {hter.std() * 100:.2f}")

    # Paired within-split delta across identical writer partitions.
    delta = (shipped[:, 0] + shipped[:, 1]) / 2 - (balanced[:, 0] + balanced[:, 1]) / 2
    print(f"\n  Recalibration buys {delta.mean() * 100:+.2f} HTER points "
          f"+/- {delta.std() * 100:.2f}, winning {int((delta > 0).sum())}/{len(delta)} splits.")


if __name__ == "__main__":
    main()
