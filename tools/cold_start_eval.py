"""
Cold-start evaluation: shipped weights against a corpus they never trained on.

Every EER in `SUMMARY.md` and `README.md` is measured on held-out *writers* of
corpora the model *did* train on. That is the standard writer-independent
protocol and it is not the same question as "what happens on a corpus from
somewhere else". This tool asks the second question, which is the one a
deployment actually faces.

Two things make it a deployment measurement rather than another benchmark:

* **It scores through `verify()`.** The shipped operating point is produced by
  the same call the GUI makes, one call per probe, with references round-tripped
  through `embedding_to_bytes`/`bytes_to_embedding` exactly as enrolment stores
  them. `benchmark.py` never touches that path -- `trajectory_gate` is reachable
  only from `verify()` -- so a serving defect is invisible to it. Three such
  defects shipped undetected (Part 9 §12).
* **It reports FAR/FRR/HTER, not EER.** The defect this tool measures is
  threshold-side, and EER is threshold-free: it *cannot* respond to a change in
  tau. Part 9 §3 records an arm that moved EER by exactly 0.00 by construction
  for this reason. Quoting EER for a threshold question is the mistake, not the
  answer.

Three operating points are reported, and only the last two are numbers a
deployment could honestly reach:

1. **Shipped tau** -- `config/settings.py` as it stands. What a user gets today.
2. **Domain-calibrated tau** -- fitted on half the unseen writers, applied
   unchanged to the other half. Mirrors `calibrated_fusion` in
   `tools/fusion_paired_eval.py`, for the reason given there: a constant chosen
   on the scores it is then quoted against is a lower bound on nothing.
3. **FAR-constrained tau** -- same writer-disjoint discipline, but tau is chosen
   to hold FAR at a target rather than to balance the two errors. This is the
   exchange rate: what FRR you pay to push FAR down.

No tau puts FAR and FRR both below the EER, because `max(FAR, FRR) >= EER` at
every threshold. "Both under 5%" is therefore a statement about separability,
not about where the threshold sits.

    python tools/cold_start_eval.py --branch spatial
    python tools/cold_start_eval.py --branch temporal --far-target 0.05
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MODEL_WEIGHTS_DIR,
    NUM_ENROLLMENT_REFERENCES,
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
                              load_model, verify)

# Which corpus is genuinely unseen by which shipped checkpoint. Both entries are
# claims about training provenance, so both cite where that provenance is
# recorded: a wrong entry turns this into an in-domain benchmark that reports
# suspiciously good numbers and reads as a success.
UNSEEN_CORPUS = {
    # WARNING (2026-09-06): the shipped spatial_cnn.pth NOW TRAINS ON
    # sigcomp2011_offline (SUMMARY.md Part 9 §13), so this default is training
    # data for it and scoring it here measures memorisation, not transfer. It
    # remains valid only for a pre-2026-09-06 checkpoint such as
    # `exp_ctrl.pth` or `archive_spatial_cnn_m20_8.59eer.pth`, passed via
    # --weights. For the shipped weight use `--corpus nfi2009`, which is
    # screened against every image corpus on disk.
    "spatial": "SigComp2011 offline (Dutch + Chinese scans) — TRAINING DATA for the shipped weight",
    # temporal_lstm.pth trained on SVC2004 Task 2 + SigComp2011 online, 62
    # writers. DeepSignDB was measured and NOT adopted (Part 9 §11), so its
    # Evaluation split -- the authors' own held-out benchmark -- is untouched.
    "temporal": "DeepSignDB Evaluation / stylus",
}


def parse_sign_data(root, allowed):
    """
    `sign_data/{split}/{writer}[_forg]/*.png` -- an ad-hoc downloaded corpus.

    Deliberately NOT registered in `src/data/parsers.py`. Perceptual hashing
    found 362 of its 2149 images to be exact duplicates of
    `data/sigcomp2011_offline`, so it is partly a redistribution and does not
    deserve the standing of a supported dataset. `allowed` carries the writers
    that survived that screen.

    Writer IDs are namespaced by split because 049-064 appear under both
    `train/` and `test/`. Merging them would pool two people's genuines into one
    reference set if the reused IDs are different signers, which would corrupt
    the enrolment the protocol depends on.
    """
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
    """
    SigComp2009 / NFI offline: `NFI-{author:03d}{session:02d}{target:03d}.png`.

    Genuine when the author signed their own name (author == target), a skilled
    forgery of `target` otherwise. Kept in `tools/` rather than
    `src/data/parsers.py` for the same reason as sign_data: it is an evaluation
    corpus pulled in to break a measurement deadlock, not a supported dataset.

    `allowed` is the writer list that survived a perceptual-hash screen against
    EVERY image corpus on disk -- not one partition of one corpus, which is how
    the previous screen let a trained-on corpus through.
    """
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
    """
    The unseen corpus for `branch`, keyed to what that checkpoint trained on.

    Writers are dropped unless they supply K references, at least one genuine
    probe *and* at least one forgery. A writer missing either arm contributes to
    only half the confusion matrix and biases whichever half it reaches, doing so
    invisibly because the probe totals still look large.
    """
    if corpus == "default" and branch == "spatial":
        # A comment cannot stop this; the default corpus is training data for the
        # shipped weight and would report memorisation as a cold-start score.
        raise ValueError(
            "sigcomp2011_offline is training data for the spatial weight shipped "
            "on 2026-09-06 (SUMMARY.md Part 9 §13), so the default corpus would "
            "measure memorisation.\n"
            "  Use --corpus nfi2009 (screened against every corpus on disk), or\n"
            "  pass a pre-2026-09-06 checkpoint via --weights, e.g.\n"
            "    --weights models/weights/archive_spatial_cnn_m20_8.59eer.pth")

    if corpus == "nfi2009":
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "data", "sigcomp2009_clean",
                               "clean_writers.txt")) as handle:
            allowed = {line.strip() for line in handle if line.strip()}
        writers = parse_nfi2009(
            SIGCOMP2009_DIR, allowed)
    elif corpus == "sign_data":
        listing = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "sign_data_clean", "clean_writers.txt")
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
    """
    Embed every sample once, after its input gate.

    The gate runs first because a probe it rejects never reaches the model in
    production either; counting a gate rejection as a model error misattributes
    the failure to the weights.

    Returns `{writer: {"genuine": (samples, [E1, E2, ...]), "forged": (...)}}`
    plus a rejection tally, where one embedding matrix is produced per model.
    A single model is the one-element case, so ensembling adds no second code
    path. Raw samples are kept beside the embeddings so the `verify()` pass can
    re-derive each score from the original input.
    """
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
    """
    One enrolment draw, kept broken out per writer.

    Pooling here would make the writer-disjoint split impossible: tau has to be
    fitted on one set of *writers* and applied to another, and a pooled array has
    forgotten which score came from whom.
    """
    rng = np.random.default_rng(seed)
    per = {}
    for writer in sorted(embedded):
        genuine_banks = embedded[writer]["genuine"][1]
        forged_banks = embedded[writer]["forged"][1]
        refs, probes = enrollment_draw(rng, len(genuine_banks[0]), num_references)

        def nearest(candidates, bank):
            distances = np.linalg.norm(candidates[:, None, :] - bank[None, :, :], axis=2)
            return similarity_from_distance(distances.min(axis=1))

        # Every model sees the SAME reference draw, so the averaged score is an
        # ensemble over one enrolment rather than an average over enrolment luck
        # as well -- otherwise the arm would flatter itself by also averaging out
        # the draw-to-draw variance the single-model arm still carries.
        genuine = np.mean([nearest(g[probes], g[refs])
                           for g in genuine_banks], axis=0)
        forged = np.mean([nearest(f, g[refs])
                          for f, g in zip(forged_banks, genuine_banks)], axis=0)
        per[writer] = (genuine, forged)
    return per


def pool(per, writers):
    """Concatenate the genuine and impostor arms across a set of writers."""
    return (np.concatenate([per[w][0] for w in writers]),
            np.concatenate([per[w][1] for w in writers]))


def threshold_for_far(impostor_scores, target_far):
    """
    Lowest tau whose FAR on *these* impostor scores is at most `target_far`.

    Fitted on calibration writers only. Applied to the writers it was fitted on
    it would be a tautology, which is why the caller never does that.
    """
    return float(np.quantile(impostor_scores, 1.0 - target_far))


def writer_disjoint(embedded, num_references, repeats, base_seed, far_target,
                    shipped_tau):
    """
    Fit tau on half the writers, score it on the other half, `repeats` times.

    All three taus are scored on the *same* test writers in each split, so they
    are paired readings of one experiment rather than three differently-drawn
    ones. `shipped_tau` is fitted nowhere -- it is the frozen constant from
    `config/settings.py`, carried through so the gain from recalibration is a
    paired difference rather than a comparison against a separate single draw.

    Returns a list of `(shipped, balanced, constrained)` (far, frr) triples.
    """
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
    """
    Score every probe through `verify()` -- the call the GUI makes.

    References make the full enrolment round trip (embed, serialise, read back)
    because that is what the database layer stores. A byte-layout error would
    surface here and nowhere in `benchmark.py`.

    Returns (scores, truths, accepted) as aligned arrays.

    One RNG is threaded across every writer. Seeding per writer instead would
    hand identical permutations to every writer holding the same number of
    genuines, so all of them would enrol the same *positions* -- and sample index
    tracks capture session in both corpora, which turns a random draw into one
    systematically chosen session pattern repeated corpus-wide.
    """
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

    # `verify()` takes exactly one model, so the deployed path cannot serve an
    # ensemble today. Reporting a single member's operating point as though it
    # were the ensemble's would be a straightforward lie, so it is skipped.
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

    # Paired on identical splits, so the difference is a within-split delta and
    # not two independently-drawn experiments differenced against each other.
    delta = (shipped[:, 0] + shipped[:, 1]) / 2 - (balanced[:, 0] + balanced[:, 1]) / 2
    print(f"\n  Recalibration buys {delta.mean() * 100:+.2f} HTER points "
          f"+/- {delta.std() * 100:.2f}, winning {int((delta > 0).sum())}/{len(delta)} splits.")


if __name__ == "__main__":
    main()
