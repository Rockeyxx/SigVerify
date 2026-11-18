"""Late-fusion evaluation on paired multimodal signatures.

Evaluates multimodal fusion performance on signatures where both scanned images
and dynamic trajectory data are available for the same signing events.
"""
import argparse
import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import (alpha_grid_search, compute_eer, compute_far_frr,
                              embed_samples,
                              load_branch_model, similarity_from_distance, enrollment_draw,)
from config.settings import (MODEL_WEIGHTS_DIR, SIGCOMP2011_DIR, SIGCOMP2011_OFFLINE_DIR,
                             TEMPORAL_CHANNELS)
from src.data.parsers import load_sigcomp2011_file
from src.verification import nearest_reference_distances

_OFF = os.path.join(SIGCOMP2011_OFFLINE_DIR, "trainingSet", "OfflineSignatures")
_ON = os.path.join(SIGCOMP2011_DIR, "trainingSet", "OnlineSignatures")


def _decode(fname):
    """(target_writer, sample) for genuine `001_07.x` and both forgery conventions."""
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) == 2 and re.fullmatch(r"\d{3}", parts[0]):
        return int(parts[0]), int(parts[1])          # 001_07 genuine
    if len(parts) == 3:
        return int(parts[1]), int(parts[2])          # 170_001_02 Dutch forgery
    if len(parts) == 2 and len(parts[0]) >= 7:
        return int(parts[0][-3:]), int(parts[1])     # 0102014_01 glued forgery
    return None


def _index(directory, exts):
    """{(writer, sample): path} for one directory."""
    out = {}
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(exts):
            key = _decode(fname)
            if key:
                out[key] = os.path.join(directory, fname)
    return out


def build_paired_index(scripts=("Dutch", "Chinese")):
    """
    {writer_id: {"genuine": [(img, traj)], "forged": [(img, traj)]}}

    A sample is kept only when BOTH modalities exist for that exact index, so
    every retained probe is one real signing event seen two ways.
    """
    paired, stats = {}, collections.Counter()
    for script in scripts:
        for label, off_dir, on_dir in (
            ("genuine", "Offline Genuine", "Online Genuine"),
            ("forged", "Offline Forgeries", "Online Forgeries"),
        ):
            off = _index(f"{_OFF}/{script}/TrainingSet/{off_dir}", (".png", ".jpg"))
            on = _index(f"{_ON}/{script}/TrainingSet/{on_dir}", (".hwr", ".txt"))
            shared = sorted(set(off) & set(on))
            stats[f"{script}_{label}_offline"] += len(off)
            stats[f"{script}_{label}_paired"] += len(shared)
            for writer, sample in shared:
                wid = f"sigcomp_paired_{script}_{writer:03d}"
                paired.setdefault(wid, {"genuine": [], "forged": []})
                paired[wid][label].append((off[(writer, sample)],
                                           on[(writer, sample)]))
    return paired, stats


def load_paired_trajectory(path):
    """
    One online sample at the width the shipped temporal model expects.

    `load_sigcomp2011_file` defaults to the legacy 3-channel `xyp`. Calling it
    bare here crashed against the 4-channel `dt` weights and, before that, let
    this tool report a fusion number for a model other than the one shipped.
    `TEMPORAL_CHANNELS` is the single knob; nothing here may name a width.
    """
    extra = None if TEMPORAL_CHANNELS == "xyp" else TEMPORAL_CHANNELS
    return load_sigcomp2011_file(path, extra_channels=extra)


def _embed_both(paired, spatial_model, temporal_model, device):
    """Embed every paired sample once per branch, preserving list order."""
    embedded = {}
    for wid, record in sorted(paired.items()):
        entry = {}
        for label in ("genuine", "forged"):
            items = record[label]
            if not items:
                entry[label] = (np.zeros((0, 1), np.float32),
                                np.zeros((0, 1), np.float32))
                continue
            images = [p for p, _ in items]
            trajectories = [load_paired_trajectory(t) for _, t in items]
            entry[label] = (
                embed_samples(spatial_model, images, "spatial", device),
                embed_samples(temporal_model, trajectories, "temporal", device),
            )
        embedded[wid] = entry
    return embedded


def paired_scores_per_writer(embedded, num_references=5, seed=42):
    """Same protocol as `paired_scores`, kept per writer so splits stay writer-disjoint."""
    rng = np.random.default_rng(seed)
    out = {}
    for wid in sorted(embedded):
        gen_sp, gen_tp = embedded[wid]["genuine"]
        forg_sp, forg_tp = embedded[wid]["forged"]
        if len(gen_sp) < num_references + 1 or len(forg_sp) == 0:
            continue
        refs, probes = enrollment_draw(rng, len(gen_sp), num_references)

        def nearest(probe_bank, ref_bank, idx):
            return similarity_from_distance(
                nearest_reference_distances(probe_bank[idx], ref_bank[refs]))

        out[wid] = (nearest(gen_sp, gen_sp, probes),
                    nearest(forg_sp, gen_sp, np.arange(len(forg_sp))),
                    nearest(gen_tp, gen_tp, probes),
                    nearest(forg_tp, gen_tp, np.arange(len(forg_tp))))
    return out


def _pool(per_writer, writers):
    """Concatenate the four score arrays over a writer subset."""
    return [np.concatenate([per_writer[w][i] for w in writers]) for i in range(4)]


def calibrated_fusion(embedded, num_references=5, repeats=25, base_seed=42):
    """
    Fit alpha on one set of writers, score it on a disjoint set.

    `alpha_grid_search` selects the minimum of the very curve it reports, so its
    EER is optimistically biased and cannot be quoted as a result. Here alpha is
    chosen on calibration writers only and then applied, unchanged, to writers it
    never saw. That is the number that survives review.
    """
    rows = []
    for r in range(repeats):
        rng = np.random.default_rng(base_seed + r)
        per = paired_scores_per_writer(embedded, num_references, seed=base_seed + r)
        writers = sorted(per)
        if len(writers) < 4:
            return []
        order = rng.permutation(len(writers))
        half = len(writers) // 2
        fit = [writers[i] for i in order[:half]]
        test = [writers[i] for i in order[half:]]

        fsg, fsi, ftg, fti = _pool(per, fit)
        alpha, _, _ = alpha_grid_search(fsg, fsi, ftg, fti)

        tsg, tsi, ttg, tti = _pool(per, test)
        sp_eer, _ = compute_eer(tsg, tsi)
        tp_eer, _ = compute_eer(ttg, tti)
        fit_eer, _ = compute_eer(alpha * tsg + (1 - alpha) * ttg,
                                 alpha * tsi + (1 - alpha) * tti)
        half_eer, half_thr = compute_eer(0.5 * tsg + 0.5 * ttg, 0.5 * tsi + 0.5 * tti)
        far, frr = compute_far_frr(0.5 * tsg + 0.5 * ttg, 0.5 * tsi + 0.5 * tti, half_thr)
        rows.append((alpha, sp_eer, tp_eer, fit_eer, half_eer, far, frr))
    return rows


def paired_scores(embedded, num_references=5, seed=42):
    """
    Enrollment protocol run once, with the SAME reference draw for both branches.

    That shared draw is the whole point: it makes `spatial[i]` and
    `temporal[i]` two views of one probe, so they can be fused and their errors
    correlated. Returns four aligned arrays.
    """
    rng = np.random.default_rng(seed)
    sg, tg, si, ti = [], [], [], []

    for wid in sorted(embedded):
        gen_sp, gen_tp = embedded[wid]["genuine"]
        forg_sp, forg_tp = embedded[wid]["forged"]
        if len(gen_sp) < num_references + 1 or len(forg_sp) == 0:
            continue

        refs, probes = enrollment_draw(rng, len(gen_sp), num_references)

        def nearest(probe_bank, ref_bank, idx):
            return similarity_from_distance(
                nearest_reference_distances(probe_bank[idx], ref_bank[refs]))

        sg.append(nearest(gen_sp, gen_sp, probes))
        tg.append(nearest(gen_tp, gen_tp, probes))
        si.append(nearest(forg_sp, gen_sp, np.arange(len(forg_sp))))
        ti.append(nearest(forg_tp, gen_tp, np.arange(len(forg_tp))))

    return (np.concatenate(sg), np.concatenate(si),
            np.concatenate(tg), np.concatenate(ti))


def error_correlation(sp_gen, sp_imp, tp_gen, tp_imp):
    """
    Do the two branches fail on the *same* probes?

    Each branch is thresholded at its own EER point, giving a per-probe error
    indicator. The phi coefficient between those indicators bounds what fusion
    can deliver: near 1 means the branches are redundant and no alpha helps;
    near 0 (or negative) means they fail independently and fusion has room.
    """
    out = {}
    _, sp_thr = compute_eer(sp_gen, sp_imp)
    _, tp_thr = compute_eer(tp_gen, tp_imp)
    sp_err = np.concatenate([sp_gen < sp_thr, sp_imp >= sp_thr]).astype(float)
    tp_err = np.concatenate([tp_gen < tp_thr, tp_imp >= tp_thr]).astype(float)
    out["spatial_error_rate"] = float(sp_err.mean())
    out["temporal_error_rate"] = float(tp_err.mean())
    out["both_wrong"] = float(((sp_err == 1) & (tp_err == 1)).mean())
    out["either_wrong"] = float(((sp_err == 1) | (tp_err == 1)).mean())
    if sp_err.std() > 0 and tp_err.std() > 0:
        out["phi"] = float(np.corrcoef(sp_err, tp_err)[0, 1])
    else:
        out["phi"] = float("nan")
    # If the branches were independent, both would fail this often:
    out["both_wrong_if_independent"] = out["spatial_error_rate"] * out["temporal_error_rate"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spatial-weights", default=os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth"))
    p.add_argument("--temporal-weights",
                   default=os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm_mobisig_svc.pth"))
    p.add_argument("--enroll", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    paired, stats = build_paired_index()
    print("Paired index")
    for k in sorted(stats):
        print(f"  {k:32s} {stats[k]}")
    print(f"  writers paired                   {len(paired)}")

    import torch
    device = torch.device(args.device)
    spatial = load_branch_model("spatial", args.spatial_weights, device)
    temporal = load_branch_model("temporal", args.temporal_weights, device)
    print(f"\nspatial  {args.spatial_weights}")
    print(f"temporal {args.temporal_weights}")

    embedded = _embed_both(paired, spatial, temporal, device)

    rows = []
    for seed in args.seeds:
        sg, si, tg, ti = paired_scores(embedded, args.enroll, seed)
        sp_eer, _ = compute_eer(sg, si)
        tp_eer, _ = compute_eer(tg, ti)
        best_alpha, best_eer, _ = alpha_grid_search(sg, si, tg, ti)
        corr = error_correlation(sg, si, tg, ti)
        rows.append((seed, sp_eer, tp_eer, best_alpha, best_eer, corr))
        print(f"\n--- seed {seed}  ({len(sg)} genuine / {len(si)} forgery probes) ---")
        print(f"  spatial alone   {sp_eer * 100:6.2f}%")
        print(f"  temporal alone  {tp_eer * 100:6.2f}%")
        print(f"  fused (a={best_alpha:.2f})  {best_eer * 100:6.2f}%")
        print(f"  error phi {corr['phi']:+.3f}   both-wrong {corr['both_wrong'] * 100:.1f}% "
              f"(independent would give {corr['both_wrong_if_independent'] * 100:.1f}%)")
        # Does calibrating alpha matter, or is plain equal weighting enough?
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            e, _ = compute_eer(a * sg + (1 - a) * tg, a * si + (1 - a) * ti)
            print(f"      alpha {a:.2f} -> {e * 100:6.2f}%"
                  + ("   <- current ALPHA_DEFAULT" if a == 0.5 else ""))

    print("\n=== HONEST CALIBRATION: alpha fitted on writers it is not scored on ===")
    cal = calibrated_fusion(embedded, args.enroll, repeats=args.repeats)
    if cal:
        arr = np.array(cal)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        print(f"  {len(cal)} writer-disjoint splits, 10 fit / 10 test writers each")
        print(f"  fitted alpha            {mean[0]:.3f} +/- {std[0]:.3f}")
        print(f"  spatial alone           {mean[1] * 100:6.2f}% +/- {std[1] * 100:.2f}")
        print(f"  temporal alone          {mean[2] * 100:6.2f}% +/- {std[2] * 100:.2f}")
        print(f"  fused, fitted alpha     {mean[3] * 100:6.2f}% +/- {std[3] * 100:.2f}")
        print(f"  fused, alpha=0.5 fixed  {mean[4] * 100:6.2f}% +/- {std[4] * 100:.2f}")
        better = np.minimum(arr[:, 1], arr[:, 2])
        print(f"  gain over better branch (fitted alpha): "
              f"{(better - arr[:, 3]).mean() * 100:+.2f} points")
        print(f"  splits where fusion beat both branches: "
              f"{int((arr[:, 3] < better).sum())}/{len(cal)}")
        print(f"  at the alpha=0.5 EER threshold: FAR {mean[5] * 100:.2f}%  "
              f"FRR {mean[6] * 100:.2f}%")

    if len(rows) > 1:
        m = lambda i: np.mean([r[i] for r in rows])
        print(f"\n=== {len(rows)}-seed means ===")
        print(f"  spatial {m(1) * 100:.2f}%   temporal {m(2) * 100:.2f}%   "
              f"fused {m(4) * 100:.2f}%  (best alpha mean {m(3):.2f})")
        gain = min(m(1), m(2)) - m(4)
        print(f"  fusion gain over the better branch: {gain * 100:+.2f} points")
    return rows


if __name__ == "__main__":
    main()
