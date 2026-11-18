"""Choose the fused alpha and tau the way the shipped values were chosen, then check them.

Both branches score the same signing: the scan through the spatial model and the pen
recording through the temporal model. Chosen on the BiosecurID-SONOF paired writers,
which neither branch trained on: K = 5 references, 10 enrolment draws seeded 42 + draw,
one draw shared by both branches, nearest-reference similarity. alpha is the grid point
(ALPHA_GRID_STEP) with the lowest mean skilled-forgery EER over the draws, and tau is the
mean EER-point threshold at that alpha. Both are then applied once to the screened NFI
writers (NFI_FUSION_WRITERS), who took no part in the choice.

Shipped from this method: ALPHA_DEFAULT = 0.4, DECISION_THRESHOLD = 0.7564
(docs/ENG_REPORT.md F16). The script prints whether a run reproduces them.

    python scripts/calibrate_fusion.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    ALPHA_DEFAULT,
    ALPHA_GRID_STEP,
    DECISION_THRESHOLD,
    NFI_FUSION_WRITERS,
    NUM_ENROLLMENT_REFERENCES,
    SIGCOMP2009_DIR,
    SPATIAL_WEIGHTS_PATH,
    TEMPORAL_CHANNELS,
    TEMPORAL_WEIGHTS_PATH,
)
from scripts.evaluate import compute_eer, compute_far_frr, embed_samples, enrollment_draw
from src.data.parsers import (load_nfi2009_file, load_sonof_online_file,
                              parse_biosecurid_sonof_pairs)
from src.verification import load_model, nearest_reference_distances, similarity_from_distance

DRAWS = 10
_NFI_SCAN = re.compile(r"^NFI-(\d{3})(\d{2})(\d{3})\.png$", re.IGNORECASE)


def parse_nfi_pairs(root=SIGCOMP2009_DIR, writers=NFI_FUSION_WRITERS):
    """Pair each NFI scan with the pen file of the same signing, for the listed writers.

    `NFI-{author}{session}{target}.png` sits beside its `.hwr`. A genuine has author ==
    target; a forgery targets someone else. Returns the same shape as
    parse_biosecurid_sonof_pairs: {writer: {"genuine": [(png, pen)], "forged": [...]}}.
    """
    out = {}
    for kind, folder in (("genuine", "genuines"), ("forged", "forgeries")):
        for png in sorted(glob.glob(os.path.join(root, folder, "*.png"))):
            match = _NFI_SCAN.match(os.path.basename(png))
            pen = png[:-4] + ".hwr"
            if not match or not os.path.exists(pen):
                continue
            author, target = match.group(1), match.group(3)
            writer = f"nfi_{target}"
            if (author == target) == (kind == "genuine") and writer in writers:
                out.setdefault(writer, {"genuine": [], "forged": []})[kind].append((png, pen))
    return out


def embed_pairs(pairs, spatial_model, temporal_model, load_pen, device="cpu"):
    """{writer: {"genuine": (scan embeddings, pen embeddings), "forged": (...)}}."""
    channels = None if TEMPORAL_CHANNELS == "xyp" else TEMPORAL_CHANNELS
    return {writer: {kind: (embed_samples(spatial_model, [png for png, _ in record[kind]],
                                          "spatial", device),
                            embed_samples(temporal_model, [load_pen(pen, channels)
                                                           for _, pen in record[kind]],
                                          "temporal", device))
                     for kind in ("genuine", "forged")}
            for writer, record in pairs.items()}


def draw_scores(embedded, draw, num_references=NUM_ENROLLMENT_REFERENCES):
    """One enrolment draw: per writer, (scan, pen) similarities of genuine probes and forgeries.

    Both branches use the same references. A writer needs more than num_references
    genuines and at least one forgery, or it is skipped before drawing.
    """
    rng = np.random.default_rng(42 + draw)
    rows = {}
    for writer, record in embedded.items():
        (scan_g, pen_g), (scan_f, pen_f) = record["genuine"], record["forged"]
        if len(scan_g) <= num_references or not len(scan_f):
            continue
        ref, probe = enrollment_draw(rng, len(scan_g), num_references)

        def score(probes, bank):
            return similarity_from_distance(nearest_reference_distances(probes, bank[ref]))

        rows[writer] = {"genuine": (score(scan_g[probe], scan_g), score(pen_g[probe], pen_g)),
                        "forged": (score(scan_f, scan_g), score(pen_f, pen_g))}
    return rows


def fused(rows, kind, alpha):
    """alpha * scan + (1 - alpha) * pen over every writer of one draw."""
    scan = np.concatenate([r[kind][0] for r in rows.values()])
    pen = np.concatenate([r[kind][1] for r in rows.values()])
    return alpha * scan + (1 - alpha) * pen


def evaluate(draws, alpha, tau):
    """Mean and sd over draws of EER, its threshold, and FAR / FRR at tau, in percent."""
    out = {"eer": [], "eer_tau": [], "far": [], "frr": []}
    for rows in draws:
        genuine, forged = fused(rows, "genuine", alpha), fused(rows, "forged", alpha)
        eer, eer_tau = compute_eer(genuine, forged)
        far, frr = compute_far_frr(genuine, forged, tau)
        out["eer"].append(100 * eer)
        out["eer_tau"].append(eer_tau)
        out["far"].append(100 * far)
        out["frr"].append(100 * frr)
    return {key: (float(np.mean(v)), float(np.std(v))) for key, v in out.items()}


def choose(draws, step=ALPHA_GRID_STEP):
    """alpha with the lowest mean EER over the draws, then tau at that alpha's mean EER point."""
    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    eers = [evaluate(draws, a, DECISION_THRESHOLD)["eer"][0] for a in alphas]
    alpha = float(alphas[int(np.argmin(eers))])
    return alpha, evaluate(draws, alpha, DECISION_THRESHOLD)["eer_tau"][0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spatial-weights", default=SPATIAL_WEIGHTS_PATH)
    parser.add_argument("--temporal-weights", default=TEMPORAL_WEIGHTS_PATH)
    args = parser.parse_args(argv)

    spatial = load_model("spatial", args.spatial_weights)
    temporal = load_model("temporal", args.temporal_weights)

    sonof = embed_pairs(parse_biosecurid_sonof_pairs(), spatial, temporal, load_sonof_online_file)
    sonof_draws = [draw_scores(sonof, d) for d in range(DRAWS)]
    alpha, tau = choose(sonof_draws)
    at_sonof = evaluate(sonof_draws, alpha, tau)
    print(f"SONOF, {len(sonof_draws[0])} writers: alpha {alpha:.2f}, tau {tau:.4f}, "
          f"EER {at_sonof['eer'][0]:.2f} +/- {at_sonof['eer'][1]:.2f}%")

    nfi = embed_pairs(parse_nfi_pairs(), spatial, temporal, load_nfi2009_file)
    nfi_draws = [draw_scores(nfi, d) for d in range(DRAWS)]
    at_nfi = evaluate(nfi_draws, alpha, tau)
    print(f"NFI, {len(nfi_draws[0])} writers, at the SONOF choice: "
          f"EER {at_nfi['eer'][0]:.2f} +/- {at_nfi['eer'][1]:.2f}%, "
          f"FAR {at_nfi['far'][0]:.2f}%, FRR {at_nfi['frr'][0]:.2f}%")

    same = alpha == ALPHA_DEFAULT and round(tau, 4) == DECISION_THRESHOLD
    print(f"Shipped: alpha {ALPHA_DEFAULT}, tau {DECISION_THRESHOLD} -> "
          + ("reproduced." if same else "NOT reproduced; check the weights and the data."))
    return {"alpha": alpha, "tau": tau, "sonof": at_sonof, "nfi": at_nfi}


if __name__ == "__main__":
    main()
