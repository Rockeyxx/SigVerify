"""White-box PGD attack on the scan branch (alone and fused) using BiosecurID-SONOF."""
import argparse

import numpy as np
import torch

from config.settings import (ALPHA_DEFAULT, DECISION_THRESHOLD, NUM_ENROLLMENT_REFERENCES,
                             SPATIAL_DECISION_THRESHOLD, SPATIAL_WEIGHTS_PATH,
                             TEMPORAL_WEIGHTS_PATH)
from scripts.calibrate_fusion import DRAWS, draw_scores, embed_pairs
from scripts.evaluate import enrollment_draw
from src.data.dataset import preprocess_image
from src.data.parsers import load_sonof_online_file, parse_biosecurid_sonof_pairs
from src.verification import load_model, similarity_from_distance

EPSILONS = (0, 1, 2, 4, 8)  # in 1/255 pixel steps
STEPS = 20


def pgd(model, x, references, eps, steps=STEPS):
    """Run L_inf PGD on x [N, 1, H, W] toward its nearest reference embedding."""
    if eps == 0:
        return x
    step = 2.5 * eps / steps
    adv = x.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        distance = torch.cdist(model.encode(adv), references).min(dim=1).values
        (grad,) = torch.autograd.grad(distance.sum(), adv)
        adv = adv.detach() - step * grad.sign()
        adv = (x + (adv - x).clamp(-eps, eps)).clamp(0, 1)
    return adv.detach()


def attack_draw(spatial, pairs, embedded, draw, device, target="references"):
    """Return per-epsilon attacked scan similarities and untouched pen similarities for one draw."""
    rng = np.random.default_rng(42 + draw)
    baseline = draw_scores(embedded, draw)
    scan, pen = {eps: [] for eps in EPSILONS}, []
    for writer, rows in baseline.items():
        genuine = embedded[writer]["genuine"][0]
        ref, probe = enrollment_draw(rng, len(genuine), NUM_ENROLLMENT_REFERENCES)
        references = torch.tensor(genuine[ref], device=device)
        aim = references if target == "references" else torch.tensor(genuine[probe[:1]], device=device)
        forged = torch.tensor(np.stack([preprocess_image(png) for png, _ in pairs[writer]["forged"]]),
                              device=device)
        for eps in EPSILONS:
            adv = pgd(spatial, forged, aim, eps / 255)
            with torch.no_grad():
                distance = torch.cdist(spatial.encode(adv), references).min(dim=1).values
            scan[eps].append(similarity_from_distance(distance.cpu().numpy()))
        # Verify attack at eps=0 matches baseline scorer before reading higher epsilons.
        assert np.allclose(scan[0][-1], rows["forged"][0], atol=1e-4), writer
        pen.append(rows["forged"][1])
        print(f"  draw {draw + 1}: writer {len(pen)}/{len(baseline)}", flush=True)
    return {eps: np.concatenate(v) for eps, v in scan.items()}, np.concatenate(pen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spatial-weights", default=SPATIAL_WEIGHTS_PATH)
    parser.add_argument("--temporal-weights", default=TEMPORAL_WEIGHTS_PATH)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target", choices=("references", "public"), default="references")
    args = parser.parse_args(argv)

    spatial = load_model("spatial", args.spatial_weights, args.device)
    temporal = load_model("temporal", args.temporal_weights, args.device)
    pairs = parse_biosecurid_sonof_pairs()
    embedded = embed_pairs(pairs, spatial, temporal, load_sonof_online_file, args.device)

    results = {eps: {"scan": [], "fused": []} for eps in EPSILONS}
    ceiling = []
    for draw in range(args.draws):
        scan, pen = attack_draw(spatial, pairs, embedded, draw, args.device, args.target)
        for eps in EPSILONS:
            results[eps]["scan"].append(100 * np.mean(scan[eps] >= SPATIAL_DECISION_THRESHOLD))
            fused = ALPHA_DEFAULT * scan[eps] + (1 - ALPHA_DEFAULT) * pen
            results[eps]["fused"].append(100 * np.mean(fused >= DECISION_THRESHOLD))
        ceiling.append(100 * np.mean(ALPHA_DEFAULT + (1 - ALPHA_DEFAULT) * pen >= DECISION_THRESHOLD))
        print(f"draw {draw + 1}/{args.draws} done, {len(pen)} forgeries", flush=True)

    print(f"\nForgeries accepted (%), mean +/- sd over {args.draws} draws, "
          f"PGD {STEPS} steps, target {args.target}, device {args.device}")
    print(f"{'eps/255':>8} {'scan alone':>16} {'fused':>16}")
    for eps in EPSILONS:
        s, f = results[eps]["scan"], results[eps]["fused"]
        print(f"{eps:>8} {np.mean(s):>9.2f} +/- {np.std(s):<4.2f} {np.mean(f):>9.2f} +/- {np.std(f):.2f}")
    print(f"Fused ceiling (scan similarity forced to 1.0): "
          f"{np.mean(ceiling):.2f} +/- {np.std(ceiling):.2f}")


if __name__ == "__main__":
    main()
