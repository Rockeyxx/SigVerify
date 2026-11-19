"""The shipped weights on the XPPen in-house campaign, and the D6 threshold check.

Campaign spec (docs/superpowers/specs/2026-11-02-xppen-capture-campaign-design.md):
M2 shipped weights, M3 calibration by rule D6, M7 warm-up, and the forgers' second attempt
after "Sign again" for the step-up decision. Product order: each writer's first 5 genuine
signings (tablet and paper of the same index) are the references, the later genuines and the
team forgeries are probes. Forgers are team members and never scored as writers (spec 6).
Prints aggregates only: no file name and no signature leaves this script.

    python tools/research/campaign_eval.py
"""
from __future__ import annotations

import numpy as np

from config.settings import (ALPHA_DEFAULT, DECISION_THRESHOLD, NUM_ENROLLMENT_REFERENCES,
                             SPATIAL_DECISION_THRESHOLD, SPATIAL_WEIGHTS_PATH, STEP_UP_ACCEPT,
                             STEP_UP_REJECT, STEP_UP_TWO, TARGET_FAR, TARGET_FRR,
                             TEMPORAL_DECISION_THRESHOLD, TEMPORAL_WEIGHTS_PATH)
from scripts.calibrate_fusion import DRAWS, draw_scores, evaluate, fused
from scripts.evaluate import compute_eer, embed_samples
from src.data.campaign import inhouse_split, load_inhouse_tablet_file, parse_inhouse
from src.data.dataset import quality_gate, trajectory_gate
from src.gui.image_cleanup import SignatureNotFoundError, crop_to_ink
from src.verification import load_model, nearest_reference_distances, similarity_from_distance

MIN_GENUINE = 10            # spec 6: fewer usable genuines per modality leaves the headline arm
BOOTSTRAP, HALF_SPLITS = 1000, 200


def embed_campaign(writers, spatial, temporal):
    """{writer: {"genuine": (scan, pen), "forged": (scan, pen), "attempt", "os", "lost"}}.

    Only signings with both sides count. The scan goes through the app's own crop
    (crop_to_ink). A signing the app would refuse (no ink found, or either input gate fails)
    is dropped and counted in "lost": at enrolment the operator retakes it, so the next
    signing becomes a reference; as a probe it would be decided on one branch, not fused."""
    out = {}
    for writer, record in writers.items():
        lost = 0

        def embed(entries):
            nonlocal lost
            scans, pens, kept = [], [], []
            for e in entries:
                if not (e["tablet"] and e["paper"]):
                    continue
                try:
                    scan = crop_to_ink(e["paper"])
                except SignatureNotFoundError:
                    lost += 1
                    continue
                pen = load_inhouse_tablet_file(e["tablet"], e["pressure_scale"])
                if not (quality_gate(scan)[0] and trajectory_gate(pen)[0]):
                    lost += 1
                    continue
                scans.append(scan)
                pens.append(pen)
                kept.append(e)
            return (embed_samples(spatial, scans, "spatial", "cpu"),
                    embed_samples(temporal, pens, "temporal", "cpu")), kept

        genuine, g_kept = embed(record["genuine"])
        forged, f_kept = embed(record["forged"])
        out[writer] = {"genuine": genuine, "forged": forged, "lost": lost,
                       "attempt": np.array([e["attempt"] for e in f_kept]),
                       "os": sorted({e["os"] for e in g_kept if e["os"]}),
                       "kept": (g_kept, f_kept)}
    return out


def _score(probes, references):
    return similarity_from_distance(nearest_reference_distances(probes, references))


def product_rows(embedded, ref=range(NUM_ENROLLMENT_REFERENCES), probe_from=None):
    """One "draw" in product order: genuines are index-sorted, so `ref` positions are the
    first signings; probes are every later genuine (from probe_from, default after the refs)
    and all forgeries. Same shape as calibrate_fusion.draw_scores."""
    ref = list(ref)
    start = max(ref) + 1 if probe_from is None else probe_from
    rows = {}
    for writer, rec in embedded.items():
        (scan_g, pen_g), (scan_f, pen_f) = rec["genuine"], rec["forged"]
        rows[writer] = {"genuine": (_score(scan_g[start:], scan_g[ref]),
                                    _score(pen_g[start:], pen_g[ref])),
                        "forged": (_score(scan_f, scan_g[ref]), _score(pen_f, pen_g[ref]))}
    return rows


def random_forgery_far(embedded, alpha, tau, ref=range(NUM_ENROLLMENT_REFERENCES)):
    """FAR when another writer's genuine probes are scored against this writer's references."""
    ref, start = list(ref), NUM_ENROLLMENT_REFERENCES
    scores = []
    for writer, rec in embedded.items():
        for other, orec in embedded.items():
            if other == writer:
                continue
            scan = _score(orec["genuine"][0][start:], rec["genuine"][0][ref])
            pen = _score(orec["genuine"][1][start:], rec["genuine"][1][ref])
            scores.append(alpha * scan + (1 - alpha) * pen)
    return 100 * float(np.mean(np.concatenate(scores) >= tau))


def step_up_accepts(first, second):
    """verify()'s step-up rule on a pair of fused scores: accept, reject, or ask once more."""
    if first >= STEP_UP_ACCEPT:
        return True
    if first < STEP_UP_REJECT:
        return False
    return (first + second) / 2 >= STEP_UP_TWO


def _eer(rows, alpha):
    return compute_eer(fused(rows, "genuine", alpha), fused(rows, "forged", alpha))


def spread(rows, alpha, rng):
    """D6 spread: writer bootstrap of the EER and its threshold, and 200 writer-disjoint half
    splits (threshold at the EER point of one half, FAR and FRR on the other)."""
    names = list(rows)
    boot = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(len(names), len(names))
        boot.append(_eer({i: rows[names[j]] for i, j in enumerate(pick)}, alpha))
    boot = np.array(boot)
    halves = []
    for _ in range(HALF_SPLITS):
        order = rng.permutation(len(names))
        a = {names[i]: rows[names[i]] for i in order[: len(names) // 2]}
        b = {names[i]: rows[names[i]] for i in order[len(names) // 2:]}
        res = evaluate([b], alpha, _eer(a, alpha)[1])
        halves.append((res["far"][0], res["frr"][0]))
    halves = np.array(halves)
    return {"eer_90": np.percentile(100 * boot[:, 0], [5, 95]),
            "tau_90": np.percentile(boot[:, 1], [5, 95]),
            "half_far": (halves[:, 0].mean(), halves[:, 0].std()),
            "half_frr": (halves[:, 1].mean(), halves[:, 1].std())}


def report_branch(name, rows, alpha, tau, rng):
    """D6 for one decision: keep tau if FAR and FRR both meet target, else its EER point."""
    res = evaluate([rows], alpha, tau)
    far, frr, eer, eer_tau = (res[k][0] for k in ("far", "frr", "eer", "eer_tau"))
    keep = far <= 100 * TARGET_FAR and frr <= 100 * TARGET_FRR
    s = spread(rows, alpha, rng)
    n_g = len(fused(rows, "genuine", alpha))
    n_f = len(fused(rows, "forged", alpha))
    print(f"\n{name} (alpha {alpha}): {len(rows)} writers, {n_g} genuine probes, {n_f} forgeries")
    print(f"  at shipped tau {tau:.4f}: FAR {far:.2f}%  FRR {frr:.2f}%")
    print(f"  EER {eer:.2f}% (writer bootstrap 90%: {s['eer_90'][0]:.2f}-{s['eer_90'][1]:.2f}) "
          f"at tau {eer_tau:.4f} (90%: {s['tau_90'][0]:.4f}-{s['tau_90'][1]:.4f})")
    print(f"  half splits: FAR {s['half_far'][0]:.2f} +/- {s['half_far'][1]:.2f}%, "
          f"FRR {s['half_frr'][0]:.2f} +/- {s['half_frr'][1]:.2f}%")
    verdict = ("keep shipped tau" if keep else
               f"replace tau with {eer_tau:.4f}" if eer <= 100 * TARGET_FAR else
               f"no tau meets both 5% targets (EER {eer:.2f}%)")
    print(f"  D6: {verdict}")
    return res


def main():
    spatial = load_model("spatial", SPATIAL_WEIGHTS_PATH)
    temporal = load_model("temporal", TEMPORAL_WEIGHTS_PATH)
    writers = parse_inhouse()
    split = inhouse_split(writers)
    embedded = embed_campaign(writers, spatial, temporal)
    headline = {w: r for w, r in embedded.items()
                if w not in split["forgers"] and len(r["genuine"][0]) >= MIN_GENUINE
                and len(r["forged"][0])}
    print(f"read {len(writers)} writers; forgers (team, not scored): {split['forgers']}; "
          f"headline {len(headline)}; signings the app refused (crop or gate): "
          f"{sum(r['lost'] for r in embedded.values())}")
    print("left out:", sorted(set(embedded) - set(headline) - set(split["forgers"])))

    rng = np.random.default_rng(42)
    rows = product_rows(headline)
    report_branch("Fused", rows, ALPHA_DEFAULT, DECISION_THRESHOLD, rng)
    report_branch("Pen alone", rows, 0.0, TEMPORAL_DECISION_THRESHOLD, rng)
    report_branch("Scan alone", rows, 1.0, SPATIAL_DECISION_THRESHOLD, rng)

    print("\nRandom forgeries (other writers' genuines), FAR at the shipped thresholds: "
          f"fused {random_forgery_far(headline, ALPHA_DEFAULT, DECISION_THRESHOLD):.2f}%, "
          f"pen {random_forgery_far(headline, 0.0, TEMPORAL_DECISION_THRESHOLD):.2f}%, "
          f"scan {random_forgery_far(headline, 1.0, SPATIAL_DECISION_THRESHOLD):.2f}%")

    draws = [draw_scores(headline, d) for d in range(DRAWS)]
    for name, a in (("fused", ALPHA_DEFAULT), ("pen", 0.0), ("scan", 1.0)):
        e = evaluate(draws, a, DECISION_THRESHOLD)["eer"]
        print(f"Literature arm ({DRAWS} random draws of 5) {name}: EER {e[0]:.2f} +/- {e[1]:.2f}%")

    for os_name in ("Linux", "Windows"):
        part = {w: r for w, r in rows.items() if headline[w]["os"] == [os_name]}
        if part:
            res = evaluate([part], ALPHA_DEFAULT, DECISION_THRESHOLD)
            print(f"{os_name} writers ({len(part)}): fused EER {res['eer'][0]:.2f}%, "
                  f"FAR {res['far'][0]:.2f}%, FRR {res['frr'][0]:.2f}% at the shipped tau")

    # M7 warm-up: references 1-5 against 2-6, the same probes (7 onward) for both.
    for label, ref in (("refs 1-5", range(5)), ("refs 2-6", range(1, 6))):
        r = product_rows(headline, ref, probe_from=6)
        print(f"Warm-up, {label}, probes 7-12: fused EER {_eer(r, ALPHA_DEFAULT)[0] * 100:.2f}%")

    # Step-up: genuine probe pairs (6,7), (8,9), (10,11) against forgers' attempts 1 and 2.
    g_acc, f_acc, f1, f2 = [], [], [], []
    for w, r in rows.items():
        g = ALPHA_DEFAULT * r["genuine"][0] + (1 - ALPHA_DEFAULT) * r["genuine"][1]
        g_acc += [step_up_accepts(g[i], g[i + 1]) for i in range(0, len(g) - 1, 2)]
        f = ALPHA_DEFAULT * r["forged"][0] + (1 - ALPHA_DEFAULT) * r["forged"][1]
        att = headline[w]["attempt"]
        if (att == 1).any() and (att == 2).any():
            a1, a2 = f[att == 1][0], f[att == 2][0]
            f1.append(a1)
            f2.append(a2)
            f_acc.append(step_up_accepts(a1, a2))
    f1, f2 = np.array(f1), np.array(f2)
    print(f"\nStep-up: genuine pairs refused {100 * (1 - np.mean(g_acc)):.1f}% of {len(g_acc)}, "
          f"forger pairs passed {100 * np.mean(f_acc):.1f}% of {len(f_acc)}")
    print(f"  forger after 'Sign again': attempt 1 mean {f1.mean():.4f}, attempt 2 mean "
          f"{f2.mean():.4f}; attempt 2 higher for {int((f2 > f1).sum())} of {len(f1)}; "
          f"one signing at tau passes attempt 1 {100 * np.mean(f1 >= DECISION_THRESHOLD):.1f}%, "
          f"attempt 2 {100 * np.mean(f2 >= DECISION_THRESHOLD):.1f}%")


if __name__ == "__main__":
    main()
