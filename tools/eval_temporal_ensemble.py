"""
Evaluate the 6-seed Temporal Ensemble on held-out test writers.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    MODEL_WEIGHTS_DIR,
    TEMPORAL_CHANNELS,
    TEMPORAL_ENSEMBLE_SEEDS,
    TEMPORAL_ENSEMBLE_TEMPLATE,
    TEMPORAL_SPLIT_RATIOS,
    NUM_ENROLLMENT_REFERENCES,
)
from scripts.train import build_writer_samples
from src.data.dataset import writer_independent_split
from src.models.siamese import TemporalSiamese
from src.verification import similarity_from_distance
from scripts.evaluate import compute_eer, compute_far_frr, enrollment_draw

SEEDS = list(TEMPORAL_ENSEMBLE_SEEDS)
WEIGHT_FILES = [TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=s) for s in SEEDS]


def load_ensemble(device="cpu"):
    models = []
    for wf in WEIGHT_FILES:
        path = os.path.join(MODEL_WEIGHTS_DIR, wf)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing ensemble weight: {path}")
        sd = torch.load(path, map_location=device, weights_only=True)
        model = TemporalSiamese.from_state_dict(sd).to(device).eval()
        models.append(model)
    return models


def embed_trajectory(model, traj, device="cpu"):
    tensor = torch.as_tensor(np.asarray(traj, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode(tensor).squeeze(0).cpu().numpy()
    return emb


def main():
    device = "cpu"
    data_dirs = ["data/svc2004_task2", "data/sigcomp2011"]
    print(f"Loading paired temporal datasets: {data_dirs}")
    data = build_writer_samples(data_dirs, extra_channels=TEMPORAL_CHANNELS)
    print(f"Total writers loaded: {len(data)}")

    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(data), *TEMPORAL_SPLIT_RATIOS, seed=42
    )
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test writers")

    models = load_ensemble(device=device)
    print(f"Loaded {len(models)} temporal models for ensemble.")

    # Evaluate each model individually and as an ensemble
    # Draw K=5 reference sets
    eval_draw_seed = 42
    rng = np.random.default_rng(eval_draw_seed)

    all_single_gen_scores = [[] for _ in range(len(models))]
    all_single_imp_scores = [[] for _ in range(len(models))]
    all_ens_gen_scores = []
    all_ens_imp_scores = []

    for writer_id in test_ids:
        genuines = data[writer_id]["genuine"]
        forgeries = data[writer_id].get("forged", [])
        if len(genuines) < NUM_ENROLLMENT_REFERENCES + 1:
            continue

        ref_idx, probe_idx = enrollment_draw(rng, len(genuines), NUM_ENROLLMENT_REFERENCES)
        ref_trajs = [genuines[i] for i in ref_idx]
        probe_gen_trajs = [genuines[i] for i in probe_idx]
        probe_forg_trajs = forgeries

        # Per model reference embeddings
        model_ref_embs = []
        for model in models:
            refs = [embed_trajectory(model, t, device) for t in ref_trajs]
            model_ref_embs.append(np.stack(refs))

        # Evaluate Genuines
        for g_traj in probe_gen_trajs:
            sims = []
            for i, model in enumerate(models):
                probe_emb = embed_trajectory(model, g_traj, device)
                dists = np.linalg.norm(model_ref_embs[i] - probe_emb[None, :], axis=1)
                sim = float(similarity_from_distance(dists.min()))
                all_single_gen_scores[i].append(sim)
                sims.append(sim)
            all_ens_gen_scores.append(np.mean(sims))

        # Evaluate Forgeries
        for f_traj in probe_forg_trajs:
            sims = []
            for i, model in enumerate(models):
                probe_emb = embed_trajectory(model, f_traj, device)
                dists = np.linalg.norm(model_ref_embs[i] - probe_emb[None, :], axis=1)
                sim = float(similarity_from_distance(dists.min()))
                all_single_imp_scores[i].append(sim)
                sims.append(sim)
            all_ens_imp_scores.append(np.mean(sims))

    print("\n================ SINGLE MODEL PERFORMANCE ================")
    print(f"{'Model':<25}{'EER':>10}{'Threshold':>12}")
    single_eers = []
    for i, wf in enumerate(WEIGHT_FILES):
        gen = np.array(all_single_gen_scores[i])
        imp = np.array(all_single_imp_scores[i])
        eer, tau = compute_eer(gen, imp)
        single_eers.append(eer)
        print(f"{wf:<25}{eer * 100:>9.2f}%{tau:>12.4f}")
    print(f"{'Mean Single Model':<25}{np.mean(single_eers) * 100:>9.2f}%")

    print("\n================ ENSEMBLE PERFORMANCE ================")
    ens_gen = np.array(all_ens_gen_scores)
    ens_imp = np.array(all_ens_imp_scores)
    ens_eer, ens_tau = compute_eer(ens_gen, ens_imp)
    ens_far, ens_frr = compute_far_frr(ens_gen, ens_imp, ens_tau)

    print(f"{'Ensemble EER:':<20}{ens_eer * 100:>9.2f}%")
    print(f"{'FAR @ EER:':<20}{ens_far * 100:>9.2f}%")
    print(f"{'FRR @ EER:':<20}{ens_frr * 100:>9.2f}%")
    print(f"{'Decision Threshold:':<20}{ens_tau:>9.4f}")
    print(f"{'Ensemble Gain:':<20}{(np.mean(single_eers) - ens_eer) * 100:>+9.2f} points over mean single model")
    print("=======================================================")


if __name__ == "__main__":
    main()
