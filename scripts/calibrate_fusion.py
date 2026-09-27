"""
Paired Multimodal Calibration Engine.

Renders online trajectories into paired spatial images, evaluates both branches
on the exact same signatures across validation writers, and derives the true
empirically optimal fusion weight alpha and decision threshold tau.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    NUM_ENROLLMENT_REFERENCES,
    SIGCOMP2011_DIR,
    SPATIAL_WEIGHTS_PATH,
    SVC2004_TASK2_DIR,
    TEMPORAL_CHANNELS,
    TEMPORAL_WEIGHTS_PATH,
    TEMPORAL_SPLIT_RATIOS,
)
from scripts.evaluate import compute_eer, compute_far_frr
from src.data.dataset import writer_independent_split
from src.data.parsers import merge_datasets
from src.data.renderer import render_trajectory_to_image
from src.verification import similarity_from_distance
from src.models.siamese import SpatialSiamese, TemporalSiamese


def grid_search_fusion_params(
    s_gen: np.ndarray,
    s_imp: np.ndarray,
    t_gen: np.ndarray,
    t_imp: np.ndarray,
    alpha_steps: int = 21,
) -> tuple[float, float, float]:
    """
    Search alpha in [0, 1] to minimize EER on paired scores.

    Returns:
        (best_alpha, best_tau, best_eer)
    """
    best_eer = 1.0
    best_alpha = 0.5
    best_tau = 0.5

    for alpha in np.linspace(0.0, 1.0, alpha_steps):
        f_gen = alpha * s_gen + (1.0 - alpha) * t_gen
        f_imp = alpha * s_imp + (1.0 - alpha) * t_imp

        eer, tau = compute_eer(f_gen, f_imp)
        if eer < best_eer:
            best_eer = eer
            best_alpha = float(alpha)
            best_tau = float(tau)

    return best_alpha, best_tau, float(best_eer)


def evaluate_paired_corpus(
    spatial_model: SpatialSiamese,
    temporal_model: TemporalSiamese,
    writer_samples: dict,
    writer_ids: list[str],
    num_references: int = NUM_ENROLLMENT_REFERENCES,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute paired (spatial, temporal) scores on genuine probes and forgeries
    using K-nearest reference enrollment.
    """
    rng = np.random.default_rng(seed)
    s_gen_all, s_imp_all = [], []
    t_gen_all, t_imp_all = [], []

    spatial_model = spatial_model.to(device).eval()
    temporal_model = temporal_model.to(device).eval()

    with torch.no_grad():
        for writer in writer_ids:
            samples = writer_samples.get(writer, {})
            genuines = samples.get("genuine", [])
            forgeries = samples.get("forged", [])

            if len(genuines) < num_references + 1 or not forgeries:
                continue

            order = rng.permutation(len(genuines))
            ref_trajs = [genuines[i] for i in order[:num_references]]
            probe_gen_trajs = [genuines[i] for i in order[num_references:]]
            probe_forg_trajs = forgeries

            # Render spatial images from trajectories
            ref_imgs = [render_trajectory_to_image(t) for t in ref_trajs]
            probe_gen_imgs = [render_trajectory_to_image(t) for t in probe_gen_trajs]
            probe_forg_imgs = [render_trajectory_to_image(t) for t in probe_forg_trajs]

            # 1. Spatial Embeddings
            ref_s_tensors = torch.tensor(np.stack(ref_imgs), dtype=torch.float32).to(device)
            ref_s_emb = spatial_model.encode(ref_s_tensors).cpu().numpy()

            gen_s_tensors = torch.tensor(np.stack(probe_gen_imgs), dtype=torch.float32).to(device)
            gen_s_emb = spatial_model.encode(gen_s_tensors).cpu().numpy()

            forg_s_tensors = torch.tensor(np.stack(probe_forg_imgs), dtype=torch.float32).to(device)
            forg_s_emb = spatial_model.encode(forg_s_tensors).cpu().numpy()

            # Nearest-of-K spatial distances & similarities
            s_d_gen = np.linalg.norm(gen_s_emb[:, None, :] - ref_s_emb[None, :, :], axis=2).min(axis=1)
            s_d_forg = np.linalg.norm(forg_s_emb[:, None, :] - ref_s_emb[None, :, :], axis=2).min(axis=1)
            s_gen_all.append(similarity_from_distance(s_d_gen))
            s_imp_all.append(similarity_from_distance(s_d_forg))

            # 2. Temporal Embeddings (padded variable length)
            def embed_trajs(traj_list):
                seqs = [torch.tensor(t, dtype=torch.float32) for t in traj_list]
                lengths = torch.tensor([len(t) for t in traj_list], dtype=torch.long)
                padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
                return temporal_model.encode(padded, lengths).cpu().numpy()

            ref_t_emb = embed_trajs(ref_trajs)
            gen_t_emb = embed_trajs(probe_gen_trajs)
            forg_t_emb = embed_trajs(probe_forg_trajs)

            t_d_gen = np.linalg.norm(gen_t_emb[:, None, :] - ref_t_emb[None, :, :], axis=2).min(axis=1)
            t_d_forg = np.linalg.norm(forg_t_emb[:, None, :] - ref_t_emb[None, :, :], axis=2).min(axis=1)
            t_gen_all.append(similarity_from_distance(t_d_gen))
            t_imp_all.append(similarity_from_distance(t_d_forg))

    if not s_gen_all:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(s_gen_all),
        np.concatenate(s_imp_all),
        np.concatenate(t_gen_all),
        np.concatenate(t_imp_all),
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate multimodal fusion parameters")
    parser.add_argument("--spatial-weights",
                        default=SPATIAL_WEIGHTS_PATH)
    parser.add_argument("--temporal-weights",
                        default=TEMPORAL_WEIGHTS_PATH)
    parser.add_argument("--data-dirs", nargs="+",
                        default=[SVC2004_TASK2_DIR, SIGCOMP2011_DIR])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-steps", type=int, default=21)
    args = parser.parse_args()

    from scripts.train import build_writer_samples
    data = build_writer_samples(args.data_dirs, extra_channels=TEMPORAL_CHANNELS)

    if not data:
        print("No datasets found in specified paths.")
        sys.exit(1)

    train_ids, val_ids, test_ids = writer_independent_split(
        sorted(data), *TEMPORAL_SPLIT_RATIOS, seed=args.seed
    )

    print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test writers")

    spatial_model = SpatialSiamese.from_state_dict(
        torch.load(args.spatial_weights, map_location="cpu", weights_only=True))
    temporal_model = TemporalSiamese.from_state_dict(
        torch.load(args.temporal_weights, map_location="cpu", weights_only=True))

    print("Evaluating paired validation split...")
    s_val_gen, s_val_imp, t_val_gen, t_val_imp = evaluate_paired_corpus(
        spatial_model, temporal_model, data, val_ids, seed=args.seed
    )

    alpha, tau, val_eer = grid_search_fusion_params(
        s_val_gen, s_val_imp, t_val_gen, t_val_imp, alpha_steps=args.alpha_steps
    )
    print(f"\n[Validation Optimal] alpha = {alpha:.4f}, tau = {tau:.4f}, EER = {val_eer * 100:.2f}%\n")

    print("Evaluating paired held-out test split...")
    s_test_gen, s_test_imp, t_test_gen, t_test_imp = evaluate_paired_corpus(
        spatial_model, temporal_model, data, test_ids, seed=args.seed
    )

    # Spatial alone
    s_eer, s_tau = compute_eer(s_test_gen, s_test_imp)
    s_far, s_frr = compute_far_frr(s_test_gen, s_test_imp, s_tau)

    # Temporal alone
    t_eer, t_tau = compute_eer(t_test_gen, t_test_imp)
    t_far, t_frr = compute_far_frr(t_test_gen, t_test_imp, t_tau)

    # Fused (fitted alpha)
    f_test_gen = alpha * s_test_gen + (1.0 - alpha) * t_test_gen
    f_test_imp = alpha * s_test_imp + (1.0 - alpha) * t_test_imp
    f_eer, _ = compute_eer(f_test_gen, f_test_imp)
    f_far, f_frr = compute_far_frr(f_test_gen, f_test_imp, tau)

    # Fixed alpha = 0.5 (Project default; SUMMARY.md §7 proved fixed beats fitted across 25 splits)
    from config.settings import ALPHA_DEFAULT
    fixed_gen = ALPHA_DEFAULT * s_test_gen + (1.0 - ALPHA_DEFAULT) * t_test_gen
    fixed_imp = ALPHA_DEFAULT * s_test_imp + (1.0 - ALPHA_DEFAULT) * t_test_imp
    fixed_eer, fixed_tau = compute_eer(fixed_gen, fixed_imp)
    fixed_far, fixed_frr = compute_far_frr(fixed_gen, fixed_imp, fixed_tau)

    print("=================== TEST RESULTS ===================")
    print(f"{'Branch':<22}{'EER':>10}{'FAR':>10}{'FRR':>10}{'Threshold':>12}")
    print(f"{'Spatial':<22}{s_eer * 100:>9.2f}%{s_far * 100:>9.2f}%{s_frr * 100:>9.2f}%{s_tau:>12.4f}")
    print(f"{'Temporal':<22}{t_eer * 100:>9.2f}%{t_far * 100:>9.2f}%{t_frr * 100:>9.2f}%{t_tau:>12.4f}")
    print(f"{f'FUSED (fitted a={alpha:.2f})':<22}{f_eer * 100:>9.2f}%{f_far * 100:>9.2f}%{f_frr * 100:>9.2f}%{tau:>12.4f}")
    print(f"{f'FUSED (fixed a={ALPHA_DEFAULT:.2f})':<22}{fixed_eer * 100:>9.2f}%{fixed_far * 100:>9.2f}%{fixed_frr * 100:>9.2f}%{fixed_tau:>12.4f}")
    print("====================================================")
    print("Note: ALPHA_DEFAULT = 0.5 remains fixed in config/settings.py per SUMMARY.md §7.")


if __name__ == "__main__":
    main()
