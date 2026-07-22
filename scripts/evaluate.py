"""
Evaluation Script — Signature Verification System

Standalone evaluation and calibration script. Imports from src.models
and runs analysis independently of the training pipeline.

Usage:
    python scripts/evaluate.py --branch spatial --weights models/weights/spatial_cnn.pth --data-dir ./data
    python scripts/evaluate.py --alpha-sweep --spatial-weights ... --temporal-weights ... --data-dir ./data
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import ALPHA_GRID_STEP, TARGET_FAR, TARGET_FRR


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate from genuine and impostor score distributions.

    Returns: (eer: float, threshold_at_eer: float)
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # EER is where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """Compute FAR and FRR at a given threshold."""
    far = np.mean(impostor_scores >= threshold)
    frr = np.mean(genuine_scores < threshold)
    return float(far), float(frr)


def plot_roc(genuine_scores, impostor_scores, title, output_path):
    """Generate and save ROC curve."""
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved: {output_path}")


def plot_det(genuine_scores, impostor_scores, title, output_path):
    """Generate and save DET curve."""
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, fnr, _ = det_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr * 100, fnr * 100, linewidth=2)
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"DET curve saved: {output_path}")


def plot_confusion_matrix(genuine_scores, impostor_scores, threshold, title, output_path):
    """Generate and save confusion matrix plot."""
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Forged", "Genuine"])
    ax.set_yticklabels(["Forged", "Genuine"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {output_path}")


def alpha_grid_search(spatial_genuine, spatial_impostor,
                      temporal_genuine, temporal_impostor,
                      step=ALPHA_GRID_STEP, output_path=None):
    """
    Grid search α ∈ [0, 1] to minimize EER on validation set.
    Returns: (best_alpha, best_eer, results: list of (alpha, eer))
    """
    results = []
    for alpha_int in range(0, int(1.0 / step) + 1):
        alpha = alpha_int * step
        fused_genuine = alpha * spatial_genuine + (1 - alpha) * temporal_genuine
        fused_impostor = alpha * spatial_impostor + (1 - alpha) * temporal_impostor
        eer, _ = compute_eer(fused_genuine, fused_impostor)
        results.append((alpha, eer))

    best_alpha, best_eer = min(results, key=lambda x: x[1])

    if output_path:
        alphas = [r[0] for r in results]
        eers = [r[1] for r in results]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, eers, 'o-', linewidth=2, markersize=4)
        ax.axvline(x=best_alpha, color='r', linestyle='--', label=f"Best α={best_alpha:.2f}")
        ax.set_xlabel("α (spatial weight)")
        ax.set_ylabel("EER")
        ax.set_title("α vs EER — Fusion Weight Calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"α sweep plot saved: {output_path}")

    return best_alpha, best_eer, results


def tri_point_report(spatial_eer, temporal_eer, fused_eer, fused_is_provisional=True):
    """Print the tri-point EER report."""
    print("\n" + "=" * 50)
    print("TRI-POINT EER REPORT")
    print("=" * 50)
    print(f"  Spatial-only EER:   {spatial_eer*100:.2f}%  (target: <{TARGET_FAR*100:.0f}%)")
    print(f"  Temporal-only EER:  {temporal_eer*100:.2f}%  (target: <{TARGET_FRR*100:.0f}%)")
    label = "PROVISIONAL" if fused_is_provisional else "FINAL"
    print(f"  Fused EER:          {fused_eer*100:.2f}%  [{label}]")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate signature verification models")
    parser.add_argument("--branch", choices=["spatial", "temporal", "fused"],
                        help="Which branch to evaluate")
    parser.add_argument("--weights", type=str, help="Path to model weights (.pth)")
    parser.add_argument("--spatial-weights", type=str, help="Spatial model weights for fused eval")
    parser.add_argument("--temporal-weights", type=str, help="Temporal model weights for fused eval")
    parser.add_argument("--data-dir", type=str, required=True, help="Test data directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for plots")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run α grid search")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluation mode: {args.branch or 'fused'}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    print("Evaluation script ready. Connect dataset loading and model inference to run metrics.")


if __name__ == "__main__":
    main()
