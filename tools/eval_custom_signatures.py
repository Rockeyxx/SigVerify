"""Evaluate spatial models on custom signature image samples."""

import os
import tempfile
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.verification import load_model, embed, _nearest_reference_score
from config.settings import SPATIAL_DECISION_THRESHOLD, MODEL_WEIGHTS_DIR


def auto_crop_signature(image_path):
    """Auto-crop signature photo tight to ink contours before embedding."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_path

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_path

    valid_contours = [c for c in contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > 20]
    if not valid_contours:
        valid_contours = contours

    x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
    y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)

    h, w = img.shape
    pad = 15
    x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
    x2, y2 = min(w, x_max + pad), min(h, y_max + pad)

    cropped = img[y1:y2, x1:x2]
    # tempfile, not a literal "/tmp": AGENTS.md requires the project run
    # identically on Windows, which has no /tmp.
    tmp_path = os.path.join(tempfile.gettempdir(),
                            f"crop_{os.path.basename(image_path)}")
    cv2.imwrite(tmp_path, cropped)
    return tmp_path


def evaluate_writer_samples(model, references, genuine_probe, forged_probe, device="cpu"):
    """
    Evaluate a single writer set (references, genuine probe, forged probe) with a spatial model.
    """
    cropped_refs = [auto_crop_signature(r) for r in references]
    cropped_gen = auto_crop_signature(genuine_probe)
    cropped_forg = auto_crop_signature(forged_probe)

    ref_embs = [embed(model, ref_path, "spatial", device=device) for ref_path in cropped_refs]
    gen_emb = embed(model, cropped_gen, "spatial", device=device)
    forg_emb = embed(model, cropped_forg, "spatial", device=device)


    gen_score = _nearest_reference_score(ref_embs, gen_emb)
    forg_score = _nearest_reference_score(ref_embs, forg_emb)

    gen_decision = "Genuine" if gen_score >= SPATIAL_DECISION_THRESHOLD else "Forged"
    forg_decision = "Genuine" if forg_score >= SPATIAL_DECISION_THRESHOLD else "Forged"

    gen_correct = (gen_decision == "Genuine")
    forg_correct = (forg_decision == "Forged")

    return {
        "gen_score": gen_score,
        "forg_score": forg_score,
        "gen_decision": gen_decision,
        "forg_decision": forg_decision,
        "gen_correct": gen_correct,
        "forg_correct": forg_correct,
        "margin": gen_score - forg_score,
    }


def main():
    downloads_dir = os.path.expanduser("~/Downloads")
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_with_utsig_path = os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth")
    model_without_utsig_path = os.path.join(MODEL_WEIGHTS_DIR,
                                            "spatial_cnn_without_utsig.pth")

    model_with = load_model("spatial", model_with_utsig_path, device=device) if os.path.exists(model_with_utsig_path) else None
    model_without = load_model("spatial", model_without_utsig_path, device=device) if os.path.exists(model_without_utsig_path) else None


    if model_with is None and model_without is None:
        print("Error: Neither model weights file exists.")
        return


    # Custom writer definitions from ~/Downloads
    writer_sets = {
        "Writer H (English)": {
            "refs": [os.path.join(downloads_dir, "Hrefrence.jpg")],
            "gen": os.path.join(downloads_dir, "HGenuie.jpg"),
            "forg": os.path.join(downloads_dir, "Hforged.jpg"),
        },
        "Writer M (Arabic)": {
            "refs": [os.path.join(downloads_dir, "Mrefrence.jpg")],
            "gen": os.path.join(downloads_dir, "Mgenuie.jpg"),
            "forg": os.path.join(downloads_dir, "Mforged.jpg"),
        },
        "Writer Omar (Arabic)": {
            "refs": [
                os.path.join(downloads_dir, "Omarrefrence.jpg"),
                os.path.join(downloads_dir, "Omarrefrnce.jpg"),
                os.path.join(downloads_dir, "Omarrefrnce3.jpg"),
                os.path.join(downloads_dir, "OmarRefrnce.jpg"),
            ],
            "gen": os.path.join(downloads_dir, "OmarGenuie.jpg"),
            "forg": os.path.join(downloads_dir, "OmarForged.jpg"),
        },
    }

    print("=" * 80)
    print("CUSTOM SIGNATURES EVALUATION: WITH UTSIG vs WITHOUT UTSIG")
    print(f"Decision Threshold: tau = {SPATIAL_DECISION_THRESHOLD:.4f}")
    print("=" * 80)

    for writer_name, data in writer_sets.items():
        refs = [r for r in data["refs"] if os.path.exists(r)]
        gen = data["gen"]
        forg = data["forg"]

        if not refs or not os.path.exists(gen) or not os.path.exists(forg):
            print(f"Skipping {writer_name} — files missing.")
            continue

        print(f"\n[{writer_name}]")
        print(f"  Reference Samples: {len(refs)}")
        if model_with:
            res_with = evaluate_writer_samples(model_with, refs, gen, forg, device=device)
            print(f"  --- Model WITH UTSig (spatial_cnn.pth) ---")
            print(f"    Genuine Probe Score:  {res_with['gen_score']:.4f} -> {res_with['gen_decision']} ({'CORRECT' if res_with['gen_correct'] else 'WRONG'})")
            print(f"    Forged Probe Score:   {res_with['forg_score']:.4f} -> {res_with['forg_decision']} ({'CORRECT' if res_with['forg_correct'] else 'WRONG'})")
            print(f"    Score Delta (Gen-Forg): {res_with['margin']:+.4f}")
        else:
            print(f"  --- Model WITH UTSig (spatial_cnn.pth): Not Loaded ---")

        if model_without:
            res_without = evaluate_writer_samples(model_without, refs, gen, forg, device=device)
            print(f"  --- Model WITHOUT UTSig (spatial_cnn_without_utsig.pth) ---")
            print(f"    Genuine Probe Score:  {res_without['gen_score']:.4f} -> {res_without['gen_decision']} ({'CORRECT' if res_without['gen_correct'] else 'WRONG'})")
            print(f"    Forged Probe Score:   {res_without['forg_score']:.4f} -> {res_without['forg_decision']} ({'CORRECT' if res_without['forg_correct'] else 'WRONG'})")
            print(f"    Score Delta (Gen-Forg): {res_without['margin']:+.4f}")
        else:
            print(f"  --- Model WITHOUT UTSig (spatial_cnn_without_utsig.pth): Training in Progress ---")



if __name__ == "__main__":
    main()
