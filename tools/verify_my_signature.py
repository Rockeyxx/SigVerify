"""Run your own signature images through the shipped spatial model.

A hands-on check of the deployed decision path — the same `verify()` the GUI
calls, not the evaluation harness. Prints the raw similarity score alongside the
decision, because the score is what tells you whether a call was marginal or
clear-cut; the decision alone hides that.

    python tools/verify_my_signature.py \
        --references mine/ref1.png mine/ref2.png \
        --probes mine/genuine.png mine/forged.png

Spatial only. The temporal branch needs a pen trajectory (x, y, pressure at
100 Hz) which a photograph cannot supply.
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (MODEL_WEIGHTS_DIR, NUM_ENROLLMENT_REFERENCES,
                             SPATIAL_DECISION_THRESHOLD)
from src.verification import embed, load_model, similarity_from_distance, verify


def crop_to_ink(path, destination):
    """
    Turn a phone photo into something resembling the scans the model trained on.

    Three things a raw photo gets wrong, all of which sink the comparison:

    1. **EXIF rotation.** `cv2.imread` ignores the orientation tag while every
       photo viewer honours it, so a portrait shot silently arrives 90 degrees
       off. The model has no rotation invariance, so this alone is fatal.
    2. **Framing.** The signature may be a fifth of the frame, surrounded by desk,
       shadow and other writing. The pipeline resizes the whole image to 155x220,
       so an uncropped photo spends most of its canvas on background.
    3. **Lighting.** Otsu thresholding flattens shadow and paper texture into the
       clean two-tone the training scans have.

    Ink is found by **saturation**, not darkness: a blue or black pen is far more
    saturated than paper, whereas a shadow is just as dark as ink and would be
    picked up by any brightness threshold.

    Returns the destination path.
    """
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    grey_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Saturated blue/black ink on plain paper. Strict, and preferred when it
    # fires, because saturation ignores shadow entirely.
    ink = ((hsv[:, :, 1] > 60) & (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 160)).astype(np.uint8)

    if ink.sum() < 800:
        # Pale ink on tinted paper — a light blue biro on cream reads as almost
        # unsaturated, so the strict test finds nothing.
        #
        # "Darker than the page median" is NOT a safe fallback: the desk behind
        # the paper, and the shadow along its edge, are far darker than any ink,
        # so the bounding box swallows the whole photo and Otsu then erases the
        # signature. Locate the sheet of paper first and search only inside it.
        bright = cv2.threshold(grey_full, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if count < 2:
            raise ValueError(f"No page found in {path}.")
        page = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # Erode so the shaded rim of the sheet cannot masquerade as a stroke.
        mask = cv2.erode((labels == page).astype(np.uint8),
                         np.ones((15, 15), np.uint8))

        paper = float(np.median(grey_full[mask > 0]))
        spread = float(grey_full[mask > 0].std())
        ink = ((mask > 0) & (grey_full < paper - 1.5 * spread)).astype(np.uint8)
        print(f"  {os.path.basename(path)}: pale ink, searching inside the page only "
              f"(paper level {paper:.0f}, spread {spread:.0f})")

    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # Union of every sizeable blob, not just the largest — a signature whose
    # strokes do not touch would otherwise be cropped down to one stroke.
    blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 800]
    if not blobs:
        raise ValueError(
            f"No pen ink found in {path}. The detector looks for saturated blue "
            f"or black strokes; a pencil or a very pale scan will not register.")

    left = min(stats[i, cv2.CC_STAT_LEFT] for i in blobs)
    top = min(stats[i, cv2.CC_STAT_TOP] for i in blobs)
    right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in blobs)
    bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in blobs)
    pad = int(0.08 * max(right - left, bottom - top))

    grey = grey_full[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad]
    binary = cv2.threshold(cv2.GaussianBlur(grey, (5, 5), 0), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(destination, binary)
    print(f"  cropped {os.path.basename(path)}: {right - left}x{bottom - top} "
          f"(aspect {(right - left) / (bottom - top):.2f}) -> {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", nargs="+", required=True,
                        help="Enrolled genuine signatures for this person. The "
                             "measured numbers assume 5; fewer is a harder task.")
    parser.add_argument("--probes", nargs="+", required=True,
                        help="Signatures to test. Name them so you know which "
                             "you expect to be genuine and which forged.")
    parser.add_argument("--weights", default=os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop", action="store_true",
                        help="Treat the inputs as phone photos: fix EXIF rotation, "
                             "crop to the ink, and binarise. Skip this for images "
                             "that are already tight, clean scans.")
    parser.add_argument("--crop-dir", default=None,
                        help="Where to write cropped images (default: a temp dir). "
                             "Give a real path to inspect them — a bad crop is the "
                             "most common reason a genuine signature is rejected.")
    args = parser.parse_args()

    if args.crop:
        crop_dir = args.crop_dir or tempfile.mkdtemp(prefix="sigcrop_")
        os.makedirs(crop_dir, exist_ok=True)
        print(f"Cropping into {crop_dir}")

        def prepare(paths):
            return [crop_to_ink(p, os.path.join(
                crop_dir, os.path.splitext(os.path.basename(p))[0] + ".png"))
                for p in paths]

        args.references = prepare(args.references)
        args.probes = prepare(args.probes)
        print()

    model = load_model("spatial", args.weights, args.device)
    reference_embeddings = [embed(model, path, "spatial", args.device)
                            for path in args.references]
    print(f"Enrolled {len(reference_embeddings)} reference(s) from "
          f"{', '.join(os.path.basename(p) for p in args.references)}")
    if len(reference_embeddings) < NUM_ENROLLMENT_REFERENCES:
        print(f"  NOTE: the reported accuracy assumes "
              f"{NUM_ENROLLMENT_REFERENCES} references. With fewer, "
              "expect worse — this is the harder pair-scoring regime.")
    # This script is spatial-only, so verify() decides against the spatial
    # threshold. Printing the fused one showed a negative margin beside a
    # "Genuine" verdict for any score between the two (0.6299 to 0.8196).
    print(f"Threshold t = {SPATIAL_DECISION_THRESHOLD:.4f}\n")

    stacked = np.vstack(reference_embeddings)
    for path in args.probes:
        result = verify(spatial=(model, reference_embeddings, path), device=args.device)
        # Recomputed here only so the raw numbers can be shown; verify() already
        # did this internally to reach its decision.
        distance = float(np.linalg.norm(
            stacked - embed(model, path, "spatial", args.device)[None, :], axis=1).min())
        score = similarity_from_distance(distance)

        flags = " [DEGRADED — reduced evidence]" if result["degraded"] else ""
        print(f"{os.path.basename(path)}")
        print(f"  decision   {result['decision']}{flags}")
        print(f"  confidence {result['confidence']:.1f}%")
        print(f"  score      {score:.4f}  (nearest-reference distance {distance:.4f})")
        print(f"  margin     {score - SPATIAL_DECISION_THRESHOLD:+.4f} from the threshold\n")


if __name__ == "__main__":
    main()
