"""Verify signature images using the spatial branch model."""
import argparse
import os
import shutil
import tempfile

import cv2
import numpy as np


from config.settings import (MODEL_WEIGHTS_DIR, NUM_ENROLLMENT_REFERENCES,
                             SPATIAL_DECISION_THRESHOLD, SPATIAL_INPUT_SIZE)
from src.gui.image_cleanup import crop_to_ink as app_crop_to_ink
from src.verification import (_as_spatial_input, embed, load_model,
                              similarity_from_distance, verify)


def dump_model_input(path, destination):
    """Save the preprocessed tensor array for an input image as a PNG file."""
    array = _as_spatial_input(path)[0]
    cv2.imwrite(destination, np.clip(array * 255.0, 0, 255).astype(np.uint8))
    ink = float((array > 0.1).mean() * 100)
    print(f"  model input {os.path.basename(path)}: {array.shape[1]}x{array.shape[0]} "
          f"(WxH), ink covers {ink:.1f}% of the canvas -> {destination}")
    return destination


def crop_to_ink(path, destination):
    """Crop a phone photo via app_crop_to_ink and save to destination."""
    shutil.move(app_crop_to_ink(path), destination)  # skylos: ignore[SKY-D215] operator-supplied --crop-dir
    print(f"  cropped {os.path.basename(path)} -> {destination}")
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
                        help="Treat the inputs as phone photos and crop them the way "
                             "the app does (EXIF rotation, ink crop, size check). Skip "
                             "this for images that are already tight, clean scans.")
    parser.add_argument("--crop-dir", default=None,
                        help="Where to write cropped images (default: a temp dir). "
                             "Give a real path to inspect them — a bad crop is the "
                             "most common reason a genuine signature is rejected.")
    parser.add_argument("--dump-model-input", metavar="DIR", default=None,
                        help="Write what the model actually receives for every "
                             f"reference and probe: {SPATIAL_INPUT_SIZE[1]}x{SPATIAL_INPUT_SIZE[0]} "
                             "(HxW), inverted so ink is bright on black. This is the "
                             "tensor, not a preview of it, and it is the fastest way "
                             "to see a crop or rotation problem.")
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

    if args.dump_model_input:
        os.makedirs(args.dump_model_input, exist_ok=True)
        print(f"Model inputs written to {args.dump_model_input}")
        for kind, paths in (("ref", args.references), ("probe", args.probes)):
            for path in paths:
                dump_model_input(path, os.path.join(
                    args.dump_model_input,
                    f"{kind}_{os.path.splitext(os.path.basename(path))[0]}.png"))
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
    print(f"Threshold t = {SPATIAL_DECISION_THRESHOLD:.4f}\n")

    stacked = np.vstack(reference_embeddings)
    for path in args.probes:
        result = verify(spatial=(model, reference_embeddings, path), device=args.device)
        # Recompute distance locally to print raw score and threshold margin.
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
