"""Profile the end-to-end verification decision against the latency SLA.

AGENTS.md and docs/rules.md both state a hard constraint — an end-to-end
verification decision in **< 3.0 s** on consumer hardware, inside a **4 GB** RSS
ceiling, on CPU only. Until this script existed, nothing in the repository
measured either one: `scripts/benchmark.py` reports EER and Cohen's d, never
wall-clock time. The constraint was documented and unenforced.

Two numbers matter, and they answer different questions:

  * **cold** — a fresh process: interpreter start, `import torch`, weight load,
    then one verification. This is what a subprocess-per-call deployment pays on
    every single decision, and it is dominated by the torch import, not by the
    model.
  * **warm** — one verification with the models already resident. This is what
    the persistent PyQt5 application pays per decision once it has started.

Both are reported because the gap between them is large, and which one the SLA
must be measured against is a deployment decision, not a measurement detail.

    python tools/profile_latency.py
    python tools/profile_latency.py --repeats 20 --branch fused
"""

import argparse
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    INFERENCE_NUM_THREADS,
    INFERENCE_RAM_CEILING_GB,
    LATENCY_SLA_SECONDS,
    MODEL_WEIGHTS_DIR,
    NUM_ENROLLMENT_REFERENCES,
    TEMPORAL_INPUT_FEATURES,
)


def _synthetic_probe(tmpdir):
    """A signature-shaped image that clears the quality gate, so no dataset is required."""
    import cv2
    import numpy as np

    image = np.full((400, 400), 255, dtype=np.uint8)
    for x in range(50, 350, 5):
        cv2.line(image, (x, 100), (x + 30, 300), 0, 2)
    path = os.path.join(tmpdir, "probe.png")
    cv2.imwrite(path, image)
    return path


def _synthetic_trajectory(length=200):
    """A gate-passing trajectory at whatever width `TEMPORAL_CHANNELS` selects."""
    import numpy as np

    rng = np.random.default_rng(0)
    columns = [rng.standard_normal((length, 2)),
               rng.uniform(0.0, 1.0, size=(length, 1))]
    for _ in range(TEMPORAL_INPUT_FEATURES - 3):
        columns.append(np.full((length, 1), 0.1))
    return np.hstack(columns).astype(np.float32)


def measure_cold():
    """Wall-clock for a fresh interpreter that imports torch and loads a model.

    Measured in a subprocess because an already-warm import cannot be un-imported,
    so timing it in-process would report a number no real cold start ever sees.
    """
    script = (
        "import time;t=time.perf_counter();"
        "import torch;"
        "import sys;sys.path.insert(0,%r);"
        "from src.verification import load_model;"
        "load_model('spatial', %r, 'cpu');"
        "print(time.perf_counter()-t)"
    ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth"))
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def measure_warm(branch, repeats, tmpdir):
    """Median and worst-case wall-clock for one `verify()` with models resident."""
    import numpy as np
    import torch

    from src.verification import embed, load_model, verify

    probe = _synthetic_probe(tmpdir)
    kwargs = {}

    if branch in ("spatial", "fused"):
        model = load_model("spatial", os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth"), "cpu")
        references = [embed(model, probe, "spatial")
                      for _ in range(NUM_ENROLLMENT_REFERENCES)]
        kwargs["spatial"] = (model, references, probe)

    if branch in ("temporal", "fused"):
        weights = os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm.pth")
        model = load_model("temporal", weights, "cpu")
        trajectory = _synthetic_trajectory()
        references = [embed(model, trajectory, "temporal")
                      for _ in range(NUM_ENROLLMENT_REFERENCES)]
        kwargs["temporal"] = (model, references, trajectory)

    verify(**kwargs)                                    # warm the thread pools
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        verify(**kwargs)
        timings.append(time.perf_counter() - start)

    return float(np.median(timings)), float(max(timings)), torch.get_num_threads()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", choices=["spatial", "temporal", "fused"], default="fused")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-cold", action="store_true",
                        help="Skip the subprocess cold-start measurement")
    args = parser.parse_args(argv)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cold = None if args.skip_cold else measure_cold()
        median, worst, threads = measure_warm(args.branch, args.repeats, tmpdir)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    def verdict(value, ceiling):
        return "PASS" if value < ceiling else "FAIL"

    print(f"branch={args.branch}  repeats={args.repeats}  "
          f"torch threads={threads} (cap {INFERENCE_NUM_THREADS})")
    print("-" * 62)
    if cold is not None:
        print(f"  cold start (import torch + load + verify) "
              f"{cold:7.3f} s   {verdict(cold, LATENCY_SLA_SECONDS)}")
    print(f"  warm verify(), median                     {median:7.3f} s   "
          f"{verdict(median, LATENCY_SLA_SECONDS)}")
    print(f"  warm verify(), worst of {args.repeats:<3d}               {worst:7.3f} s   "
          f"{verdict(worst, LATENCY_SLA_SECONDS)}")
    print(f"  peak RSS                                  {peak_gb:7.3f} GB  "
          f"{verdict(peak_gb, INFERENCE_RAM_CEILING_GB)}")
    print("-" * 62)
    print(f"  SLA {LATENCY_SLA_SECONDS} s / {INFERENCE_RAM_CEILING_GB} GB "
          f"(config/settings.py)")

    breached = [v for v in (cold, median, worst) if v is not None and v >= LATENCY_SLA_SECONDS]
    return 1 if breached or peak_gb >= INFERENCE_RAM_CEILING_GB else 0


if __name__ == "__main__":
    raise SystemExit(main())
