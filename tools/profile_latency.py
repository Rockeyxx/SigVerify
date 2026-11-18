"""Profile the end-to-end verification decision against latency constraints.

Evaluates performance under two conditions:
  * first Enter: a fresh process imports and calls warm_up(), as the app does when it
    opens (reported, not budgeted); then the budgeted clock runs from Enter, after the
    references are submitted, through enrolling them and one multimodal verify.
  * warm: verification against stored references with the models already in memory.
"""

import argparse
import os
import resource
import subprocess
import sys
import time


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


def measure_first_enter(tmpdir):
    """
    Time the first Enter in a freshly opened app: the budgeted path.

    A new process imports, builds `SignatureService` and calls `warm_up()`, as the app
    does when its window opens; that is reported as `startup`, outside the budget. The
    clock for `enter_to_result` starts after the references are submitted and covers
    enrolling them and one `multimodal` verify.
    """
    import json

    probe = _synthetic_probe(tmpdir)
    script = (
        "import json,sys,time;t0=time.perf_counter();"
        "sys.path.insert(0,%(root)r);"
        "from src.verification import SignatureService;"
        "from tools.profile_latency import _synthetic_trajectory as pen;"
        "t1=time.perf_counter();s=SignatureService();s.warm_up();"
        "t2=time.perf_counter();ready_at=time.time();"
        "refs=[pen() for _ in range(%(k)d)];probe_pen=pen();"
        "t3=time.perf_counter();"
        "e=s.enroll_multimodal([%(probe)r]*%(k)d,refs);t4=time.perf_counter();"
        "r=s.verify(mode='multimodal',probe_image=%(probe)r,probe_trajectory=probe_pen,"
        "reference_spatial=e['spatial_embeddings'],reference_temporal=e['temporal_embeddings'],"
        "reference_spatial_origin=e.get('spatial_origin','scan'));t5=time.perf_counter();"
        "print(json.dumps({'decision':r['decision'],'routing':r['routing'],"
        "'references':{'spatial':len(e['spatial_embeddings']),'temporal':len(e['temporal_embeddings'])},"
        "'imports':t1-t0,'warm_up':t2-t1,'enrol':t4-t3,'verify':t5-t4,"
        "'enter_to_result':t5-t3,'ready_at':ready_at}),flush=True)"
    ) % {"root": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         "probe": probe, "k": NUM_ENROLLMENT_REFERENCES}
    start = time.time()
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, check=True)
    run = json.loads(out.stdout.strip().splitlines()[-1])
    run["startup"] = run.pop("ready_at") - start
    return run


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
                        help="Skip the fresh-process first-Enter measurement")
    args = parser.parse_args(argv)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cold = None if args.skip_cold else measure_first_enter(tmpdir)
        median, worst, threads = measure_warm(args.branch, args.repeats, tmpdir)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    def verdict(value, ceiling):
        return "PASS" if value < ceiling else "FAIL"

    print(f"branch={args.branch}  repeats={args.repeats}  "
          f"torch threads={threads} (cap {INFERENCE_NUM_THREADS})")
    print("-" * 62)
    width = 44
    if cold is not None:
        k = NUM_ENROLLMENT_REFERENCES
        label = f"Enter to fused result, {k}+{k} references"
        print(f"  {label:<{width}}{cold['enter_to_result']:7.3f} s   "
              f"{verdict(cold['enter_to_result'], LATENCY_SLA_SECONDS)}")
        print(f"    first Enter after the app opened: enrol {cold['enrol']:.2f} s, "
              f"verify {cold['verify']:.2f} s")
        print(f"  {'app start to ready (not budgeted)':<{width}}{cold['startup']:7.3f} s")
        print(f"    imports {cold['imports']:.2f} s, warm_up {cold['warm_up']:.2f} s")
    print(f"  {'verify() on stored references, median':<{width}}{median:7.3f} s   "
          f"{verdict(median, LATENCY_SLA_SECONDS)}")
    label = f"verify() on stored references, worst of {args.repeats}"
    print(f"  {label:<{width}}{worst:7.3f} s   "
          f"{verdict(worst, LATENCY_SLA_SECONDS)}")
    print(f"  {'peak RSS':<{width}}{peak_gb:7.3f} GB  "
          f"{verdict(peak_gb, INFERENCE_RAM_CEILING_GB)}")
    print("-" * 62)
    print(f"  SLA {LATENCY_SLA_SECONDS} s / {INFERENCE_RAM_CEILING_GB} GB "
          f"(config/settings.py)")

    breached = [v for v in (cold and cold["enter_to_result"], median, worst)
                if v is not None and v >= LATENCY_SLA_SECONDS]
    return 1 if breached or peak_gb >= INFERENCE_RAM_CEILING_GB else 0


if __name__ == "__main__":
    raise SystemExit(main())
