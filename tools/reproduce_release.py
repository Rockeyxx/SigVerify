"""Score, or retrain, a published release and compare with the numbers its notes published.

    python tools/reproduce_release.py --list
    python tools/reproduce_release.py v0.6.0                    # score its weights
    python tools/reproduce_release.py v0.10.0 --branch spatial
    python tools/reproduce_release.py v0.6.0 --retrain           # hours to days: run it detached

Weights download once into models/weights/releases/<tag>/ (gitignored), with their
split records where the release has them. Scoring runs the current
scripts/benchmark.py under the enrolment protocol in config/settings.py, on the
corpora in SPATIAL_TRAIN_DIRS / TEMPORAL_TRAIN_DIRS, on the device each release
was measured on (CPU-measured ones run with the GPU hidden). Run it with the
CUDA venv so both kinds are available.

--retrain checks out the commit that ran the release's recorded training
command into a git worktree under the cache, trains there, and scores the
result the same way. Releases whose notes recorded no command are refused: a
guessed recipe would reproduce nothing.
"""
import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (ENROLLMENT_TRIALS, MODEL_WEIGHTS_DIR,  # noqa: E402
                             NUM_ENROLLMENT_REFERENCES, PROJECT_ROOT, SEED,
                             SPATIAL_TRAIN_DIRS, TEMPORAL_TRAIN_DIRS)

CACHE_DIR = os.path.join(MODEL_WEIGHTS_DIR, "releases")


@dataclass(frozen=True)
class Train:
    """A recorded training command and the commit whose code ran it."""
    commit: str
    args: tuple


@dataclass(frozen=True)
class Weight:
    """
    One release asset. `published` holds the notes' figures by kind: "mean" and
    "sd" are the mean and spread over the ten enrolment draws, "pooled" is the EER
    of the representative draw, and a corpus name maps to that corpus's mean.
    `device` is where those figures were measured: the same file differs in the
    second decimal between CPU and CUDA, and each figure reproduces exactly only
    on its own device (checked for every release on 2026-09-27, ENG_REPORT M10).
    """
    branch: str
    asset: str
    published: dict
    device: str
    channels: str = None
    train: Train = None


_V05_TEMPORAL = {"mean": 10.26, "sigcomp_Dutch": 4.19, "sigcomp_Chinese": 8.11, "svc": 17.46}
_DT_RECIPE = Train("4a9f6ed", ("--branch", "temporal", "--temporal-channels", "dt", "--seed", "45"))

# From the published notes, `gh release view <tag>`. Commits are where each recipe
# runs; tags v0.1.0 to v0.7.0 point at older code (ENG_REPORT M9).
RELEASES = {
    "v0.1.0": (Weight("spatial", "archive_spatial_cnn_compact_12.31eer.pth", {"mean": 12.31, "sd": 0.96}, "cuda"),),
    "v0.2.0": (Weight("spatial", "archive_spatial_cnn_wide_11.78eer.pth", {"mean": 11.78}, "cuda"),),
    "v0.3.0": (Weight("spatial", "spatial_cnn_aug_lr.pth", {"mean": 11.23}, "cuda"),),
    "v0.4.0": (Weight("spatial", "spatial_cnn_pairfix.pth", {"mean": 10.58}, "cuda"),),
    "v0.5.0": (
        Weight("spatial", "spatial_cnn.pth",
               {"mean": 10.10, "sd": 0.38, "cedar": 0.00, "bhsig_Hindi": 4.30,
                "bhsig_Bengali": 12.07, "utsig": 30.74}, "cuda",
               train=Train("af67d37", ("--branch", "spatial", "--seed", "42", "--lr-schedule",
                                       "plateau", "--random-impostor-ratio", "0.5"))),
        Weight("temporal", "temporal_lstm.pth", _V05_TEMPORAL, "cuda", channels="xyp"),
    ),
    "v0.6.0": (
        Weight("spatial", "spatial_cnn.pth",
               {"mean": 8.59, "cedar": 0.00, "bhsig_Hindi": 3.64, "bhsig_Bengali": 8.64, "utsig": 30.26}, "cpu",
               train=Train("96e297b", ("--branch", "spatial", "--seed", "42", "--lr-schedule", "plateau",
                                       "--margin", "2.0", "--random-impostor-ratio", "0.5"))),
        Weight("temporal", "temporal_lstm.pth", _V05_TEMPORAL, "cuda", channels="xyp"),
    ),
    "v0.7.0": (
        Weight("spatial", "spatial_cnn.pth", {"pooled": 8.45}, "cuda"),
        Weight("temporal", "temporal_lstm.pth", {"mean": 9.41}, "cuda", channels="xyp"),
    ),
    "v0.8.0": (
        Weight("spatial", "spatial_cnn.pth", {"mean": 8.59, "sd": 0.58, "cedar": 0.00,
                                              "bhsig_Hindi": 3.64, "bhsig_Bengali": 8.64}, "cpu"),
        Weight("temporal", "temporal_lstm.pth", {"mean": 7.81, "sd": 0.63}, "cuda", channels="dt", train=_DT_RECIPE),
    ),
    "v0.9.0": (
        Weight("spatial", "spatial_cnn.pth",
               {"mean": 8.72, "sd": 0.42, "bhsig_Bengali": 8.17, "bhsig_Hindi": 5.33,
                "utsig": 28.10, "cedar": 0.00}, "cuda",
               train=Train("7811f1a", ("--branch", "spatial", "--seed", "42"))),
        Weight("temporal", "temporal_lstm.pth", {"mean": 7.77, "sd": 0.67}, "cpu", channels="dt", train=_DT_RECIPE),
    ),
    "v0.10.0": (
        Weight("spatial", "spatial_cnn.pth",
               {"mean": 7.46, "sd": 0.46, "bhsig_Bengali": 6.82, "bhsig_Hindi": 3.47,
                "utsig": 26.34, "cedar": 0.00}, "cpu",
               train=Train("8bb6f5e", ("--branch", "spatial", "--seed", "42", "--split-seed", "42",
                                       "--aug-stroke-prob", "0"))),
        Weight("temporal", "temporal_lstm.pth", {"mean": 7.77, "sd": 0.67}, "cpu", channels="dt", train=_DT_RECIPE),
    ),
}


def _corpora(weight):
    return list(SPATIAL_TRAIN_DIRS if weight.branch == "spatial" else TEMPORAL_TRAIN_DIRS)


def benchmark_command(weight, weights_path):
    """The documented scoring command for one release weight."""
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "benchmark.py"),
           "--branch", weight.branch, "--weights", weights_path,
           "--data-dir", *_corpora(weight),
           "--enroll", str(NUM_ENROLLMENT_REFERENCES), "--enroll-trials", str(ENROLLMENT_TRIALS),
           "--seed", str(SEED)]
    if weight.branch == "temporal":
        cmd += ["--temporal-channels", weight.channels]
    return cmd


def benchmark_env(weight):
    """Environment for scoring: a CPU-measured release is scored with the GPU hidden."""
    env = os.environ.copy()
    if weight.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def train_command(tag, weight, out_dir):
    """(commit, argv) for retraining a weight, or ValueError when its notes recorded no command."""
    if weight.train is None:
        raise ValueError(f"{tag} {weight.branch}: its release notes recorded no training command, "
                         "so there is no recipe to reproduce (see docs/RELEASE_PLAN.md)")
    argv = ["scripts/train.py", *weight.train.args, "--data-dir", *_corpora(weight),
            "--weights-dir", out_dir, "--weights-name", weight.asset]
    return weight.train.commit, argv


def parse_benchmark(text):
    """Overall and per-corpus EERs from benchmark.py output, in percent."""
    result, last, breakdown = {"corpora": {}}, None, False
    for line in text.replace("\r", "\n").splitlines():
        draws = re.search(r"EER over \d+ draws: mean ([\d.]+)%\s+std ([\d.]+)%", line)
        if draws:
            last = (float(draws.group(1)), float(draws.group(2)))
            result.setdefault("mean", last[0])
            result.setdefault("sd", last[1])
            continue
        if "PER-DATASET BREAKDOWN" in line:
            breakdown = True
            continue
        pooled = re.match(r"\s+EER:\s+([\d.]+)%", line)
        if pooled and not breakdown:
            result.setdefault("pooled", float(pooled.group(1)))
            continue
        corpus = re.match(r"\s+(\S+)\s+\d+ writers\s+EER\s+([\d.]+)%", line)
        if corpus and breakdown and last:
            result["corpora"][corpus.group(1)] = {"mean": last[0], "sd": last[1],
                                                  "pooled": float(corpus.group(2))}
    return result


def verdict(published, measured):
    diff = round(measured - published, 2)
    return "match" if diff == 0 else f"differs by {diff:+.2f}"


def _has_cuda():
    import torch
    return torch.cuda.is_available()


def _measured(result, key):
    if key in ("mean", "sd", "pooled"):
        return result.get(key)
    return result["corpora"].get(key, {}).get("mean")


def _download(tag, weight):
    folder = os.path.join(CACHE_DIR, tag)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, weight.asset)
    if not os.path.exists(path):
        subprocess.run(["gh", "release", "download", tag, "-p", weight.asset, "-D", folder],
                       check=True, cwd=PROJECT_ROOT)
        record = os.path.splitext(weight.asset)[0] + ".split.json"
        subprocess.run(["gh", "release", "download", tag, "-p", record, "-D", folder],
                       cwd=PROJECT_ROOT, capture_output=True)  # older releases have none
    return path


def _retrain(tag, weight):
    out_dir = os.path.join(CACHE_DIR, tag, "retrained")
    commit, argv = train_command(tag, weight, out_dir)
    tree = os.path.join(CACHE_DIR, "_src", commit)
    if not os.path.isdir(tree):
        subprocess.run(["git", "worktree", "add", "--detach", tree, commit], check=True, cwd=PROJECT_ROOT)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Retraining {tag} {weight.branch} at {commit}: {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, *argv], check=True, cwd=tree)
    return os.path.join(out_dir, weight.asset)


def score(tag, weight, weights_path):
    """Run the benchmark on one weight and print published against measured."""
    cmd = benchmark_command(weight, weights_path)
    print(f"\n{tag} {weight.branch}: {weight.asset}, published figures measured on {weight.device}"
          f"\n  $ {' '.join(cmd)}", flush=True)
    if weight.device == "cuda" and not _has_cuda():
        print("  this interpreter has no GPU: expect differences in the second decimal (use venv)")
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=benchmark_env(weight))
    if run.returncode:
        print(run.stdout[-2000:] + run.stderr[-2000:])
        return False
    result = parse_benchmark(run.stdout)
    ok = True
    for key, published in weight.published.items():
        measured = _measured(result, key)
        line = verdict(published, measured) if measured is not None else "not reported"
        ok &= line == "match"
        shown = "-" if measured is None else f"{measured:.2f}"
        print(f"  {key:<16} published {published:6.2f}   measured {shown:>6}   {line}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", nargs="?", choices=list(RELEASES))
    parser.add_argument("--list", action="store_true", help="list releases and exit")
    parser.add_argument("--branch", choices=["spatial", "temporal"])
    parser.add_argument("--retrain", action="store_true",
                        help="retrain at the recipe's commit before scoring (hours to days)")
    args = parser.parse_args(argv)

    if args.list or not args.tag:
        for tag, weights in RELEASES.items():
            print(f"{tag:<8} " + "   ".join(
                f"{w.branch} {w.asset}{' (retrainable)' if w.train else ''}" for w in weights))
        return 0

    ok = True
    for weight in RELEASES[args.tag]:
        if args.branch and weight.branch != args.branch:
            continue
        path = _retrain(args.tag, weight) if args.retrain else _download(args.tag, weight)
        ok &= score(args.tag, weight, path)
    print("\nAll published numbers reproduced." if ok else
          "\nSome numbers differ.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
