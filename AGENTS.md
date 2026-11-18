# AGENTS.md — Contributor & Agent Directives

## Signature Verification System (EE 499, Team 06)

This file defines the hard architectural constraints for anyone working on
this codebase — human contributors and AI coding agents alike. These rules
are distilled from the approved Term-1 design baseline and are **non-negotiable**
unless the team explicitly authorizes a deviation.

Read this file before writing or reviewing any code in this repository.

---

## Which document gets what

**`README.md` is for a user, not a reviewer.** It carries what the system is, how
to run it, and a bare results table. Every explanation of how a number was
measured, why a design choice was made, or which experiments failed stays out
of the repository. Rules and contracts go here in `AGENTS.md`; dataset terms go in
`DATASET_LICENSES.md`.

**Do not move report content into the README.** Protocol derivations, experiment
history and rejected arms stay out of the repository, even when they explain a
number the README quotes.

**Before proposing an improvement, ask the model owner for the negative-results
record.** It lists every approach that was tested and measured worse, with the
numbers and the reason. Most ideas that get proposed again are already in it.

**`docs/` is gitignored, so never link to it from a tracked file.** The link is
dead for anyone who clones. Inline the fact instead. Full conventions are in the
`docs-style` skill.

**Design documents from the EGM plugin go in one local folder.** Briefs, drafts
and review reports are gitignored and never committed.

EGM artifact directory: .egm

---

## Drift Prevention

The following were **evaluated and deliberately excluded**. Do **not** add
them, even if they appear in the literature review or older drafts:

- **No Spatial Transformer Network (STN)** in the pipeline
- **No BioHashing / quantization / Hamming distance** — matching uses
  continuous embeddings with L2 or cosine distance
- **No SQLCipher / whole-file DB encryption** — encryption is
  application-layer AES-256-GCM only
- **No extra _captured_ channels beyond the four in the contract** — the capture
  contract is `[1, L, 4]`, `(X, Y, Pressure, dt)`. A timestamp is standard
  across digitizers and forms `dt`, whereas channels that common hardware cannot
  deliver are excluded. The XPPen Deco 640 does report tilt X and Y, so the
  capture widget keeps them in its raw event log for research, but nothing
  passes them to the model. Features **derived** from these four channels
  inside the model are allowed and expected (see Temporal Branch below).
- **Research exception: the tilt channel modes.** `dt_tilt0`, `dt_tilt1` (6
  channels) and `dt_tilt_flag` (7) add SVC2004's azimuth and altitude to `dt`,
  with a constant fill for corpora that have no tilt. Use them only through the
  `--temporal-channels` flag of `scripts/train.py` and `scripts/benchmark.py`.
  `TEMPORAL_CHANNELS` stays `dt`, and the capture and `verify()` contract stays
  `[L, 4]`. This ban is lifted only if a tilt arm passes its pre-registered rule
  against the shipped `dt` family (six paired seeds) and is shipped. Until then,
  no tilt checkpoint goes through `verify()`, the gate or the config-driven
  scorers, and no document may claim that tilt helps the shipped model.
- **No cloud APIs, no remote inference, no discrete GPU at inference**

---

## Hard Constraints

| Constraint                      | Value                                             |
| ------------------------------- | ------------------------------------------------- |
| Inference RAM ceiling           | ≤ 4 GB                                            |
| End-to-end verification latency | < 3.0 s, Enter to fused result (see below)        |
| Execution                       | Local only — no cloud, no dedicated GPU           |
| Platform support                | Windows + Arch Linux (x86_64 and ARM64)           |
| Native build steps              | None — all dependencies must have prebuilt wheels |

The latency budget starts when the user presses Enter after submitting
`NUM_ENROLLMENT_REFERENCES` scans, as many pen signings and one probe of each, and
ends at the fused result. It covers enrolling those references and one `multimodal`
verify on CPU. Starting the app is outside it: the app must call
`SignatureService.warm_up()` once when it opens, on a background thread, so the
imports, the model loads and the first forward pass happen before Enter. Keep one
`SignatureService` for the whole session, because a second instance loads the models
again. The budget must stay well under 3.0 s, because the front end needs time too.
`tools/profile_latency.py` reports it on its first row and `tests/test_latency.py`
asserts it off CI. Measured on an 8-core laptop CPU: 0.5 s from Enter (0.8 s on 2
cores, 1.7 s on one), and 1.35 s to start the app, 1.0 s of it importing PyTorch,
reported but not budgeted (1.9 s on the first launch after a reboot). A `verify()` against
stored references takes 0.08 s (0.28 s on one core); never quote it as the budget.

---

## Architecture Rules

### Spatial Branch (Offline)

- Input tensor: **`[1, 1, 155, 220]`** (8-bit grayscale), resized **directly** onto
  the target, with no square pad. This beat the earlier `[1, 1, 256, 256]`
  square-padded input on every dataset (UTSig 3.38 points better, pooled 1.41)
  and trains about twice as fast.
  Many signature images are wide ribbons (around 1:3.5), so square padding left ~70%
  of the canvas blank and lost valuable stroke resolution. `SPATIAL_SQUARE_PAD` in
  `config/settings.py` keeps both options configurable.
- Pixel polarity: **inverted — background 0, ink high.** Ink covers only ~1%
  of the canvas, so non-inverted images feed convolutions a large constant background
  offset that provides no identity signal. Inverting ink sharply improves separation.
- Backbone: **Siamese CNN** → **512-d** embedding. The _model type_ is fixed;
  the embedding width is a tunable measured on held-out EER. Changing it is also
  a **database contract change** — 512-d stores as 2048 bytes, 128-d as 512 —
  so it requires re-enrolling every stored template.
- Trunk width is a **tunable**, the model class is not. `SPATIAL_TRUNKS` in
  `src/models/siamese.py` holds the convolutional stacks (`compact` is this
  project's 32-64-128-256-256; `wide` is the Dey et al. reference's 96×11×11 stem
  into 256, 384, 256). Selecting one is still a Siamese CNN with contrastive
  loss, so it is inside the architecture constraint. Run trunk, dropout and FC
  head as **separate** runs — bundled, their attribution is lost.
- The `wide` trunk reached 11.78% pooled EER against 12.31% for `compact`. Adding
  Dey's convolution dropout cost 6.78 points, and adding their FC-1024 head on top
  of that collapsed every signature to one embedding before it partly recovered.
  **Do not set `SPATIAL_CONV_DROPOUT` or `SPATIAL_FC_HIDDEN` without a new measured
  reason.** With about 300 writers the problem is too little data, not too much
  capacity, so regularizing removes capacity the model needs. The shipped model
  uses `SPATIAL_TRUNK = "wide"` without that dropout and without the FC-1024 head.
- **A checkpoint carries its own writer split.** `scripts/train.py` writes
  `<weights>.split.json` next to the weights (`data_dirs`, `split_seed`,
  `holdout_writers`, and the `train` / `val` / `test` writer IDs). `benchmark.py`
  and `evaluate.py` refuse to score a checkpoint on any writer from its stored
  `train` set, or on a held-out set that is not the stored one (`--writers` may
  pick a subset of it). This check would have stopped the `exp_sc11` leak at the
  first benchmark: it was trained on one 65-writer list and scored on another that
  shared 34 writers with its training set. A checkpoint
  without a record is scored with a printed warning, never silently. **Ship the
  `.split.json` next to every `.pth` in a release, and never edit a record by hand
  to make a benchmark pass.** For a spatial checkpoint the record also stores
  `input_size`, because the image size cannot be read from the weights (adaptive
  pooling accepts any size). A checkpoint trained with the sweep-only override
  `SIGVER_SPATIAL_INPUT_SIZE=WxH` is scored only with the same value, and
  `benchmark.py` refuses any other. The override exists for the input-size sweep
  only. Shipped weights train and serve at `SPATIAL_INPUT_SIZE`.
- **A checkpoint carries its own architecture.** `SpatialSiamese.from_state_dict`
  recovers trunk, embedding width and FC head from the saved parameter shapes,
  and every load path (`benchmark.py`, `evaluate.py`, `src/verification.py`) goes
  through it. Never add a load path that builds the model from `config/` instead:
  config points at whatever is being trained _now_, and scoring an archived
  checkpoint under the wrong architecture yields a plausible number that means
  nothing.
- **Training-time augmentation is on, but affine warps only**
  (`SPATIAL_AUGMENT = True`, `SPATIAL_AUG_STROKE_PROB = 0.0`). Without
  augmentation every spatial run was best at epoch 1, because the model saw the
  same pixels every epoch. Warps plus stroke-width jitter moved the best epoch to
  6 and pooled EER from 11.78% to 11.19%. That bundle was measured as one switch
  and hid a part that hurt: turning the stroke dilate/erode jitter **off** took
  pooled EER from 8.69% to **7.46%**, with no corpus getting worse. Forgery
  detection depends on fine stroke width, and that jitter is the warp that erases
  it. Keep the affine ranges small, keep the jitter at zero, and never judge a
  group of augmentations only by its total effect. Warp each side of a pair
  **independently**, or the distance between them does not change and the metric
  learns nothing.
- **LR schedule: optional, not part of the base recipe.** `--lr-schedule plateau`
  (halving learning rate after stalled validation epochs) can reduce evaluation variance,
  but defaults to `none`. Base weights are trained with a constant learning rate.
  Pass the flag explicitly if you want to experiment with adaptive schedules.
- Normalization: **GroupNorm or LayerNorm** — never BatchNorm (pair-sampled batches)
- Quality gate: effective-resolution proxies (bounding-box pixel size +
  Laplacian-variance sharpness). Do **not** gate on the file's DPI tag

### Temporal Branch (Online)

- Input tensor: **`[1, L, 4]`** — `(X, Y, Pressure, dt)`, variable length `L`.
  The width is **not** a literal anywhere in the code: it is
  `TEMPORAL_INPUT_FEATURES`, derived from `TEMPORAL_CHANNELS` in
  `config/settings.py` (`xyp` → 3, `dt` → 4, `tilt` → 5, and the research-only
  `dt_tilt0`/`dt_tilt1` → 6, `dt_tilt_flag` → 7, see Drift Prevention). Flipping
  that one setting moves the gate, the parsers and the test suite together, so a
  checkpoint trained under any of them stays reproducible. `dt` is what
  ships and is what `models/weights/temporal_lstm.pth` was trained as
- **Derived dynamics:** the model appends one per-frame delta per input channel
  to form a `[1, L, TEMPORAL_INPUT_FEATURES x 2]` tensor _internally_, before the
  LSTM — **8** under the shipped `dt` contract (4 channels); 6 was the legacy
  3-channel (`xyp`) value. Velocity is the most
  discriminative feature family in the online-verification literature: a forger
  can copy the shape of a signature while tracing it far more slowly and evenly.
  This costs the capture device nothing, since deltas come from consecutive
  frames of the three channels it already reports. Feeding raw coordinates alone
  produced a collapsed model with train loss pinned at the theoretical
  no-discrimination value (recorded as P-002)
- Backbone: **Siamese LSTM** → **128-d** embedding
- **Training:** pressure-channel dropout is **off** (`PRESSURE_DROPOUT_RATE = 0`).
  Experiments showed that pressure dropout degraded discrimination both on standard
  inputs (10.26% vs 12.73% with dropout) and on pressure-blind inputs (20.32% vs 23.32%).
  It lost on both inputs at once, so no robustness argument is left for it.
  Turning it back on needs a measurement that beats these numbers, not an appeal
  to robustness.
- **Inference:** if no pressure sensor is available, impute `P = 1.0` and set the
  degraded-mode flag in the GUI.
- **Tilt (azimuth and altitude) is not in the shipped model.** On SVC2004 Task 2
  alone (40 writers, one seed, no `dt`, 3-channel control 19.55% at K=10),
  adding azimuth and altitude reached 11.17%. That run says nothing about tilt
  on top of `dt` with SigComp2011, which has no tilt, in training. The XPPen
  Deco 640 reports tilt X and Y in degrees (about ±60), not SVC2004's azimuth
  and altitude, so a tilt model would also need a conversion before it could
  run on the device. Tilt stays banned from the contract except for the
  research modes under Drift Prevention. `button` is almost always 1.0 and
  carries no information.
- `dt` is the time between two sampled points. SigComp2011 has no clock column,
  but it samples at a fixed 200 Hz and records pen-up travel as `force = 0`, which
  `load_sigcomp2011_file` drops. The length of the dropped run is the elapsed
  time, so `dt` is a real measured duration in both pen corpora, not a constant that
  only tags which corpus a sample came from. Over all 5295 signatures the two
  corpora have the same median `dt` (0.1000) and only 0.364 pooled sd of
  separation. Over six paired
  seeds, `dt` improved pooled EER from 10.45% to 8.97% (t = 3.25). Attention
  pooling alone is worth 1.16 points, but the two **do not stack** (9.53% together, worse
  than `dt` alone), because they recover the same information.
- **Training reads SVC2004 and DeepSignDB as float32, and every scorer reads them
  exactly.** float32 cannot hold every millisecond past 2^24 ms, and SVC2004's
  clock runs to 7.5 x 10^7 ms, so on most of its files training sees 10 ms steps
  as 8, 12 or 16 ms. That rounding is the only noise the shipped `dt` recipe
  trains with, and the `dt` gain above depends on it: retrained on exact
  timestamps over six paired seeds, the recipe reads 11.76% in-domain against
  8.97% (worse on every seed), no better than `xyp`, and ±2 ms of explicit jitter
  came within noise of 8.97% without passing its adoption rule.
  `TEMPORAL_TRAIN_FLOAT32_PARSE` in `config/settings.py` holds the training read;
  with it, seed 45 retrains to the shipped `temporal_lstm.pth` tensor for tensor.
  Never turn it off without a six-seed retrain that beats these numbers. Scorers
  keep the exact read, which is what the tablet sends;
  `scripts/benchmark.py --float32-parse` exists only so
  `tools/reproduce_release.py` can reproduce the releases scored before this
  split.

- **`supports_extra_channel()` in `src/data/parsers.py` is the single authority**
  on which corpus can supply which channel, and `build_writer_samples` raises
  rather than letting a corpus quietly contribute narrower samples to one merge.
  SVC2004 Task1 is excluded — its four columns carry neither channel, and asking
  it for one used to return `[L, 3]` silently.
- **A temporal checkpoint carries its own channel count *and* whether it derives
  deltas.** `TemporalSiamese.from_state_dict` reads the width from
  `lstm.weight_ih_l0` and the delta policy from the `derive_deltas_flag` buffer,
  which is persisted in `state_dict`. Every load path uses it, so a 4- or
  5-channel experiment can never be silently scored as the 3-channel default.

  The flag is stored rather than inferred because **width alone is ambiguous**: a
  6-wide LSTM is either 3 channels with deltas appended or 6 channels without,
  and guessing wrong produces a plausible meaningless number instead of an error.
  Checkpoints saved before the flag existed have no key and default to `True`,
  which is what they were trained with.

- **`derive_deltas=False` is required for pre-derived inputs.**
  `src/data/features.py` already emits `dx`, `dy`, `ddx`, `ddy`. Appending deltas
  on top differentiates them again — `ddx` becomes a *third* derivative of raw
  tablet coordinates, which is amplified noise — and doubles the input width for
  nothing. `train.py` no longer trains derived features (the option is in commit
  3c80324); `benchmark.py --temporal-features derived` still scores such a
  checkpoint, and the flag comes from the checkpoint.

  **The bug was real; it was not the explanation.** Re-run at three seeds with
  deltas correctly off, the derived features measured **14.99% pooled** against a
  10.88% control — statistically identical to the 14.00% the buggy version
  produced. Derived features genuinely hurt this model. Do not re-litigate this
  without a new mechanism to test.

#### Capture Output Contract (device → model)

A signature pad emits more channels than the model accepts. Tablets typically
report **7**: `X, Y, timestamp, button/contact, azimuth, altitude, pressure` —
the same shape as SVC2004 Task2, which is why Task2 is the dataset that matches
the deployment device.

The target device is an **XPPen Deco 640**. Measured on the tablet: it reports
position, pressure (0 to 16383) and tilt X and Y, at 250 Hz on the host clock
with no device timestamp. The contract keeps four of these channels; tilt is
recorded in the raw event log only (research exception under Drift Prevention).

`src/capture/` is responsible for narrowing whatever the device produces down
to the **4** channels the model takes — `X, Y, Pressure, dt`, i.e. `[L, 4]`.
Widening the _captured_ contract beyond those four is banned under Drift
Prevention. Deriving further features from them inside the model is not.

**The temporal capture contract is 4 channels, `[L, 4]`:** `X, Y, Pressure, dt`.
`dt` is worth 1.49 pooled EER points (t = 3.25 over six paired seeds), and a
timestamp is one of the standard channels every digitizer already reports. Any
driver must emit `[L, 4]`; a 3-channel capture path cannot feed a shipped model.
`src/capture/tablet.py` (Windows, PyQt5 tablet events) emits `[L, 4]`, but nothing
calls it yet and it cannot feed the model as written. It has three known bugs: it
normalizes with its own placeholder instead of `parsers.normalize_trajectory()`,
it does not resample to 100 Hz (see Device timing and pressure below), and it stops recording at the first pen lift, so a
signature with several strokes keeps only the first one. Fix all three before
connecting it to `verify()`.

| Device channel          | Fate                                                                     |
| ----------------------- | ------------------------------------------------------------------------ |
| X, Y                    | **Kept** — z-scored per signature (zero mean, unit variance per axis)    |
| Pressure                | **Kept** — scaled to `[0.0, 1.0]`; absent ⇒ impute `1.0` + degraded flag |
| Timestamp               | **Kept** — as `dt`, the gap to the previous sample, clipped 100 ms → `[0,1]` |
| Button / contact state  | Dropped — it is pen-down/up, **not** a pressure substitute               |
| Azimuth, altitude, tilt | Dropped from the contract (raw log only; research modes in train/benchmark) |

- Capture **must** normalize via `src.data.parsers.normalize_trajectory()` —
  the same function the training parsers call. Any separate normalization path
  creates train/inference skew and silently invalidates every embedding.
- X/Y are standardized **per signature** so the model keys on stroke shape and
  dynamics, not on where the person signed on the pad or how large they wrote.
- Raw coordinate streams are **never persisted** — derive the embedding, store
  that, discard the stream (ISO/IEC 24745:2022).

### Loss

- **Contrastive loss:** `L = y·d² + (1−y)·max(0, m−d)²`
- Use dynamic **hard-negative mining**
- Distance metric: **L2 norm or cosine similarity** on continuous embeddings

### Fusion

```
S_final = α · S_spatial + (1 − α) · S_temporal
```

- If a modality is absent: drop that branch (α = 1 for image-only, α = 0 for dynamic-only)
- Decision: `S_final ≥ τ → Genuine`, else `Forged`
- Surface the calibrated score as a confidence percentage

### Enrollment

- Store **embeddings only** — no backpropagation, no layer recompilation
- Support multiple references per user (few-shot)
- Match probe against reference set (nearest-reference or mean — fixed at calibration)

---

## Capture Contract (for the capture and GUI layers)

This section defines the boundary contract between capture and verification modules.
Every rule below represents a data distribution assumption guaranteed during training;
violating these rules degrades verification accuracy.

Both gates are **enforced in `verify()`** — a failing probe is dropped from
fusion, not merely flagged. A single-branch verification whose only probe fails
raises, naming which gate rejected it, rather than returning a verdict built on
nothing.

### Spatial gate — `quality_gate()`

| gate | rule | constant |
| --- | --- | --- |
| Ink bounding box | `min(width, height) >= 64 px` **in the submitted image** | `MIN_BBOX_PIXELS = 64` |
| Sharpness | Laplacian variance `>= 50.0` | `MIN_LAPLACIAN_VARIANCE = 50.0` |

Gate on these proxies, never on the file's DPI tag.

### What the frontend must do before submitting an image

1. **Apply EXIF rotation.** `cv2.imread` ignores the orientation tag while every
   phone gallery honours it, so a portrait photo arrives 90° off and looks correct
   in every viewer you would check it in. The model has no rotation invariance.
   Use `PIL.ImageOps.exif_transpose`, or re-encode to strip EXIF.
2. **Crop tight to the ink.** `SPATIAL_SQUARE_PAD = False`, so the image is
   resized *straight* onto 220×155 with no padding. Desk, margin and shadow are
   canvas taken from the signature.
3. **Use one fixed capture frame for enrollment and verification.** Because
   nothing is padded, different framing stretches the signature by a different
   amount. Measured on three photos of one signature: aspect ratios 0.67, 1.18
   and 0.85 left the genuine probe at 0.6498 against a forgery's 0.6366 —
   correctly ordered but 0.0132 apart and both below τ. Scan datasets are already
   tight and consistent, so this never appears in the reported EER.
4. **Dark ink on light paper.** The pipeline inverts internally; do not pre-invert.
5. Colour is discarded at load — greyscale in is fine.

### Temporal gate — `trajectory_gate()`

The capture layer **must call `parsers.normalize_trajectory()`**. Nothing used to
check this, and the failure was silent and confident: on one SigComp writer the
same signature scored **0.9370 (Genuine)** when normalized and **0.4888 (Forged)**
in raw tablet units. A genuine user was rejected with no `degraded` flag and no
error.

| gate | rule | constant |
| --- | --- | --- |
| Length | `>= 10` points | `MIN_TRAJECTORY_POINTS = 10` |
| X / Y centred | `abs(mean) <= 0.5` per axis | `MAX_NORMALIZED_XY_MEAN = 0.5` |
| X / Y scaled | `std` within `[0.2, 5.0]` per axis | `NORMALIZED_XY_STD_RANGE` |
| Pressure range | within `[0, 1]` | — |
| `dt` range | within `[0, 1]`, when the contract is 4 channels or wider | — |
| Finite | no NaN or infinity | — |

The X/Y bounds detect the *class* of error rather than a device's particular
scale, so no per-device configuration is needed. A flat `P = 1.0` **passes**:
that is the legitimate pressure-blind device, reported as `degraded`, not
rejected.

**The width check is the one that failed before.** After the contract moved to 4
channels, the gate still required exactly 3 columns while the shipped weights
required 4. The temporal branch could not be reached through `verify()` at all:
the parser emitted `[L, 4]`, `embed()` accepted it, and the gate refused it. A
temporal-only call raised, and a multimodal call silently fell back to
spatial-only. The gate now reads the width from `TEMPORAL_INPUT_FEATURES`, never
from a literal, so the next contract change cannot break it the same way.

### Trajectory format

| requirement | value |
| --- | --- |
| Shape | `[L, 4]` float32 — X, Y, Pressure, dt. Width follows `TEMPORAL_INPUT_FEATURES` |
| Sample rate | **100 Hz** (10 ms). Resample by timestamp onto a 10 ms grid; never decimate by an integer factor |
| Pressure | scaled to [0, 1] by the **maximum the device reports**, never 1023 by default and never the file's own max |
| Pen-up samples | dropped (`P = 0`) |
| X / Y | raw device units in; `normalize_trajectory` z-scores them per signature |
| No pressure sensor | `P` imputes to 1.0 and `degraded` is set |

### Device timing and pressure (for the capture team)

Measured 2026-10-21 on e-BioSign DS1 (inside DeepSignDB): the same 65 people signed on
five devices, so the device was the only thing that changed. Three of the devices sample at
200 Hz, and each stamps time differently: one sends bursts of samples under a single
timestamp, one has a millisecond counter that wraps every second, one is clean. Their
pressure maxima were 833, 1023 and 511. Through integer decimation the model saw intervals
of 0 to 32 ms instead of 10 ms. Repairing the clock and resampling by timestamp lowered pen
EER by 0.9 to 2.8 points on four of the five devices; scaling pressure by each device's own
maximum lowered it by up to 1.4 more. The rate itself mattered less: a 125 Hz device ended
level with the 200 Hz ones, and only a 64 Hz device stayed worst. Matching specifications
did not mean matching signals.

Rules for `src/capture/`, for the XPPen and any later device:

- Resample X, Y and P onto an exact 10 ms grid by timestamp (linear interpolation). Do not
  decimate by an integer factor: that is exact only for a clean clock at a multiple of 100 Hz.
- Repair the clock first. Unwrap a counter that wraps, spread samples that share one
  timestamp evenly up to the next timestamp, and treat a backward step or a jump of more than
  1 s as one sample period.
- A grid point is pen-down only when its nearest real sample is pen-down and lies within two
  sample periods. Otherwise it is pen-up and dropped.
- Scale pressure by the maximum the driver reports for the device.
- Enrol and verify on the same device model. References from one device scored probes from
  another at 13 to 42% EER even after every repair (24 to 69% before).
- On the first XPPen recordings, log the raw timestamp intervals and the pressure maximum, and
  check them against these rules before any threshold is set from them.

### High-Level Service Contract: `SignatureService` (for GUI & Backend Teams)

To isolate frontend (PyQt5) and backend (SQLite/AES) developers from PyTorch internals,
model weights management, and byte deserialization, `src/verification.py` exposes
`SignatureService`. Developers should instantiate this class once at app startup:

```python
from src.verification import SignatureService

service = SignatureService(device="cpu")
```

#### 1. Verification Modes (`service.verify(mode=...)`)

The verification engine provides three explicit operational modes:

| Mode | Input Arguments Required | Processing & Routing |
| --- | --- | --- |
| **`"unimodal"`** | `probe_trajectory` + `reference_temporal`<br>*OR*<br>`probe_image` + `reference_spatial` | Runs single-branch verification (`spatial_only` or `temporal_only`). Flagged as `degraded: True` because it relies on one modality. |
| **`"temporal-multi"`** | `probe_trajectory` + `reference_temporal` + `reference_spatial` | **Single-action digitizer verification.** The user signs once on the tablet. The engine extracts the temporal embedding, auto-renders the spatial stroke image ($1.8\text{ ms}$), extracts the spatial embedding, and evaluates late fusion. Routing: `multimodal`. |

**A spatial probe is scored only against references of the same origin.** Measured
on 20 paired SigComp2011 writers: rendered probe against rendered references 22.9%
EER, scanned against scanned 24.5%, rendered probe against scanned references
**57.4%**, scanned probe against rendered references **54.1%**. The last two are
worse than chance. A render sits about 3.6 times further from its own scan than
two scans of one writer sit from each other, so across that gap the spatial score
says nothing about identity. `SignatureService.verify()` enforces this. Every
`enroll_*` helper returns `"spatial_origin"` (`"render"` from `enroll_from_tablet`,
`"scan"` from `enroll_multimodal` and spatial `enroll_unimodal`). `verify()`
requires `reference_spatial_origin` whenever spatial references are passed, and
raises instead of guessing. On a mismatch it drops the spatial branch (routing
`temporal_only`, `degraded: True`, and a reason naming both origins). A unimodal
spatial call with mismatched references raises. **The database therefore stores
`spatial_origin` next to each spatial template, with `model_version`** (store
`config.settings.MODEL_VERSION`, the release tag of the shipped weights; a template whose
tag differs from the running one must be re-enrolled). A user who
must be verified both by tablet and by scan enrolls through both paths and holds
two spatial reference sets.
| **`"multimodal"`** | `probe_image` + `probe_trajectory` + `reference_spatial` + `reference_temporal` | **Dual physical verification.** Paper scan/photo + tablet trajectory evaluated together under late fusion. Routing: `multimodal`. |

**Several signings at verification (a toggle for the GUI).** `config.settings.VERIFICATION_SIGNINGS`
(default 1) is how many signings of each kind the app asks for at verification; the GUI may
offer 1 or 2 as an option. Pass them as lists: `probe_trajectory=[t1, t2]`,
`probe_image=[img1, img2]`, in every mode. Each branch scores the **mean** of its signings
against the references, and the thresholds are unchanged. Every signing must pass its input
gate; if one fails, that branch is dropped exactly as a failed single probe is (`degraded`,
with `"<branch>, signing <n>: ..."` in `reasons`), so ask for that signing again. Never
accept on the better of two signings: a forger would get two tries. Measured 2026-10-21, pen,
K = 5, 140 BiosecureDS2 writers enrolled at one sitting: two signings took EER from 16.06%
to 13.70% a session later and from 8.42% to 6.04% at the same sitting.

**Step-up verification (a contract for the GUI team; off until the GUI handles it).** With
`config.settings.VERIFICATION_STEP_UP = True`, a verification that fuses paper and tablet
(routing `multimodal`) can return a third decision, `SIGN_AGAIN` (`"Sign again"`, import it
from `src.verification`), besides `"Genuine"` and `"Forged"`. The flow:

1. Ask for one signing of each kind and call `service.verify()` as today.
2. On `"Genuine"` or `"Forged"`, stop: the verdict is final.
3. On `"Sign again"`, keep the first signings, ask for one more signing of each kind, and call
   `service.verify()` again with both as lists, first signing first:
   `probe_image=[img1, img2]`, `probe_trajectory=[t1, t2]`. That call returns only `"Genuine"`
   or `"Forged"`, and its verdict is final.

Rules:

- Never ask a third time. Under step-up, `verify()` raises on more than two signings of a
  kind, and on different paper and tablet counts.
- On `"Sign again"`, show a neutral request to sign once more. Never show the score or say how
  close the signing was: a forger would learn how near they came.
- Log one verification event per visit, with its final verdict; `"Sign again"` is not a verdict.
  Today `src/gui/verification.py` (around line 150) styles every decision other than `"Genuine"`
  as a rejection and logs it, and `src/gui/dashboard.py` counts `"Genuine"` events: change both
  before turning the setting on.
- A signing that fails its input gate is a capture retry, as in the rule above, not a step. If a
  kind is dropped, the other kind decides alone at its own threshold, with no step-up.
- Keep `VERIFICATION_SIGNINGS = 1` while step-up is on. It applies to fused decisions only,
  `temporal-multi` included (its scan-calibrated thresholds stand in there as
  `DECISION_THRESHOLD` already does); pen-only and scan-only verification keep their single
  thresholds.

Measured on 132 BiosecurID-SONOF writers three sessions after enrolment, with the three
thresholds in `config/settings.py` calibrated for forgers who repeat their first attempt (the
worst case for false accepts): forgeries passed stay at 5% over both steps. Of the signers asked
again, 36.5% of genuine signers pass and under 1% of forgers. Overall 11.4% of genuine signers are
refused against 15.7% for one signing at the same 5%, and 18% of them sign twice. Not measured: a
forger who tries harder after being asked again.

**Adding confirmed signings to the references (for the database and GUI teams).** A person's
signature drifts over months, and references from one sitting fall behind it: on 132
BiosecurID-SONOF writers enrolled at one sitting, the product refused 19.5% of genuine signers
three sessions later at the shipped threshold, against 4.4% at the same sitting. References
from several occasions follow the drift. After a verification that was accepted **and** in
which the employee confirmed the person's identity, append that visit's embeddings (from the
same `verify()` inputs, re-embedded with `enroll_*`) to the person's stored references. The
model side needs no change: `verify()` takes any number of references. Measured on SONOF:
references from three occasions lowered the fused EER from 8.38% to 6.68%, and adding accepted
signatures lowered FRR from 13.90% to 8.07% while FAR at the same threshold rose from 4.57% to
6.31%.

- Add only results with `decision == "Genuine"`, `degraded == False`, and the employee's
  confirmation. Never add a rejected, degraded or unconfirmed signing: an added forgery would
  stay in the references.
- Store embeddings only, as at enrolment, with the date and `spatial_origin` of each one.
- Cap the set per modality (the measured arms held 6 to 8) and drop the oldest added signing
  first. The cap is not measured beyond that.
- This does not help the first visit after enrolment, which still faces the whole gap; it helps
  every visit after a confirmed one.

**Returns:**
```python
{
    "decision": "Genuine" | "Forged",
    "confidence": float,       # 0.0-100.0. The fused similarity scaled by 100,
                               # NOT a calibrated probability. S = 1/(1+d) against
                               # the nearest reference; do not present it as a
                               # percentage likelihood of being genuine.
    "routing": str,            # "multimodal", "spatial_only", or "temporal_only"
    "degraded": bool,          # True if a modality was dropped or pressure was imputed
    "reasons": list[str],      # Explanations if degraded or gate failure occurred
    "threshold": float         # The acceptance bar applied, for the audit log:
                               # DECISION_THRESHOLD fused, SPATIAL_/TEMPORAL_DECISION_
                               # THRESHOLD alone, STEP_UP_ACCEPT on a step-up first
                               # signing (also when it returns "Sign again" or a reject
                               # below STEP_UP_REJECT), STEP_UP_TWO on the second.
}
```

#### 2. Enrollment Helpers (Returns DB-Ready Blobs)

Frontend and database developers should use these helper methods during enrollment:

- **`service.enroll_from_tablet(trajectories: list[np.ndarray]) -> dict`**
  - Input: List of $K=5$ normalized $[L, 4]$ trajectories from the tablet.
  - Output: `{"spatial_embeddings": [bytes...], "temporal_embeddings": [bytes...], "spatial_origin": "render"}`
  - Automatically renders each trajectory to spatial images and computes both 512-d and 128-d embeddings. The spatial templates are renders and can only be verified against rendered probes (`temporal-multi`).
- **`service.enroll_multimodal(images: list, trajectories: list) -> dict`**
  - Input: $K=5$ image paths/arrays and $K=5$ trajectories.
  - Output: `{"spatial_embeddings": [bytes...], "temporal_embeddings": [bytes...], "spatial_origin": "scan"}`.
- **`service.enroll_unimodal(samples: list, modality: "spatial"|"temporal") -> dict`**
  - Output: `{"embeddings": [bytes...], "modality": str}`, plus `"spatial_origin": "scan"` when the modality is spatial.

Store `spatial_origin` with the spatial templates and pass it back as `verify(..., reference_spatial_origin=...)`; `verify()` raises without it whenever `reference_spatial` is given.

All reference templates accept either decrypted raw Python `bytes` (contiguous
IEEE 754 float32 bytes directly from the database layer) or `np.ndarray` float32 embeddings.

#### 3. Architectural Boundary: Model Branch vs. Database/Security Layer

To preserve strict separation of concerns and engineering modularity:
- **Model Branch (`src/`):** Performs biometric feature extraction and mathematical verification only.
  Outputs plain, unencrypted embeddings (512-d and 128-d float32) and expects plain, decrypted
  reference templates for comparison. The model branch never holds encryption keys and never performs
  cryptographic operations.
- **Backend / Database Layer:** Manages application-layer AES-256-GCM encryption and decryption (ISO/IEC 18033-1),
  key management (OS keyring / environment), and SQLite persistence (ISO/IEC 9075-1). It receives plain
  embeddings from `SignatureService.enroll_*()`, encrypts them for storage, and decrypts them before passing
  to `SignatureService.verify()`.

**Image arrays must be raw scans** — 2-D, paper bright, ink dark, any size, exactly what
`cv2.imread(..., IMREAD_GRAYSCALE)` returns. They are resized and inverted on the way in,
identically to a path. An already-canonical `[1, 155, 220]` float array in `[0, 1]` (what
the renderer emits) is passed through untouched. Anything else raises rather than being
guessed at: `SpatialSiamese` pools adaptively, so a wrong-sized or un-inverted array would
otherwise yield a plausible 512-d embedding with no error and be stored as a template.

---

## Standards

Cite **exactly** these five standards and no others:

| Standard                 | Application                                                 |
| ------------------------ | ----------------------------------------------------------- |
| **IEEE 730**             | Configuration control, deviation log, audit log             |
| **ISO/IEC 23053:2022**   | Writer-independent data splits, documented training config  |
| **ISO/IEC 24745:2022**   | Only derived embeddings persisted, never raw biometric data |
| **ISO/IEC 9075-1:2023**  | Structured relational SQLite schema                         |
| **ISO/IEC 18033-1:2021** | AES-256-GCM encryption of stored embedding payloads         |

Do **not** cite ISO/IEC 18033-3 or a generic "ISO/IEC 9075".

---

## Security Rules

- Encrypt embedding payloads with **AES-256-GCM** at the application layer
  via the `cryptography` package
- Encryption key held **outside** the template store (key file / OS keyring /
  operator-credential derivation) — never in the database
- **Never persist raw images or raw coordinate streams** — only derived
  embeddings (ISO/IEC 24745)
- Audit log: record every verification event — `User ID, Timestamp,
Modality State, Similarity Score, Decision` — plaintext and queryable

---

## Dependencies

**Allowed:**

```
torch, torchvision, opencv-python, numpy, PyQt5,
cryptography, matplotlib, scikit-learn, Pillow, scipy, sqlite3 (stdlib)
```

`scipy` is used by `src/data/features.py` (signal filtering) and `src/data/parsers.py`
(`.mat` corpus loading).

**Allowed for training and tests only** (never imported by `src/`):

```
tqdm, pytest
```

**Not allowed:**

- No vendor signature-pad SDKs
- No SQLCipher
- No packages requiring native build steps on either platform
- No cloud/API client libraries

---

## Module Boundaries

| Directory       | Responsibility                                          | Does NOT handle                         |
| --------------- | ------------------------------------------------------- | --------------------------------------- |
| `src/models/`   | Network architectures, loss, fusion                     | Data loading, evaluation metrics, GUI   |
| `src/data/`     | Dataset loading, preprocessing, augmentation, splitting | Model training, database operations     |
| `src/database/` | SQLite schema, embedding CRUD, encryption, audit log    | Model inference, GUI rendering          |
| `src/capture/`  | Tablet hardware abstraction, platform drivers           | Image preprocessing, model inference    |
| `src/gui/`      | PyQt5 application UI, user interaction                  | Direct model training, database schema  |
| `scripts/`      | Training and evaluation entry points                    | Core library logic (import from `src/`) |
| `config/`       | Centralized hyperparameters and constants               | Runtime logic                           |
| `assets/screenshots/` | Hand-made screenshots shown in the README, for every team | Generated figures. Never put screenshots in `docs/` (gitignored) |
| `results/`      | Figures written by code through `RESULTS_DIR`           | Hand-made images                        |

---

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore, perf
Scopes: models, data, database, capture, gui, config, scripts
```

---

## Run-to-run variance and significance testing

**Each branch has its own noise floor, and they are far apart. Never use one branch's number for the other.**

### Spatial branch (~0.7 EER points)

The same spatial configuration on the same writer split, with seeds 42 and 43,
scored 8.59% and 9.27% pooled. Bengali moved 1.7 points between the two.
**Before claiming a spatial arm beat another**, the gap must be more than one
point, or the arm must run with two seeds and the same `--split-seed`, so both are
scored on the same held-out writers.

### Temporal branch (~2.6 EER points pooled)

The temporal branch trains on 62 writers against about 300 for spatial, so its
spread between seeds is much wider. Same configuration, `--split-seed 42`, training
seeds 42, 43 and 44:

| seed | pooled | svc | Dutch | Chinese |
| --- | --- | --- | --- | --- |
| 42 | 9.61% | 16.32% | 4.24% | 7.94% |
| 43 | 10.86% | 15.85% | 6.69% | 8.38% |
| 44 | 12.18% | 16.95% | 7.72% | 7.14% |
| **range** | **2.57** | 1.10 | **3.48** | 1.24 |

sigcomp_Dutch is tested on only 15 writers and is much noisier than the table
shows: a six-seed control spans **6.90 points** there (3.42% to 10.32%). Treat any
Dutch-only claim under about 5 points as unmeasurable.

**Before claiming a temporal arm beat another**, run **six seeds, paired by seed,
against a control re-run under current code.**

- Three seeds is not enough. At three seeds one arm gave three wrong readings at
  once (a variance drop of F = 42.6 that was F = 1.20 at six seeds, and a Chinese
  regression that vanished). The same rule had earlier rejected two real effects
  of 1.2 to 1.5 points: `dt`, and attention pooling, which was deleted from the
  code on a 1.12-point gap.
- Never compare a new arm against numbers from an earlier session. Re-run the
  control under the current code.
- `scripts/train.py` sets `torch.use_deterministic_algorithms(True)`, so a
  non-deterministic kernel raises an error instead of silently changing results. A
  seed is reproducible under the same code and data, but the spread between seeds
  (about 0.7 points) is still real.
- A run takes about 12 minutes with early stopping, so six seeds is about 75
  minutes. There is no cost argument for three.

## Evaluation Protocol

- **Writer-independent split** across all datasets — no signer in multiple partitions
- All metrics reported under the **skilled-forgery** scenario
- Tune `α` and `τ` on **validation only**, report on **held-out test**
- If FAR target is missed, **prioritize keeping FAR low** over FRR
- Label fused EER as **provisional** until in-house multimodal data exists
- **Never report a pooled temporal EER on its own.** SVC2004 and SigComp2011 are
  both pen-tablet datasets, and one training run scores 29.62% on the first and
  10.92% on the second. A pooled figure averages those into a number that
  describes neither. Same rule the spatial branch already follows for UTSig.

### Two thresholding protocols — never mix them

Equal Error Rate (EER) can be computed under two standard protocols, which answer
different questions and yield substantially different numbers on the same model:

| protocol | how | what it answers |
| --- | --- | --- |
| **pooled** (global threshold) | every writer's scores concatenated, one EER | what a system shipping **one** threshold achieves |
| **per-writer** (user threshold) | EER computed inside each writer, then averaged | what the literature quotes, and what per-user enrolment can reach |

Measured on one model, same writers, same draws:

| | pooled | per-writer |
| --- | --- | --- |
| SynSig2Vec, 31 test writers | 8.92% | **4.41%** (median 0.79%) |
| the same model, SVC2004 only | 18.58% | **9.93%** |

**Rules:**

- **Report both.** `tools/train_synsig2vec.py` prints them side by side per
  corpus; `per_writer_eer()` is the implementation.
- **Pooled is the honest headline** for this project, because the deployed system
  ships a single `τ`. Per-writer exists for comparison against published work.
- **Never compare across protocols.** A per-writer number placed beside a pooled
  one makes the model look 2-3x better than it is. Published SVC2004 figures near
  2-3% — including the competition's own winning 2.89% — are user-threshold
  numbers. The two MOBISIG rows in `README.md` show the same effect within one
  paper: 5.81% user-specific against 14.31% global.
- **Per-writer EER is optimistic.** It is computed post-hoc, choosing each
  writer's best threshold with the labels known, so it is a best case rather than
  a fieldable operating point.

### Merging a New Online Dataset

Before a new online dataset is merged with SVC2004, **measure** these against it
and align inside that dataset's loader, not in the model:

| property       | SVC2004 Task2                           | why it matters                                                                                                              |
| -------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| sample rate    | 100 Hz (timestamp steps 10 ms)          | The model derives `(dX, dY, dP)` from consecutive frames, so a 200 Hz dataset reports half the delta for the same pen speed |
| pen-up samples | none — 0 of 100,494 points have `P = 0` | A dataset that records in-air travel gives the pressure channel a second meaning                                            |
| pressure scale | 0–1023 device units                     | A per-file max would make a light signature look heavy                                                                      |

This is not hypothetical bookkeeping. Merging SVC2004 Task1 into Task2 without
reconciling the pressure channel collapsed the model to Cohen's d 0.279, and
SigComp2011 differed on the first two rows above. Alignment belongs in the
parser so the capture contract stays untouched.

A fourth property, added after DeepSignDB: **the sample rate can vary per file
inside one dataset.** BiosecurID's Development half is a near-even mix of 100 Hz
and 200 Hz captures across four recording sessions. Measure the rate distribution
before choosing between a per-dataset constant and a per-file factor; assuming
uniformity would have halved the apparent pen speed for 180 of 376 sampled files.

### Licensed Datasets

Some corpora are held under a signed agreement rather than published: DeepSignDB
and BiosecurID-SONOF. For any such dataset:

- **Never commit it, nor anything derived from it** — no signature files, no
  cached features, no embedding dumps. `data/` being gitignored is the safety net,
  not the policy. Check `git status` before committing while one is on disk.
- **Never render its signatures into a plot or figure** unless the agreement's own
  whitelist allows that specific user. DeepSignDB permits only `u1010`, `u1013`,
  `u1028` (Development) and `u0103`, `u0145` (Evaluation).
- **Carry its required citations** into anything public. DeepSignDB clause 4.5
  names four, listed under ⚖ in the README bibliography. BiosecurID-SONOF clause
  4.4 names one (Galbally et al., Pattern Recognition 48, 2015).
- **Treat derived weights as research-only.** The published release weights are
  trained on open corpora; licensed-data weights stay local. See the README's
  Model Weights section.
- **Record the expiry.** DeepSignDB's agreement runs to 25 July 2028 and
  BiosecurID-SONOF's to 14 October 2028, after which all copies are destroyed.

The clause-by-clause reading lives in DATASET_LICENSES.md. This is the
operational summary; that file is the authority if the two ever disagree.

### Pressure Ablation (temporal branch — two required runs)

The temporal branch is trained **twice**, on the two SVC2004 tasks, and both
results are reported. The tasks share filenames, writer count, and split seed,
so the only variable between the runs is the pressure channel — which makes the
difference in EER a direct measurement of what pressure contributes.

| Run              | Dataset              | Channel 3                                        | Pressure dropout                                     | Role                                                                                        |
| ---------------- | -------------------- | ------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **A — baseline** | `data/svc2004_task1` | `P = 1.0` imputed (4-col source)                 | Off (it does nothing when every P is already 1.0)    | Pressure-blind control. Also the honest lower bound for any device that reports no pressure |
| **B — headline** | `data/svc2004_task2` | Real pressure, scaled to `[0, 1]` (7-col source) | **Off** (`PRESSURE_DROPOUT_RATE = 0`), measured harmful | The shipped temporal model — matches the capture device's channel set                       |

Rules:

- **Run B is the deployed model.** Run A exists to quantify the delta and to
  bound degraded-mode performance; it is never the shipped weights.
- Both runs were measured on the same 20/10/10 writer split (seed 42, K=5): Run A
  (pressure-blind) **23.62% ± 2.18**, Run B (real pressure) **21.32% ± 1.50**.
  Pressure is worth about 2.3 EER points, but with only 10 test writers that is
  close to one standard deviation. The direction is the same at K=10 (22.02% vs
  19.55%), but it is not a precise measurement.
- Both runs use the **same `--seed`**, so the writer split is identical and the
  two EERs are directly comparable. A different seed invalidates the comparison.
- Report both EERs side by side. Do **not** report only the better number.
- Pressure dropout is off for Run B too. It was meant to keep Run B usable on a
  device with no pressure, but it lost even on that input (pressure-blind EER
  went from 23.32% to 20.32% when it was removed; see Temporal Branch above). A
  device without pressure gets a flat `P = 1.0` and the `degraded` flag instead.
- Weights are written to distinct filenames so Run A cannot overwrite Run B.
