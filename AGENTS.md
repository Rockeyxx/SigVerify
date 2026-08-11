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
measured, why a design choice was made, or which experiments failed goes in
**`SUMMARY.md`**. Rules and contracts go here in `AGENTS.md`; dataset terms go in
`DATASET_LICENSES.md`.

**Do not move report content into the README.** Protocol derivations, experiment
history and rejected arms belong in `SUMMARY.md` even when they explain a number
the README quotes.

**`docs/` is gitignored, so never link to it from a tracked file.** The link is
dead for anyone who clones. Inline the fact instead. Full conventions are in the
`docs-style` skill.

---

## Drift Prevention

The following were **evaluated and deliberately excluded**. Do **not** add
them, even if they appear in the literature review or older drafts:

- **No Spatial Transformer Network (STN)** in the pipeline
- **No BioHashing / quantization / Hamming distance** — matching uses
  continuous embeddings with L2 or cosine distance
- **No SQLCipher / whole-file DB encryption** — encryption is
  application-layer AES-256-GCM only
- **No extra _captured_ channels** — the capture contract is `[1, L, 3]`,
  `(X, Y, Pressure)`. The device must not be asked for azimuth, altitude or
  tilt. Features **derived** from those three channels inside the model are
  allowed and expected (see Temporal Branch below)
- **No cloud APIs, no remote inference, no discrete GPU at inference**

---

## Hard Constraints

| Constraint                      | Value                                             |
| ------------------------------- | ------------------------------------------------- |
| Inference RAM ceiling           | ≤ 4 GB                                            |
| End-to-end verification latency | < 3.0 seconds                                     |
| Execution                       | Local only — no cloud, no dedicated GPU           |
| Platform support                | Windows + Arch Linux (x86_64 and ARM64)           |
| Native build steps              | None — all dependencies must have prebuilt wheels |

---

## Architecture Rules

### Spatial Branch (Offline)

- Input tensor: **`[1, 1, 155, 220]`** (8-bit grayscale), resized **directly** onto
  the target — no square pad. Measured 2026-07-26: this beat the previous
  `[1, 1, 256, 256]` square-padded geometry on **every** dataset (utsig −3.38,
  pooled −1.41 EER points) and trains about twice as fast. BHSig260's ribbons are
  roughly 1:3.5, so square padding left ~70% of the canvas blank and cost more
  stroke resolution than the aspect distortion costs. `SPATIAL_SQUARE_PAD` in
  `config/settings.py` keeps both paths available for re-measurement.
- Pixel polarity: **inverted — background 0, ink high.** Measured: CEDAR
  4.86% → 0.00% EER. Ink covers ~1% of the canvas, so an un-inverted image feeds
  every convolution a large constant offset carrying no identity signal.
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
- **Measured 2026-07-28, all three arms.** `wide` alone reached 11.78% pooled
  against `compact`'s 12.31% (utsig −3.35, its first movement in five changes).
  Adding Dey's dropout cost 6.78 points, and adding their FC-1024 head on top of
  that collapsed the model to one embedding before it partly recovered. **Do not
  set `SPATIAL_CONV_DROPOUT` or `SPATIAL_FC_HIDDEN` without a new reason** — at
  301 writers the failure is data scarcity, not excess capacity, so regularizing
  removes capacity the model needs. **`wide` is what ships** (`SPATIAL_TRUNK =
  "wide"`); its dropout schedule and FC-1024 head are the parts that do not.
- **A checkpoint carries its own architecture.** `SpatialSiamese.from_state_dict`
  recovers trunk, embedding width and FC head from the saved parameter shapes,
  and every load path (`benchmark.py`, `evaluate.py`, `src/verification.py`) goes
  through it. Never add a load path that builds the model from `config/` instead:
  config points at whatever is being trained _now_, and scoring an archived
  checkpoint under the wrong architecture yields a plausible number that means
  nothing.
- **Training-time augmentation is on** (`SPATIAL_AUGMENT`). Measured 2026-07-30:
  every prior spatial run best-validated at **epoch 1**, which was image variety,
  not pair supply — 301 writers seen pixel-identical every epoch. Small affine
  warps plus stroke-width jitter moved the best epoch to 6 and pooled EER 11.78%
  → 11.19%. Keep the ranges small; forgery detection keys on fine stroke geometry.
  Warp each side of a pair **independently**, or the distance between them is
  unchanged and the metric learns nothing.
- **LR schedule: plateau** (`--lr-schedule plateau`, halve after 2 stalled
  epochs). Fired twice to 2.5e-4 and reached the best validation loss of any
  spatial run (0.1355). Pooled EER ties augmentation alone, but the draw spread
  tightened 40% (±0.83 → ±0.47). It ships for the spread, not the mean.
- Normalization: **GroupNorm or LayerNorm** — never BatchNorm (pair-sampled batches)
- Quality gate: effective-resolution proxies (bounding-box pixel size +
  Laplacian-variance sharpness). Do **not** gate on the file's DPI tag

### Temporal Branch (Online)

- Input tensor: **`[1, L, 3]`** — `(X, Y, Pressure)`, variable length `L`
- **Derived dynamics:** the model appends per-frame deltas `(dX, dY, dP)` to
  form a `[1, L, 6]` tensor _internally_, before the LSTM. Velocity is the most
  discriminative feature family in the online-verification literature: a forger
  can copy the shape of a signature while tracing it far more slowly and evenly.
  This costs the capture device nothing, since deltas come from consecutive
  frames of the three channels it already reports. Feeding raw coordinates alone
  produced a collapsed model with train loss pinned at the theoretical
  no-discrimination value (recorded as P-002)
- Backbone: **Siamese LSTM** → **128-d** embedding
- **Training:** pressure-channel dropout is **off** (`PRESSURE_DROPOUT_RATE = 0`).
  This rule previously mandated 30–50% and was **measured wrong on 2026-07-29**.
  Dropout at 0.4 cost 2.47 pooled EER points on normal input (12.73% → 10.26%)
  and, against its own justification, 3.00 points on pressure-blind input
  (23.32% → 20.32%). It lost on both paths at once, so no robustness argument
  survives. Re-enabling it requires a measurement that beats these numbers, not
  an appeal to robustness.
- **Inference:** if no pressure available, impute `P = 1.0` and set the
  degraded-mode flag in the GUI
- **The dropped channels were measured 2026-07-30, and the ban now has a price.**
  On SVC2004 Task2 (40 writers, seed 42, 3-channel control 19.55% at K=10):
  adding `azimuth + altitude` reaches **11.17% (−8.38)**, adding real `dt`
  reaches **15.17% (−4.38)**. `button` is near-constant (mean 1.0) and carries
  nothing. Tilt is still banned under Drift Prevention **and** unusable on the
  XPPen Deco 640, but −8.38 points is a live argument for a different tablet, and
  the hardware claim is flagged unconfirmed below. `dt` is deployable today.
  Neither is shipped; `SIGCOMP2011` has neither channel, so both cost the branch
  84 of its 124 writers.
- **A temporal checkpoint carries its own channel count.**
  `TemporalSiamese.from_state_dict` reads it from the LSTM input width
  (`2 x input_features`, because dynamics are concatenated first). Every load
  path uses it, so a 4- or 5-channel experiment can never be silently scored as
  the 3-channel default.

#### Capture Output Contract (device → model)

A signature pad emits more channels than the model accepts. Tablets typically
report **7**: `X, Y, timestamp, button/contact, azimuth, altitude, pressure` —
the same shape as SVC2004 Task2, which is why Task2 is the dataset that matches
the deployment device.

The target device is an **XPPen Deco 640**, which reports position and pressure
but no tilt or azimuth. The 3-channel contract is therefore what the hardware
can actually deliver, not merely a preference. (not confirmed yet editable)

`src/capture/` is responsible for narrowing whatever the device produces down
to the 3 channels the model takes. Widening the _captured_ contract is banned
under Drift Prevention. Deriving further features from those 3 channels inside
the model is not.

| Device channel          | Fate                                                                     |
| ----------------------- | ------------------------------------------------------------------------ |
| X, Y                    | **Kept** — z-scored per signature (zero mean, unit variance per axis)    |
| Pressure                | **Kept** — scaled to `[0.0, 1.0]`; absent ⇒ impute `1.0` + degraded flag |
| Timestamp               | Dropped                                                                  |
| Button / contact state  | Dropped — it is pen-down/up, **not** a pressure substitute               |
| Azimuth, altitude, tilt | Dropped                                                                  |

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

Moved here from the README on 2026-08-08: it is a contract between teams, not
user documentation. Every rule below is a documented assumption the training data
satisfies, so a violation puts the probe in a distribution the model never saw.

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

The capture layer **must call `parsers.normalize_trajectory()`**. Until
2026-08-05 nothing checked, and the failure was silent *and* confident: on one
SigComp writer the same signature scored **0.9370 → Genuine** normalized and
**0.4888 → Forged** in raw tablet units — a genuine user rejected with no
`degraded` flag and no error.

| gate | rule | constant |
| --- | --- | --- |
| Length | `>= 10` points | `MIN_TRAJECTORY_POINTS = 10` |
| X / Y centred | `abs(mean) <= 0.5` per axis | `MAX_NORMALIZED_XY_MEAN = 0.5` |
| X / Y scaled | `std` within `[0.2, 5.0]` per axis | `NORMALIZED_XY_STD_RANGE` |
| Pressure range | within `[0, 1]` | — |
| Finite | no NaN or infinity | — |

The X/Y bounds detect the *class* of error rather than a device's particular
scale, so no per-device configuration is needed. A flat `P = 1.0` **passes**:
that is the legitimate pressure-blind device, reported as `degraded`, not
rejected.

### Trajectory format

| requirement | value |
| --- | --- |
| Shape | `[L, 3]` float32 — X, Y, Pressure |
| Sample rate | **100 Hz** (10 ms). Faster devices must be decimated |
| Pressure | scaled to [0, 1] by a **fixed device maximum**, never the file's own max |
| Pen-up samples | dropped (`P = 0`) |
| X / Y | raw device units in; `normalize_trajectory` z-scores them per signature |
| No pressure sensor | `P` imputes to 1.0 and `degraded` is set |

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
cryptography, matplotlib, scikit-learn, Pillow, sqlite3 (stdlib)
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

---

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore, perf
Scopes: models, data, database, capture, gui, config, scripts
```

---

## Run-to-run spread: do not call a sub-point difference an improvement

**The noise floor is per branch. They are not close. Never carry one to the other.**

### Spatial — roughly 0.7 EER points

**Measured 2026-08-09.** The same spatial configuration (margin 2.0), same writer
split, seeds 42 and 43, scored **8.59%** and **9.27%** pooled. Bengali swung 1.7
points between the two.

Every one of these earlier "improvements" is inside it and none is individually
proven: compact to wide (0.53), wide to augment+LR (0.55), augment+LR to pair
diversity (0.65), pair diversity to random impostors (0.48). The cumulative
12.31% to 8.59% is real; the individual steps are not.

### Temporal — roughly **2.6 EER points pooled**, and **3.5 on sigcomp_Dutch**

**Measured 2026-08-10.** The same temporal configuration, `--split-seed 42` so the
held-out writers are identical, training seeds 42/43/44:

| seed | pooled | svc | Dutch | Chinese |
| --- | --- | --- | --- | --- |
| 42 | 9.61% | 16.32% | 4.24% | 7.94% |
| 43 | 10.86% | 15.85% | 6.69% | 8.38% |
| 44 | 12.18% | 16.95% | 7.72% | 7.14% |
| **range** | **2.57** | 1.10 | **3.48** | 1.24 |

The temporal branch trains on **62 writers** against spatial's 301, and tests
sigcomp_Dutch on 15, so it is far noisier. The 0.7 figure was measured on spatial
and **was wrongly applied to twelve temporal arms on 2026-08-10** before this was
checked; most of those arms were ranked inside noise and their verdicts do not
stand. A single temporal run at 9.61% is a lucky draw from a distribution whose
mean is about **10.9%** — which also makes the shipped 10.26% an ordinary draw
rather than a special one.

**Before claiming any temporal arm beat another:** the gap must exceed **3 points**,
or the arm must be run at three seeds with `--split-seed` held fixed. Two seeds is
not enough at this spread.

**Before claiming any spatial arm beat another:** the gap must exceed one point, or
the arm must be run at two seeds with `--split-seed` held fixed so the comparison
is on identical held-out writers.

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
parser so the `[L, 3]` capture contract stays untouched.

A fourth property, added after DeepSignDB: **the sample rate can vary per file
inside one dataset.** BiosecurID's Development half is a near-even mix of 100 Hz
and 200 Hz captures across four recording sessions. Measure the rate distribution
before choosing between a per-dataset constant and a per-file factor; assuming
uniformity would have halved the apparent pen speed for 180 of 376 sampled files.

### Licensed Datasets

Some corpora are held under a signed agreement rather than published. DeepSignDB
is the first. For any such dataset:

- **Never commit it, nor anything derived from it** — no signature files, no
  cached features, no embedding dumps. `data/` being gitignored is the safety net,
  not the policy. Check `git status` before committing while one is on disk.
- **Never render its signatures into a plot or figure** unless the agreement's own
  whitelist allows that specific user. DeepSignDB permits only `u1010`, `u1013`,
  `u1028` (Development) and `u0103`, `u0145` (Evaluation).
- **Carry its required citations** into anything public. DeepSignDB clause 4.5
  names four, listed under ⚖ in the README bibliography.
- **Treat derived weights as research-only.** The published release weights are
  trained on open corpora; licensed-data weights stay local. See the README's
  Model Weights section.
- **Record the expiry.** DeepSignDB's agreement runs to 25 July 2028, after which
  all copies are destroyed.

The clause-by-clause reading lives in DATASET_LICENSES.md. This is the
operational summary; that file is the authority if the two ever disagree.

### Pressure Ablation (temporal branch — two required runs)

The temporal branch is trained **twice**, on the two SVC2004 tasks, and both
results are reported. The tasks share filenames, writer count, and split seed,
so the only variable between the runs is the pressure channel — which makes the
difference in EER a direct measurement of what pressure contributes.

| Run              | Dataset              | Channel 3                                        | Pressure dropout                                     | Role                                                                                        |
| ---------------- | -------------------- | ------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **A — baseline** | `data/svc2004_task1` | `P = 1.0` imputed (4-col source)                 | Disabled — it is a no-op when every P is already 1.0 | Pressure-blind control. Also the honest lower bound for any device that reports no pressure |
| **B — headline** | `data/svc2004_task2` | Real pressure, scaled to `[0, 1]` (7-col source) | **Active at `PRESSURE_DROPOUT_RATE`**                | The shipped temporal model — matches the capture device's channel set                       |

Rules:

- **Run B is the deployed model.** Run A exists to quantify the delta and to
  bound degraded-mode performance; it is never the shipped weights.
- **Both runs were executed 2026-07-29** on the same 20/10/10 writer split,
  seed 42, K=5: Run A (pressure-blind) **23.62% ± 2.18**, Run B (real pressure)
  **21.32% ± 1.50**. Pressure is worth roughly 2.3 EER points, though on 10 test
  writers that sits near one standard deviation — directionally consistent at
  K=10 (22.02% vs 19.55%) but not a tight measurement.
- Both runs use the **same `--seed`**, so the writer split is identical and the
  two EERs are directly comparable. A different seed invalidates the comparison.
- Report both EERs side by side. Do **not** report only the better number.
- Pressure dropout stays on for Run B. It is what keeps Run B usable when a
  device reports no pressure at inference — without it, the degraded path falls
  outside the model's learned distribution.
- Weights are written to distinct filenames so Run A cannot overwrite Run B.
