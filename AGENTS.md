# AGENTS.md — Contributor & Agent Directives

## Signature Verification System (EE 499, Team 06)

This file defines the hard architectural constraints for anyone working on
this codebase — human contributors and AI coding agents alike. These rules
are distilled from the approved Term-1 design baseline and are **non-negotiable**
unless the team explicitly authorizes a deviation.

Read this file before writing or reviewing any code in this repository.

---

## Drift Prevention

The following were **evaluated and deliberately excluded**. Do **not** add
them, even if they appear in the literature review or older drafts:

- **No Spatial Transformer Network (STN)** in the pipeline
- **No BioHashing / quantization / Hamming distance** — matching uses
  continuous embeddings with L2 or cosine distance
- **No SQLCipher / whole-file DB encryption** — encryption is
  application-layer AES-256-GCM only
- **No extra *captured* channels** — the capture contract is `[1, L, 3]`,
  `(X, Y, Pressure)`. The device must not be asked for azimuth, altitude or
  tilt. Features **derived** from those three channels inside the model are
  allowed and expected (see Temporal Branch below)
- **No cloud APIs, no remote inference, no discrete GPU at inference**

---

## Hard Constraints

| Constraint | Value |
|---|---|
| Inference RAM ceiling | ≤ 4 GB |
| End-to-end verification latency | < 3.0 seconds |
| Execution | Local only — no cloud, no dedicated GPU |
| Platform support | Windows + Arch Linux (x86_64 and ARM64) |
| Native build steps | None — all dependencies must have prebuilt wheels |

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
- Backbone: **Siamese CNN** → **512-d** embedding. The *model type* is fixed;
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
  removes capacity the model needs. `compact` remains shipped: 0.53 points does
  not pay for ~11× the training compute plus mandatory gradient checkpointing.
  Full table in `docs/SOLUTIONS.md`.
- **A checkpoint carries its own architecture.** `SpatialSiamese.from_state_dict`
  recovers trunk, embedding width and FC head from the saved parameter shapes,
  and every load path (`benchmark.py`, `evaluate.py`, `src/verification.py`) goes
  through it. Never add a load path that builds the model from `config/` instead:
  config points at whatever is being trained *now*, and scoring an archived
  checkpoint under the wrong architecture yields a plausible number that means
  nothing.
- Normalization: **GroupNorm or LayerNorm** — never BatchNorm (pair-sampled batches)
- Quality gate: effective-resolution proxies (bounding-box pixel size +
  Laplacian-variance sharpness). Do **not** gate on the file's DPI tag

### Temporal Branch (Online)

- Input tensor: **`[1, L, 3]`** — `(X, Y, Pressure)`, variable length `L`
- **Derived dynamics:** the model appends per-frame deltas `(dX, dY, dP)` to
  form a `[1, L, 6]` tensor *internally*, before the LSTM. Velocity is the most
  discriminative feature family in the online-verification literature: a forger
  can copy the shape of a signature while tracing it far more slowly and evenly.
  This costs the capture device nothing, since deltas come from consecutive
  frames of the three channels it already reports. Feeding raw coordinates alone
  produced a collapsed model with train loss pinned at the theoretical
  no-discrimination value (see `docs/PROBLEMS_FACED.md` P-002)
- Backbone: **Siamese LSTM** → **128-d** embedding
- **Training:** pressure-channel dropout (~30–50% of samples replace real
  pressure with `P = 1.0`)
- **Inference:** if no pressure available, impute `P = 1.0` and set the
  degraded-mode flag in the GUI

#### Capture Output Contract (device → model)

A signature pad emits more channels than the model accepts. Tablets typically
report **7**: `X, Y, timestamp, button/contact, azimuth, altitude, pressure` —
the same shape as SVC2004 Task2, which is why Task2 is the dataset that matches
the deployment device.

The target device is an **XPPen Deco 640**, which reports position and pressure
but no tilt or azimuth. The 3-channel contract is therefore what the hardware
can actually deliver, not merely a preference.

`src/capture/` is responsible for narrowing whatever the device produces down
to the 3 channels the model takes. Widening the *captured* contract is banned
under Drift Prevention. Deriving further features from those 3 channels inside
the model is not.

| Device channel | Fate |
|---|---|
| X, Y | **Kept** — z-scored per signature (zero mean, unit variance per axis) |
| Pressure | **Kept** — scaled to `[0.0, 1.0]`; absent ⇒ impute `1.0` + degraded flag |
| Timestamp | Dropped |
| Button / contact state | Dropped — it is pen-down/up, **not** a pressure substitute |
| Azimuth, altitude, tilt | Dropped |

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

## Standards

Cite **exactly** these five standards and no others:

| Standard | Application |
|---|---|
| **IEEE 730** | Configuration control, deviation log, audit log |
| **ISO/IEC 23053:2022** | Writer-independent data splits, documented training config |
| **ISO/IEC 24745:2022** | Only derived embeddings persisted, never raw biometric data |
| **ISO/IEC 9075-1:2023** | Structured relational SQLite schema |
| **ISO/IEC 18033-1:2021** | AES-256-GCM encryption of stored embedding payloads |

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

| Directory | Responsibility | Does NOT handle |
|---|---|---|
| `src/models/` | Network architectures, loss, fusion | Data loading, evaluation metrics, GUI |
| `src/data/` | Dataset loading, preprocessing, augmentation, splitting | Model training, database operations |
| `src/database/` | SQLite schema, embedding CRUD, encryption, audit log | Model inference, GUI rendering |
| `src/capture/` | Tablet hardware abstraction, platform drivers | Image preprocessing, model inference |
| `src/gui/` | PyQt5 application UI, user interaction | Direct model training, database schema |
| `scripts/` | Training and evaluation entry points | Core library logic (import from `src/`) |
| `config/` | Centralized hyperparameters and constants | Runtime logic |

---

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore, perf
Scopes: models, data, database, capture, gui, config, scripts
```

---

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

| property | SVC2004 Task2 | why it matters |
|---|---|---|
| sample rate | 100 Hz (timestamp steps 10 ms) | The model derives `(dX, dY, dP)` from consecutive frames, so a 200 Hz dataset reports half the delta for the same pen speed |
| pen-up samples | none — 0 of 100,494 points have `P = 0` | A dataset that records in-air travel gives the pressure channel a second meaning |
| pressure scale | 0–1023 device units | A per-file max would make a light signature look heavy |

This is not hypothetical bookkeeping. Merging SVC2004 Task1 into Task2 without
reconciling the pressure channel collapsed the model to Cohen's d 0.279, and
SigComp2011 differed on the first two rows above. Alignment belongs in the
parser so the `[L, 3]` capture contract stays untouched.

### Pressure Ablation (temporal branch — two required runs)

The temporal branch is trained **twice**, on the two SVC2004 tasks, and both
results are reported. The tasks share filenames, writer count, and split seed,
so the only variable between the runs is the pressure channel — which makes the
difference in EER a direct measurement of what pressure contributes.

| Run | Dataset | Channel 3 | Pressure dropout | Role |
|---|---|---|---|---|
| **A — baseline** | `data/svc2004_task1` | `P = 1.0` imputed (4-col source) | Disabled — it is a no-op when every P is already 1.0 | Pressure-blind control. Also the honest lower bound for any device that reports no pressure |
| **B — headline** | `data/svc2004_task2` | Real pressure, scaled to `[0, 1]` (7-col source) | **Active at `PRESSURE_DROPOUT_RATE`** | The shipped temporal model — matches the capture device's channel set |

Rules:

- **Run B is the deployed model.** Run A exists to quantify the delta and to
  bound degraded-mode performance; it is never the shipped weights.
- Both runs use the **same `--seed`**, so the writer split is identical and the
  two EERs are directly comparable. A different seed invalidates the comparison.
- Report both EERs side by side. Do **not** report only the better number.
- Pressure dropout stays on for Run B. It is what keeps Run B usable when a
  device reports no pressure at inference — without it, the degraded path falls
  outside the model's learned distribution.
- Weights are written to distinct filenames so Run A cannot overwrite Run B.
