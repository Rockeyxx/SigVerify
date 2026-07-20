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
- **No `[1, L, 6]` temporal tensor** — the temporal input is `[1, L, 3]`
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

- Input tensor: **`[1, 1, 256, 256]`** (8-bit grayscale)
- Backbone: **Siamese CNN** → **512-d** embedding
- Normalization: **GroupNorm or LayerNorm** — never BatchNorm (pair-sampled batches)
- Quality gate: effective-resolution proxies (bounding-box pixel size +
  Laplacian-variance sharpness). Do **not** gate on the file's DPI tag

### Temporal Branch (Online)

- Input tensor: **`[1, L, 3]`** — `(X, Y, Pressure)`, variable length `L`
- Backbone: **Siamese LSTM** → **128-d** embedding
- **Training:** pressure-channel dropout (~30–50% of samples replace real
  pressure with `P = 1.0`)
- **Inference:** if no pressure available, impute `P = 1.0` and set the
  degraded-mode flag in the GUI

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
