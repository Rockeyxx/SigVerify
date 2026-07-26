# Signature Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)]()

> **Senior Design Project (EE 499)** — Department of Electrical and Computer Engineering, King Abdulaziz University (KAU), Jeddah, Saudi Arabia — Team 06

A multimodal biometric verification system that authenticates handwritten signatures using dual Siamese neural networks. Combines **offline** (image-based) and **online** (dynamic pen-trajectory) analysis through late fusion to deliver robust forgery detection.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Probe Signature Input                     │
│              (Image / Dynamic / Both)                        │
└──────────┬──────────────────────────────┬────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   Spatial Branch     │      │    Temporal Branch        │
│   Siamese CNN        │      │    Siamese LSTM           │
│                      │      │                           │
│  [1, 1, 256, 256]    │      │  [1, L, 3] (X, Y, P)     │
│       ↓              │      │       ↓                   │
│  512-d embedding     │      │  128-d embedding          │
│       ↓              │      │       ↓                   │
│  S_spatial           │      │  S_temporal               │
└──────────┬───────────┘      └──────────┬────────────────┘
           │                              │
           └──────────┬───────────────────┘
                      ▼
         ┌──────────────────────┐
         │   Late Fusion         │
         │                       │
         │  S = α·S_sp + (1-α)·S_t │
         │  S ≥ τ → Genuine      │
         │  S < τ → Forged       │
         └──────────┬────────────┘
                    ▼
         [Genuine/Forged, Confidence %]
```

**Automatic fallback:** if only one modality is available, the system routes to the corresponding unimodal branch (α = 1 for image-only, α = 0 for dynamic-only).

---

## Features

- **Dual-modality verification** — offline image analysis + online pen-trajectory analysis
- **Late-fusion scoring** — weighted combination with automatic unimodal fallback
- **Few-shot enrollment** — register users with one or more reference signatures, add more later
- **Encrypted storage** — AES-256-GCM encryption of embedding payloads at rest
- **Privacy by design** — only derived embeddings stored, never raw biometric data
- **Cross-platform** — runs on Windows and Linux with no native build steps
- **Desktop GUI** — PyQt5 application with enrollment and verification workflows
- **Audit logging** — every verification event recorded for offline QA
- **Device-agnostic capture** — abstract tablet interface with concrete driver implementations

---

## Project Structure

```
├── AGENTS.md                    # Contributor & AI agent rules
├── LICENSE
├── README.md
├── requirements.txt
├── config/
│   └── settings.py              # Hyperparameters, thresholds, paths
├── src/
│   ├── models/
│   │   └── siamese.py           # SpatialSiamese (CNN) + TemporalSiamese (LSTM)
│   │                            # + ContrastiveLoss + FusionEngine
│   ├── data/
│   │   └── dataset.py           # Data loading, preprocessing, quality gates
│   ├── database/
│   │   ├── db_manager.py        # SQLite schema, embedding storage, user management
│   │   └── encryption.py        # AES-256-GCM encryption + audit logging
│   ├── capture/
│   │   └── tablet.py            # Abstract tablet capture + platform drivers
│   └── gui/
│       ├── app.py               # Main application window
│       ├── enrollment.py        # Enrollment panel
│       └── verification.py      # Verification panel
├── scripts/
│   ├── train.py                 # Training entry point
│   └── evaluate.py              # Evaluation & calibration
└── tests/
    ├── test_models.py
    ├── test_database.py
    └── test_capture.py
```

---

## Datasets

### Local Dataset Structure (`./data`)

To train and evaluate the models locally, arrange your datasets inside the `./data` directory following this layout:

```
data/
├── cedar/                  # Offline CEDAR dataset (55 writers)
│   ├── writer_01/
│   │   ├── original_1_1.png ... original_1_24.png  (Genuine signatures)
│   │   └── forged_1_1.png   ... forged_1_24.png    (Skilled Forgeries)
│   └── ...
├── bhsig260/               # Offline BHSig260 dataset (260 writers)
│   ├── Bengali/
│   └── Hindi/
├── utsig/                  # Offline UTSig dataset (115 writers)
├── svc2004_task1/          # Online SVC2004 Task1 (40 writers, 4-col, no pressure)
│   └── U1S1.TXT ... U40S40.TXT
├── svc2004_task2/          # Online SVC2004 Task2 (40 writers, 7-col, real pressure)
│   └── U1S1.TXT ... U40S40.TXT
└── MOBISIG/                # Online MOBISIG (83 users) — temporal pre-training only
    ├── USER1/
    │   ├── SIGN_GEN_USER1_USER1_1.csv   (45 genuine)
    │   └── SIGN_FOR_USER1_USER2_1.csv   (20 skilled forgeries)
    └── ...
```

Directory names are how parsers are selected (`src/data/parsers.py`), so an
unrecognized name raises rather than silently training on zero writers.

### Writer-Independent Data Split Protocol

All model training and validation routines follow an **open-set writer-independent split protocol** compliant with **ISO/IEC 23053:2022**:

- **Partition Ratios**: 70/15/15 for the image corpora, which have 430 writers
  between them. The temporal corpora use **50/25/25**, since SVC2004's 40 writers
  would otherwise leave only 6 test writers and an EER too noisy to compare
  configurations against.
- **Zero Writer Overlap**: Writer IDs are partitioned such that $\text{Writers}_{\text{train}} \cap \text{Writers}_{\text{val}} \cap \text{Writers}_{\text{test}} = \emptyset$. No individual writer appears in more than one partition.
- **Skilled Forgery Benchmark**: Impostor pairs are evaluated strictly using skilled forgeries rather than random signers.

| Dataset                                                                                                            | Modality | Script           | Role                                      |
| ------------------------------------------------------------------------------------------------------------------ | -------- | ---------------- | ----------------------------------------- |
| [CEDAR](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)                                               | Offline  | Latin            | Spatial branch training                   |
| [BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) | Offline  | Bengali / Hindi  | Multi-script generalization               |
| [UTSig](https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification?resource=download)                   | Offline  | Persian / Arabic | Closest proxy to Arabic-script deployment |
| [SVC2004](https://cse.hkust.edu.hk/svc2004/download.html)                                                          | Online   | —                | Temporal branch training                  |
| [MOBISIG](https://www.ms.sapientia.ro/~manyi/mobisig.html)                                                         | Online   | —                | Temporal pre-training (see caveat below)  |
| In-house collection                                                                                                | Both     | Arabic           | Fine-tuning, same-signer multimodal pairs |

**Device targets are tiered.** The primary capture device is the **XPPen Deco 640
pen tablet**, which must reach the 5% FAR/FRR spec. Mobile and iPad capture is
supported as a secondary path where roughly 80% accuracy is acceptable.

That ordering decides how the two online corpora are used. MOBISIG's 83 users
signed with a **finger on a capacitive touchscreen**, so its pressure channel
comes from contact area rather than a pen sensor: it is the proxy for the mobile
and iPad path. SVC2004 is the pen-tablet proxy and therefore the corpus the
headline number is measured on.

SVC2004's 40 writers are too few to train a writer-independent LSTM from scratch,
so MOBISIG supplies volume through **pre-training** and SVC2004 supplies the
matching domain through **fine-tuning**. Training on both with equal weight is
deliberately avoided: it optimizes the average and dilutes pen accuracy in
service of a target that only needs ~20% EER.

```bash
# 1. Pre-train the temporal backbone on MOBISIG's 83 writers
python scripts/train.py --branch temporal --data-dir ./data/MOBISIG \
    --weights-dir models/pretrain

# 2. Fine-tune on SVC2004, which matches the deployment device.
#    --init-weights loads weights only; --resume would also restore optimizer
#    state and the epoch counter, both of which belong to the MOBISIG run.
python scripts/train.py --branch temporal --data-dir ./data/svc2004_task2 \
    --init-weights models/pretrain/temporal_lstm.pth --lr 1e-4

# 3. Measure each target against its own bar, never pooled
python scripts/benchmark.py --branch temporal \
    --weights models/weights/temporal_lstm.pth --data-dir ./data/svc2004_task2   # vs 5%
python scripts/benchmark.py --branch temporal \
    --weights models/weights/temporal_lstm.pth --data-dir ./data/MOBISIG         # vs ~20%
```

---

## Model Weights & GitHub Releases

Compiled PyTorch model weights (`.pth` checkpoint files) are strictly excluded from Git repository tracking via `.gitignore` to keep the repository lightweight:

- **Pre-compiled Weights Distribution**: Official pre-trained weights (`spatial_cnn.pth` and `temporal_lstm.pth`) are hosted and distributed via **[GitHub Releases](https://github.com/Rockeyxx/Signature-Verification-ML-EE499/releases)**.
- **Local Deployment**: Download the compiled weight `.pth` files from the latest GitHub Release and place them into the `models/weights/` directory:
  ```
  models/weights/
  ├── spatial_cnn.pth
  └── temporal_lstm.pth
  ```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Rockeyxx/Signature-Verification-ML-EE499.git
cd Signature-Verification-ML-EE499

python -m venv venv
source venv/bin/activate        # Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Running the Application

```bash
python -m src.gui.app
```

---

## Training

```bash
# Train spatial branch (offline signatures)
python scripts/train.py --branch spatial --data-dir ./data/cedar ./data/bhsig260 ./data/utsig

# Train temporal branch (online signatures)
python scripts/train.py --branch temporal --data-dir ./data/svc2004
```

Training uses **writer-independent splits** — no signer appears in more than one of train / validation / test.

---

## Evaluation

```bash
# Evaluate a single branch
python scripts/evaluate.py --branch spatial \
    --weights ./models/weights/spatial_cnn.pth \
    --data-dir ./data/cedar --output-dir ./results

# α grid search + tri-point report (fusion weight calibration)
python scripts/evaluate.py --branch fused --alpha-sweep \
    --spatial-weights ./models/weights/spatial_cnn.pth \
    --temporal-weights ./models/weights/temporal_lstm.pth \
    --data-dir ./data --output-dir ./results
```

Metrics reported:

- **EER** (Equal Error Rate) per branch and fused
- **FAR / FRR** under skilled-forgery protocol
- **ROC / DET curves** and confusion matrices
- **α-vs-EER sweep** plot for fusion weight selection

---

## Standards & Compliance

| Standard             | Application in This Project                                       |
| -------------------- | ----------------------------------------------------------------- |
| IEEE 730             | Configuration control, deviation log, verification audit log      |
| ISO/IEC 23053:2022   | Writer-independent data splits, documented training configuration |
| ISO/IEC 24745:2022   | Only derived embeddings persisted — never raw biometric data      |
| ISO/IEC 9075-1:2023  | Structured relational SQLite schema                               |
| ISO/IEC 18033-1:2021 | AES-256-GCM encryption of stored embedding payloads               |

---

## Hardware

The capture layer uses an **abstract tablet interface** that supports any digitizer tablet. The reference implementation targets:

| Spec            | Value      |
| --------------- | ---------- |
| Report rate     | ≥ 200 Hz   |
| Resolution      | ≥ 2540 LPI |
| Pressure levels | ≥ 1024     |
| Connection      | USB        |

**Platform capture paths:**

- **Windows:** Wintab API
- **Linux:** evdev / OpenTabletDriver

---

## Team

**Team 06** — Department of Electrical and Computer Engineering, King Abdulaziz University

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- King Abdulaziz University, Faculty of Engineering, Department of Electrical and Computer Engineering
- CEDAR, BHSig260, UTSig, and SVC2004 dataset authors
