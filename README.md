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
├── MOBISIG/                # Online MOBISIG (83 users) — mobile/iPad proxy
│   ├── USER1/
│   │   ├── SIGN_GEN_USER1_USER1_1.csv   (45 genuine)
│   │   └── SIGN_FOR_USER1_USER2_1.csv   (20 skilled forgeries)
│   └── ...
└── sigcomp2011/            # Online SigComp2011 (84 pen-tablet writers, 3695 signatures)
    ├── trainingSet/OnlineSignatures/{Dutch,Chinese}/TrainingSet/
    │   ├── Online Genuine/     001_01.HWR       (writer_index)
    │   └── Online Forgeries/   170_001_02.HWR   (forger_writer_index)
    └── Testdata_SigComp2011/SigComp11-Onlinetestset/{Dutch,Chinese}/
        ├── Reference(648)/013/   13_013.HWR     (index_writer — reversed!)
        └── Questioned(1259)/013/ 01_182_013.HWR (index_forger_writer)
```

Directory names are how parsers are selected (`src/data/parsers.py`), so an
unrecognized name raises rather than silently training on zero writers.

#### Fetching SigComp2011 (no licence request needed)

The ICDAR competition corpus is published openly, unlike MCYT / BiosecurID /
DeepSignDB, which all require a signed agreement and a wait of days to weeks.
The zip password is printed on the dataset page itself. The original host
(`iapr-tc11.org`) has been unreachable, so the commands below pull the archived
copies; substitute the live host if it returns.

```bash
mkdir -p data/sigcomp2011
BASE=https://web.archive.org/web/20250907124749if_/http://www.iapr-tc11.org/dataset/ICDAR_SignatureVerification/SigComp2011
curl -L -o /tmp/sc11-train.zip "$BASE/sigComp2011-trainingSet.zip"   # 234 MB
curl -L -o /tmp/sc11-test.zip  "$BASE/sigComp2011-test.zip"          # 540 MB

# Only the online trajectories are needed; the offline PNGs duplicate ground
# the image corpora already cover and cost ~700 MB.
PW="I hereby accept the SigComp 2011 disclaimer."
unzip -P "$PW" /tmp/sc11-train.zip 'trainingSet/OnlineSignatures/*' -d data/sigcomp2011
unzip -P "$PW" /tmp/sc11-test.zip  'Testdata_SigComp2011/SigComp11-Onlinetestset/*' \
    -d data/sigcomp2011
```

Wayback serves the 540 MB test archive slowly and returns `504` on a cold
request — retry, it succeeds once the record is warm. Verify with
`pytest tests/test_parsers.py -k sigcomp`, which asserts the documented 84
writers and 3695 signatures. A training-set-only download parses fine and simply
yields 20 writers, so a partial fetch degrades rather than fails.

### Writer-Independent Data Split Protocol

All model training and validation routines follow an **open-set writer-independent split protocol** compliant with **ISO/IEC 23053:2022**:

- **Partition Ratios**: 70/15/15 for the image corpora, which have 430 writers
  between them. The temporal corpora use **50/25/25**, since SVC2004's 40 writers
  would otherwise leave only 6 test writers and an EER too noisy to compare
  configurations against.
- **Zero Writer Overlap**: Writer IDs are partitioned such that $\text{Writers}_{\text{train}} \cap \text{Writers}_{\text{val}} \cap \text{Writers}_{\text{test}} = \emptyset$. No individual writer appears in more than one partition.
- **Skilled Forgery Benchmark**: Impostor pairs are evaluated strictly using skilled forgeries rather than random signers.
- **Enrollment protocol** (`--enroll 5`): the standard the field measures with, and
  what `src/verification.py` runs. Each user enrolls **5 genuine** signatures,
  skilled forgeries are **never** shown at enrollment, each probe is matched
  against the nearest enrolled reference, and the result is **averaged over 10
  random enrollment draws**. This is the SVC2004 competition protocol, whose best
  Task 2 result was 2.89% EER.

```bash
# The standard protocol — the headline numbers
python scripts/benchmark.py --branch temporal --weights models/weights/temporal_lstm.pth \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 --seed 42 --enroll 5

# Without --enroll the harness scores isolated pairs: a harder task that neither
# the literature nor this system performs. Kept as a conservative lower bound.
```

| branch / corpus | single-pair | **enrollment, K=5** |
|---|---|---|
| spatial pooled (65 writers) | 19.10% | **13.90% ± 0.85** |
| cedar (Latin) | 3.44% | **4.86% ± 2.14** |
| bhsig_Hindi | 8.15% | **6.70% ± 0.96** |
| bhsig_Bengali | 18.67% | **12.70% ± 1.46** |
| utsig (Persian) | 36.25% | **34.24% ± 1.90** |
| temporal pooled (31 writers) | 17.98% | **12.73% ± 0.79** |
| sigcomp_Dutch (pen) | 10.92% | **6.11% ± 1.15** |
| sigcomp_Chinese (pen) | 12.08% | **8.97% ± 0.72** |
| svc (pen) | 29.62% | **21.77% ± 1.67** |

Report the spread, not one draw. Which 5 signatures a user happens to enroll moves
CEDAR between 2.03% and 8.70%, so a single draw on an 8-writer corpus is not a
result.

| Dataset                                                                                                            | Modality | Script           | Role                                      |
| ------------------------------------------------------------------------------------------------------------------ | -------- | ---------------- | ----------------------------------------- |
| [CEDAR](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)                                               | Offline  | Latin            | Spatial branch training                   |
| [BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) | Offline  | Bengali / Hindi  | Multi-script generalization               |
| [UTSig](https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification?resource=download)                   | Offline  | Persian / Arabic | Closest proxy to Arabic-script deployment |
| [SVC2004](https://cse.hkust.edu.hk/svc2004/download.html)                                                          | Online   | —                | Pen-tablet proxy — the headline corpus    |
| [SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) | Online | Dutch / Chinese | Pen-tablet volume (84 writers)            |
| [MOBISIG](https://www.ms.sapientia.ro/~manyi/mobisig.html)                                                         | Online   | —                | Mobile / iPad proxy (see caveat below)    |
| In-house collection                                                                                                | Both     | Arabic           | Fine-tuning, same-signer multimodal pairs |

**Device targets are tiered.** The primary capture device is the **XPPen Deco 640
pen tablet**, which must reach the 5% FAR/FRR spec. Mobile and iPad capture is
supported as a secondary path where roughly 80% accuracy is acceptable.

That ordering decides how the online corpora are used. MOBISIG's 83 users signed
with a **finger on a capacitive touchscreen**, so its pressure channel comes from
contact area rather than a pen sensor: it is the proxy for the mobile and iPad
path. SVC2004 and SigComp2011 are both **pen-tablet** corpora and are where the
primary target is measured.

Two things were measured rather than assumed, and both shape the recipe:

- **Touchscreen writers do not substitute for pen writers.** Pre-training on
  MOBISIG and fine-tuning on SVC2004 *lost* 2.6 EER points against training from
  scratch. Tripling the writer count with MOBISIG moved the mobile number 14
  points and the pen number not at all.
- **SVC2004 is much harder than other pen corpora, and not because of the
  device.** One training run scores 10.92% on SigComp2011's Dutch writers and
  29.62% on SVC2004. SVC2004's contributors were advised not to use their real
  signatures, and its forgers were shown the signing process on a monitor and
  allowed to practise. Always report the two corpora separately.

The temporal branch is therefore trained on **pen corpora together, from
scratch** — no pre-training stage.

```bash
# 1. Train on the pen corpora together. SigComp2011 supplies 84 pen writers,
#    taking the trainable count from 20 to 62. expandable_segments avoids
#    fragmentation OOM on a 4 GB card.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train.py --branch temporal \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 \
    --weights-dir models/pen_sigcomp --epochs 120 --patience 20 --seed 42

# 2. Score on the same corpora and seed, which reproduces the same held-out
#    writers. Read the per-corpus breakdown, never the pooled number alone.
python scripts/benchmark.py --branch temporal \
    --weights models/pen_sigcomp/temporal_lstm.pth \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 --seed 42

# 3. Mobile is a separate bar (~20%), measured on its own corpus.
python scripts/benchmark.py --branch temporal \
    --weights models/pen_sigcomp/temporal_lstm.pth --data-dir ./data/MOBISIG
```

| temporal model | pooled | svc | sigcomp_Dutch | sigcomp_Chinese | MOBISIG |
|---|---|---|---|---|---|
| MOBISIG + SVC2004 | 24.31% | 26.75% | 18.17% | 23.75% | 20.13% |
| **SVC2004 + SigComp2011 (ships)** | **17.98%** | 29.62% | **10.92%** | **12.08%** | 24.09% |

The first four columns come from one identical 31-writer held-out split, so they
are directly comparable. Note that the older model had 3 of those 10 SVC test
writers in its *training* set, which flatters its SVC column.

**Comparing two temporal checkpoints requires care.** The writer split is derived
from whichever corpora are loaded, so adding a corpus reshuffles it and the EERs
stop being comparable. A third run including MOBISIG appeared to cut SVC2004 from
29.62% to 16.92%; scored on the 5 SVC writers both runs had actually held out,
the real gap was 33.00% against 29.50%. Use `--writers` for this, which refuses
any writer the run being scored trained on:

```bash
python scripts/benchmark.py --branch temporal --weights <a.pth> \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 --seed 42 \
    --writers svc_3 svc_6 svc_7 svc_34 svc_38
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
