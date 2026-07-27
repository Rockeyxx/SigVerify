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

The ICDAR competition dataset is published openly, unlike MCYT / BiosecurID /
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
# the image datasets already cover and cost ~700 MB.
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

- **Partition Ratios**: 70/15/15 for the image datasets, which have 430 writers
  between them. The temporal datasets use **50/25/25**, since SVC2004's 40 writers
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

All figures below are enrollment protocol, K=5, mean ± spread over 10 draws,
seed 42. Report the spread, not one draw — the enrollment draw is a sample just
as the writer split is.

**Spatial branch** — final configuration is inverted pixels, 155×220 direct
resize, 276 balanced pairs per writer, 512-d embedding. The Siamese column is
Dey et al., whose FAR = FRR, so their figures are EER and compare directly. Every
published figure quoted here, with its protocol, is in [References](#references).

| dataset | at session start | **current** | published Siamese (Dey et al.) | best published, any method |
|---|---|---|---|---|
| cedar (Latin) | 3.44% | **0.00% ± 0.00** | 0.00% | 0.00% |
| bhsig_Hindi | 8.15% | **5.55% ± 0.76** | 15.36% | 15.36% |
| bhsig_Bengali | 18.67% | **14.69% ± 1.76** | 13.89% | 13.89% |
| utsig (Persian) | 36.25% | **33.76% ± 1.52** | not evaluated | 11.16% at K = 5, writer-**dependent** |
| pooled (65 writers) | 19.10% | **12.31% ± 0.96** | — | — |

CEDAR is at parity with the reference and Hindi beats it by 9.8 points. CEDAR's
0.00% is measured on 8 held-out writers and 344 pairs — real, and stable across
all 10 draws at Cohen's d 7.18, but a small sample that deserves re-measuring on
more writers before it is quoted as a headline.

UTSig is the one genuine gap, and no Siamese network has been published on it:
its state of the art fits a classifier per enrolled user, a stage this project
measured and rejected because it made results 2–3× worse at 300 training writers.

**Temporal branch** — SVC2004 + SigComp2011, contrastive Siamese LSTM.

| dataset | single-pair | **enrollment, K=5** |
|---|---|---|
| sigcomp_Dutch (pen) | 10.92% | **6.11% ± 1.15** |
| sigcomp_Chinese (pen) | 12.08% | **8.97% ± 0.72** |
| svc (pen) | 29.62% | **21.77% ± 1.67** |
| pooled (31 writers) | 17.98% | **12.73% ± 0.79** |

SigComp2011 Dutch is the dataset that matches the XPPen deployment condition —
real signatures on a pen tablet — and is the closest to the <5% spec.

| Dataset                                                                                                            | Modality | Script           | Role                                      |
| ------------------------------------------------------------------------------------------------------------------ | -------- | ---------------- | ----------------------------------------- |
| [CEDAR](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)                                               | Offline  | Latin            | Spatial branch training                   |
| [BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) | Offline  | Bengali / Hindi  | Multi-script generalization               |
| [UTSig](https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification?resource=download)                   | Offline  | Persian / Arabic | Closest proxy to Arabic-script deployment |
| [SVC2004](https://cse.hkust.edu.hk/svc2004/download.html)                                                          | Online   | —                | Pen-tablet proxy — the headline dataset    |
| [SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) | Online | Dutch / Chinese | Pen-tablet volume (84 writers)            |
| [MOBISIG](https://www.ms.sapientia.ro/~manyi/mobisig.html)                                                         | Online   | —                | Mobile / iPad proxy (see caveat below)    |
| In-house collection                                                                                                | Both     | Arabic           | Fine-tuning, same-signer multimodal pairs |

**Device targets are tiered.** The primary capture device is the **XPPen Deco 640
pen tablet**, which must reach the 5% FAR/FRR spec. Mobile and iPad capture is
supported as a secondary path where roughly 80% accuracy is acceptable.

That ordering decides how the online datasets are used. MOBISIG's 83 users signed
with a **finger on a capacitive touchscreen**, so its pressure channel comes from
contact area rather than a pen sensor: it is the proxy for the mobile and iPad
path. SVC2004 and SigComp2011 are both **pen-tablet** datasets and are where the
primary target is measured.

Two things were measured rather than assumed, and both shape the recipe:

- **Touchscreen writers do not substitute for pen writers.** Pre-training on
  MOBISIG and fine-tuning on SVC2004 *lost* 2.6 EER points against training from
  scratch. Tripling the writer count with MOBISIG moved the mobile number 14
  points and the pen number not at all.
- **SVC2004 is much harder than other pen datasets, and not because of the
  device.** One training run scores 10.92% on SigComp2011's Dutch writers and
  29.62% on SVC2004. SVC2004's contributors were advised not to use their real
  signatures, and its forgers were shown the signing process on a monitor and
  allowed to practise. Always report the two datasets separately.

The temporal branch is therefore trained on **pen datasets together, from
scratch** — no pre-training stage.

```bash
# 1. Train on the pen datasets together. SigComp2011 supplies 84 pen writers,
#    taking the trainable count from 20 to 62. expandable_segments avoids
#    fragmentation OOM on a 4 GB card.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train.py --branch temporal \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 \
    --weights-dir models/pen_sigcomp --epochs 120 --patience 20 --seed 42

# 2. Score on the same datasets and seed, which reproduces the same held-out
#    writers. Read the per-dataset breakdown, never the pooled number alone.
python scripts/benchmark.py --branch temporal \
    --weights models/pen_sigcomp/temporal_lstm.pth \
    --data-dir ./data/svc2004_task2 ./data/sigcomp2011 --seed 42

# 3. Mobile is a separate bar (~20%), measured on its own dataset.
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
from whichever datasets are loaded, so adding a dataset reshuffles it and the EERs
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

## References

The published figures this project is measured against, with the protocol each
one used. Every "this project" number comes from `benchmark.py --enroll 5`: mean
over 10 draws, seed 42, held-out writers only.

### What the protocol column means

Three things move these numbers more than any architecture choice does, so every
row says which of them applies.

- **WI vs WD.** *Writer-independent* systems learn one distance and threshold it
  for everybody. *Writer-dependent* systems fit a classifier per enrolled user.
  Nearly every published UTSig and MCYT result is WD. This project is WI, and it
  measured the WD stage directly: a per-user linear SVM took the spatial branch
  from 13.90% to 43.89% EER (`docs/SOLUTIONS.md`). With 300 training writers
  rather than thousands, random-forgery negatives cannot teach a skilled-forgery
  boundary. Part of the distance to a WD number is protocol, not quality.
- **Global vs user-specific threshold.** A threshold tuned per user *after*
  seeing that user's scores measures something else. MOBISIG's headline 5.81% is
  a posteriori user-specific; the same work reports 14.31% under a global
  threshold. This project reports global-threshold EER only.
- **Enrolled references (K).** More references make the task easier. K appears on
  every row whose source states it. This project uses K = 5 throughout.

Dey et al. report FAR = FRR, so their accuracies convert straight across:
EER = 100 − accuracy.

### Published results on the datasets this project uses

| Dataset | Published | Protocol | Source | This project |
| --- | --- | --- | --- | --- |
| CEDAR | **0.00%** EER (100.00% acc) | WI Siamese CNN, 50 of 55 writers trained | [1] | **0.00% ± 0.00** |
| CEDAR | 4.63% EER | WD SVM on CNN features, K = 12 | [2] | **0.00% ± 0.00** |
| BHSig260 Bengali | **13.89%** EER (86.11% acc) | WI Siamese CNN, 50 of 100 writers trained | [1] | 14.69% ± 1.76 |
| BHSig260 Hindi | 15.36% EER (84.64% acc) | WI Siamese CNN, 100 of 160 writers trained | [1] | **5.55% ± 0.76** |
| UTSig | **9.80%** EER | WD SVM, transfer from Persian handwriting, K = 10 | [3] | 33.76% ± 1.52 |
| UTSig | 11.16% EER | same, K = 5 — the closest protocol match to ours | [3] | 33.76% ± 1.52 |
| UTSig | 17.45% EER | WD DMML on HOG features, K = 12 | [4] | 33.76% ± 1.52 |
| UTSig | 29.71% EER | WD SVM on fixed-point geometrics, K = 12 | [5] | 33.76% ± 1.52 |
| SVC2004 Task 2 | **2.89%** EER | competition winner of 8 teams, K = 5, skilled forgeries | [6] | 21.77% ± 1.67 |
| SVC2004 Task 2 | 77.00% acc | WI Siamese CNN, 35 writers trained / 5 tested | [7] | 21.77% ± 1.67 |
| SigComp2011 online, Dutch | not established here | competition, reference-set matching | [8] | **6.11% ± 1.15** |
| SigComp2011 online, Chinese | not established here | competition, reference-set matching | [8] | 8.97% ± 0.72 |
| MOBISIG | 5.81% EER | DTW, a posteriori **user-specific** thresholds | [9] | 24.09% |
| MOBISIG | 14.31% EER | global features, **global** threshold | [9] | 24.09% |

Two datasets carry a real gap: UTSig and SVC2004. Both have a known cause. Every
published UTSig result is writer-dependent with 10 to 12 references. SVC2004's
contributors were told not to use their real signatures, and its forgers watched
the signing process on a monitor and were allowed to practise, which is why one
temporal model scores 29.62% there and 10.92% on SigComp2011's Dutch pen writers.

Reference [8] reports the SigComp2011 online results per system. Those figures
are left out here because the tables circulating in secondary sources could not
be confirmed as the online task rather than the offline one, and quoting the
wrong one would compare this project against a different measurement.

### How the reference architecture differs

The spatial branch is measured against [1], which is the same architecture class
(Siamese CNN, contrastive loss, margin 1). Its configuration, from the paper's
own tables:

| | Dey et al. SigNet [1] | this project |
| --- | --- | --- |
| input | 155×220, inverted, ÷ std | 155×220, inverted, ÷ 255 |
| pairs per writer | 276 genuine + 276 forged | 276 + 276 |
| conv stack | 96×11×11 → 256×5×5 → 384×3×3 → 256×3×3 | 32×5×5 → 64 → 128 → 256 → 256 |
| normalization | local response norm | GroupNorm |
| head | FC-1024 (dropout 0.5) → FC-128 | 4×4 avg+max pool (8192) → FC-512 |
| dropout | 0.3 after pools 2 and 4, 0.5 after FC1 | none |
| optimizer | RMSprop 1e-4, momentum 0.9, wd 5e-4 | Adam 1e-3, wd 1e-4 |
| batch / epochs | 128 / 20 | 32 / early-stopped |
| loss | contrastive, margin 1 | contrastive, margin 1 + hard-negative mining |

**No published Siamese system reaches 95% accuracy on all of these datasets.**
SigNet's own GPDS300 result is 76.83% (23.17% EER) and its Bengali result 86.11%.
The only 100% in the table is CEDAR, measured on 5 held-out writers. Nobody has
reached 95% everywhere with this architecture class, so the shortfall belongs to
the target rather than to this implementation of it.

### Bibliography

**Methods**

1. S. Dey, A. Dutta, J. I. Toledo, S. K. Ghosh, J. Lladós, U. Pal.
   *SigNet: Convolutional Siamese Network for Writer Independent Offline
   Signature Verification.* arXiv:1707.02131, 2017.
   <https://arxiv.org/abs/1707.02131>
2. L. G. Hafemann, R. Sabourin, L. S. Oliveira. *Learning features for offline
   handwritten signature verification using deep convolutional neural networks.*
   Pattern Recognition 70:163–176, 2017. doi:10.1016/j.patcog.2017.05.012 —
   preprint arXiv:1705.05787. Writer-ID training plus a per-user SVM, **not** a
   Siamese network; cited for its CEDAR/MCYT numbers, not as an architecture model.
3. O. Mersa, F. Etaati, S. Masoudnia, B. N. Araabi. *Learning Representations
   from Persian Handwriting for Offline Signature Verification, a Deep Transfer
   Learning Approach.* arXiv:1903.06249, 2019. <https://arxiv.org/abs/1903.06249>
4. A. Soleimani, B. N. Araabi, K. Fouladi. *Deep Multitask Metric Learning for
   Offline Signature Verification.* Pattern Recognition Letters 80:84–90, 2016.
5. A. Soleimani, K. Fouladi, B. N. Araabi. *UTSig: A Persian offline signature
   dataset.* IET Biometrics 6(1):1–8, 2017. Dataset paper and its own baseline.
6. D.-Y. Yeung, H. Chang, Y. Xiong, S. George, R. Kashi, T. Matsumoto, G. Rigoll.
   *SVC2004: First International Signature Verification Competition.* ICBA 2004,
   LNCS 3072, pp. 16–22. <https://cse.hkust.edu.hk/svc2004/icba2004paper.pdf>
7. C. Sekhar, P. Mukherjee, D. S. Guru, V. Pulabaigari. *OSVNet: Convolutional
   Siamese Network for Writer Independent Online Signature Verification.*
   ICDAR 2019 — arXiv:1904.00240. <https://arxiv.org/abs/1904.00240>
   The temporal branch's closest published Siamese counterpart.
8. M. Liwicki, M. I. Malik, C. E. van den Heuvel, X. Chen, C. Berger, R. Stoel,
   M. Blumenstein, B. Found. *Signature Verification Competition for Online and
   Offline Skilled Forgeries (SigComp2011).* ICDAR 2011, pp. 1480–1484.
   doi:10.1109/ICDAR.2011.294
9. M. Antal, L. Z. Szabó, T. Tordai. *Online Signature Verification on MOBISIG
   Finger-Drawn Signature Corpus.* Mobile Information Systems, vol. 2018,
   Article ID 3127042. doi:10.1155/2018/3127042

**Datasets**

10. M. K. Kalera, S. Srihari, A. Xu. *Offline signature verification and
    identification using distance statistics.* IJPRAI 18(7):1339–1360, 2004. — CEDAR.
11. S. Pal, A. Alaei, U. Pal, M. Blumenstein. *Performance of an off-line
    signature verification method based on texture features on a large
    indic-script signature dataset.* DAS 2016, pp. 72–77. — BHSig260.

**Standards**

12. ISO/IEC 23053:2022 — framework for AI systems using machine learning
    (writer-independent splits, documented training configuration).
13. ISO/IEC 24745:2022 — biometric information protection (only derived
    embeddings are persisted).
14. ISO/IEC 18033-1:2021 — encryption algorithms (AES-256-GCM at rest).
15. IEEE 730-2014 — software quality assurance processes (configuration control,
    deviation log, verification audit log).

---

## Team

**Team 06** — Department of Electrical and Computer Engineering, King Abdulaziz University

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- King Abdulaziz University, Faculty of Engineering, Department of Electrical and Computer Engineering
- The CEDAR, BHSig260, UTSig, SVC2004, SigComp2011 and MOBISIG dataset authors —
  cited individually under [References](#references)
