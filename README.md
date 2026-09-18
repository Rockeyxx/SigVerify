# Signature Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)]()

> **Senior Design Project (EE 499)** — Department of Electrical and Computer Engineering, King Abdulaziz University (KAU), Jeddah, Saudi Arabia — Team 06

A multimodal signature verification system built with dual Siamese deep neural networks. It verifies signatures using both **offline** (scanned images) and **online** (dynamic tablet pen strokes) inputs, combining their scores via late fusion to detect genuine signatures and skilled forgeries.

This document covers installation, quickstart, project structure, evaluation benchmarks, and architectural comparisons. Comprehensive experiment logs, protocol derivations, and ablation histories are maintained in [SUMMARY.md](SUMMARY.md).

---

## Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Input Specifications](#input-specifications)
  - [Enrollment and Decision](#enrollment-and-decision)
  - [Model Weights & GitHub Releases](#model-weights--github-releases)
- [Training](#training)
  - [Datasets & Local Directory Layout](#datasets--local-directory-layout)
- [Evaluation](#evaluation)
  - [Results vs Published Literature](#results-vs-published-literature)
  - [Temporal Branch — Extended Pre-Training with DeepSignDB](#temporal-branch--extended-pre-training-with-deepsigndb)
  - [Datasets and Device Targets](#datasets-and-device-targets)
  - [Evaluation Figures & Benchmark Metrics](#evaluation-figures--benchmark-metrics)
  - [Operating Point & Confusion Matrices](#operating-point--confusion-matrices)
  - [ROC and DET Curves](#roc-and-det-curves)
- [Standards & Compliance](#standards--compliance)
- [Hardware](#hardware)
- [References](#references)
  - [How the Reference Architecture Differs](#how-the-reference-architecture-differs)
  - [Bibliography](#bibliography)
- [Team](#team)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Probe Signature Input                    │
│              (Image / Dynamic / Both)                       │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   Spatial Branch    │      │    Temporal Branch      │
│   Siamese CNN       │      │    Siamese LSTM         │
│                     │      │                         │
│  [1, 1, 155, 220]   │      │  [1, L, 4] (X,Y,P,dt)   │
│       ↓             │      │       ↓                 │
│  512-d embedding    │      │  128-d embedding        │
│       ↓             │      │       ↓                 │
│  S_spatial          │      │  S_temporal             │
└──────────┬──────────┘      └──────────┬──────────────┘
           │                              │
           └──────────┬───────────────────┘
                      ▼
         ┌──────────────────────┐
         │   Late Fusion        │
         │                      │
         │  S = α·S_sp + (1-α)·S_t │
         │  S ≥ τ → Genuine     │
         │  S < τ → Forged      │
         └──────────┬───────────┘
                    ▼
         [Genuine/Forged, Confidence %]
```

**Single-Modality Fallback:** When only one input modality is provided, the system automatically routes to that branch ($\alpha = 1.0$ for spatial image verification, $\alpha = 0.0$ for temporal trajectory verification).

---

## Features

**Implemented and Measured — Machine Learning Pipeline:**

- **Dual-Modality Architecture** — Combines spatial visual representations (offline scanned images) with kinematic stroke dynamics (online tablet trajectories).
- **Late-Fusion Engine** — Weighted score fusion with automatic single-modality fallback (`src/models/siamese.py`).
- **Few-Shot Enrollment Scoring** — Probes matched against the nearest of $K=5$ stored reference embeddings (`src/verification.py`).
- **Writer-Independent Evaluation** — Strict held-out-writer partitioning per ISO/IEC 23053:2022 with comprehensive EER, FAR, FRR, and DET metrics (`scripts/benchmark.py`, `scripts/evaluate.py`).
- **Ultra-Low Latency Inference** Fast CPU execution (0.07 s warm, 0.94 s cold) and light memory footprint (~0.45 GB RAM), well within embedded budgets (< 3.0 s, ≤ 4 GB).
- **Cross-Platform Compatibility** Runs natively on Windows and Linux (x86_64 and ARM64) without native build toolchains.

**Application Layer — Planned, Not Yet Implemented:**

`src/gui/`, `src/capture/`, and `src/database/` are interface stubs — docstrings and `# TODO` markers, no executable logic. They are not evidence of delivered functionality or standards compliance; see [Standards & Compliance](#standards--compliance).

- **PyQt5 Desktop Interface** (`src/gui/`) Planned interactive enrollment and verification workflows.
- **Digitizer Tablet Abstraction** (`src/capture/`) Planned hardware abstraction layer for USB digitizer pads (e.g., XPPen Deco 640) via Wintab (Windows) and evdev / OpenTabletDriver (Linux).
- **Encrypted Biometric Store** (`src/database/`) Planned SQLite template store enforcing AES 256 GCM application-layer encryption and ISO/IEC 24745 compliance (only derived embeddings persisted).

---

## Project Structure

```
├── AGENTS.md                    # Contributor & AI agent directives (hard architectural rules)
├── SUMMARY.md                   # System walkthrough, protocol derivations & negative results
├── DATASET_LICENSES.md          # Dataset licenses, redistribution terms, and restrictions
├── LICENSE                      # MIT License
├── README.md                    # Project documentation, quickstart & benchmark summary
├── requirements.txt             # Python dependencies
├── Signature_Verification_Pipeline.ipynb   # Interactive pipeline demo notebook (outputs stripped)
├── config/
│   └── settings.py              # Central hyperparameters, paths, and decision thresholds
├── src/
│   ├── verification.py          # Multimodal verification engine & quality gates
│   ├── models/
│   │   └── siamese.py           # SpatialSiamese (CNN), TemporalSiamese (LSTM), Loss, Fusion
│   ├── data/
│   │   ├── dataset.py           # Preprocessing, quality gate, pair generation, splits
│   │   ├── parsers.py           # Loaders for CEDAR, BHSig260, UTSig, SVC2004, SigComp2011
│   │   └── features.py          # Kinematic feature extraction (deltas, accelerations)
│   ├── database/                # STUB — interface docstrings + TODOs only
│   │   ├── db_manager.py
│   │   └── encryption.py
│   ├── capture/                 # STUB — interface docstrings + TODOs only
│   │   └── tablet.py
│   └── gui/                     # STUB — interface docstrings + TODOs only
│       ├── app.py
│       ├── enrollment.py
│       └── verification.py
├── scripts/
│   ├── train.py                 # Training entry point (spatial, temporal, or joint)
│   ├── evaluate.py              # Evaluation, threshold calibration, figure generation
│   └── benchmark.py             # Held-out writer benchmark & Cohen's d regression harness
├── tools/                       # Latency profiling and interactive test harnesses
├── models/weights/              # PyTorch .pth weight checkpoints (distributed via GitHub Releases)
├── results/                     # Generated ROC, DET, and confusion matrix figures
└── tests/                       # Automated test suite (unit, regression, and integration tests)
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` package manager

### Installation

```bash
git clone https://github.com/Rockeyxx/Signature-Verification-ML-EE499.git
cd Signature-Verification-ML-EE499

python -m venv venv
source venv/bin/activate        # On Linux / macOS
# venv\Scripts\activate         # On Windows

pip install -r requirements.txt
```

### Running the Application

```bash
# Score a branch on held-out writers under the few-shot enrollment protocol (K=5)
python scripts/benchmark.py --branch temporal \
    --weights models/weights/temporal_lstm.pth --temporal-channels dt \
    --data-dir data/svc2004_task2 data/sigcomp2011 --enroll 5

# Regenerate ROC, DET, and confusion matrix plots
python scripts/evaluate.py --branch spatial \
    --weights models/weights/spatial_cnn.pth \
    --data-dir data/cedar data/bhsig260 data/utsig --output-dir results

# Profile CPU latency and RAM footprint
python tools/profile_latency.py --branch fused

# Test your own signature images against the shipped spatial model
python tools/verify_my_signature.py --crop \
    --references mine/ref1.jpg mine/ref2.jpg mine/ref3.jpg \
    --probes mine/genuine.jpg mine/forged.jpg
```

---

### Input Specifications

- **Spatial Branch (Offline):** Single-channel grayscale images formatted to `[1, 1, 155, 220]`. Inverted pixel polarity (ink high, background 0), divided by 255. Bounding box quality gate requires `min(w, h) >= 64 px` and Laplacian variance `≥ 50.0`.
- **Temporal Branch (Online):** Trajectory tensors formatted to `[1, L, 4]` containing $(X, Y, P, dt)$. Coordinates $X$ and $Y$ are z-scored per signature; pressure $P$ is scaled to $[0.0, 1.0]$; $dt$ represents elapsed time between consecutive points (clipped at 100 ms and normalized to $[0.0, 1.0]$).

#### The `dt` Channel

`dt` represents the time delta between sampled points (admitted 2026-08-19, widening the input tensor from 3 to 4 channels).

Without timing deltas, uniform-rate models lose visibility into pen pauses and air-travel hesitations. Genuine signers execute signatures with continuous motor memory rhythm, whereas forgers hesitate and trace strokes slowly. `dt` remains ~10 ms during continuous pen-down contact and jumps significantly at pen lifts. Across six paired random seeds, adding `dt` improved pooled temporal EER from **10.45% to 8.97%** ($t = 3.25$).

---

### Enrollment and Decision

Verification evaluates a probe against an enrolled gallery using nearest-reference matching:

| Parameter                 | Value                                                                | Configuration Constant                                      |
| ------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| References per user ($K$) | **5**                                                                | `NUM_ENROLLMENT_REFERENCES = 5`                             |
| Matching Rule             | Nearest Reference: $d_{min} = \min_{i} \|e_{probe} - e_{ref, i}\|_2$ | `_nearest_reference_score`                                  |
| Similarity Score          | $S = \frac{1}{1 + d_{min}}$                                          | —                                                           |
| Decision Threshold        | Spatial $\tau = 0.7731$, Temporal $\tau = 0.7900$                    | `SPATIAL_DECISION_THRESHOLD`, `TEMPORAL_DECISION_THRESHOLD` |

Increasing enrolled references improves accuracy without retraining, with diminishing returns. Swept on the shipped weights (`benchmark.py --enroll K`, K = 1 to 15, 10 draws each): spatial pooled EER 11.71% at $K=1$, 8.72% at $K=5$, 7.86% at $K=10$, 7.82% at $K=15$; temporal 13.72%, 7.81%, 6.99%, 6.31%. Both curves are flat past $K \approx 10$ (Hindi 4.65%, SigComp2011 Dutch 3.53% at $K=10$). $K=5$ ships; see SUMMARY.md for why.

---

### Model Weights & GitHub Releases

Compiled PyTorch weight checkpoints (`.pth` files) are tracked outside Git via `.gitignore` to keep the repository compact:

- **Pre-trained Weights Distribution:** Official release weights (`spatial_cnn.pth` and `temporal_lstm.pth`) are hosted on **[GitHub Releases](https://github.com/Rockeyxx/Signature-Verification-ML-EE499/releases)**.
- **Optional accuracy tier:** six `temporal_dt_s4{2..7}.pth` checkpoints. Enabling
  them via `SignatureService(use_temporal_ensemble=True)` averages their
  embeddings and lowers temporal EER from **8.83% to 6.94%** on DeepSignDB
  Evaluation (372 held-out writers, 25 draws, winning 25/25). Costs 6x inference
  — about 32 ms against the 3.0 s budget. The service **raises** if any member is
  missing rather than quietly running one model.
- **Local Placement:** Download the compiled weights from the release page and place them into `models/weights/`:
  ```
  models/weights/
  ├── spatial_cnn.pth
  └── temporal_lstm.pth
  ```

> **DeepSignDB Licensing Notice:** Weights trained on DeepSignDB (`DONT_SHIP_*.pth`) are strictly restricted to local academic research under DeepSignDB license clause 4.4 (commercial redistribution restriction). The publicly distributed weights are trained exclusively on open academic datasets (CEDAR, BHSig260, UTSig, SVC2004, and SigComp2011).

---

## Training

Train models from scratch across single or multiple datasets using writer-independent splits (70% train / 15% val / 15% test):

```bash
# Train spatial branch (offline images)
python scripts/train.py --branch spatial \
    --data-dir data/cedar data/bhsig260 data/utsig \
    --epochs 100 --batch-size 32 --seed 42

# Train temporal branch (online pen trajectories)
python scripts/train.py --branch temporal --temporal-channels dt \
    --data-dir data/svc2004_task2 data/sigcomp2011 \
    --epochs 100 --batch-size 32 --seed 42
```

### Datasets & Local Directory Layout

Organize downloaded datasets in the `./data` directory according to the following layout (dataset directory names automatically determine the corresponding parser in `src/data/parsers.py`):

```
data/
├── cedar/                  # Offline CEDAR (55 writers)
│   ├── writer_01/
│   │   ├── original_1_1.png ... original_1_24.png  (Genuine)
│   │   └── forged_1_1.png   ... forged_1_24.png    (Skilled forgeries)
│   └── ...
├── bhsig260/               # Offline BHSig260 (260 writers)
│   ├── Bengali/
│   └── Hindi/
├── utsig/                  # Offline UTSig (115 writers, Persian script)
├── svc2004_task1/          # Online SVC2004 Task 1 (40 writers, 4 columns, no pressure)
├── svc2004_task2/          # Online SVC2004 Task 2 (40 writers, 7 columns, real pressure)
├── MOBISIG/                # Online MOBISIG (83 writers) — mobile / touch proxy
└── sigcomp2011/            # Online SigComp2011 (84 writers, Dutch & Chinese)
    ├── trainingSet/OnlineSignatures/{Dutch,Chinese}/
    └── Testdata_SigComp2011/SigComp11-Onlinetestset/{Dutch,Chinese}/
```

Refer to [DATASET_LICENSES.md](DATASET_LICENSES.md) for terms of use, licensing, and acquisition details.

---

## Evaluation

### Results vs Published Literature

The table below benchmarks this project against published state-of-the-art results across standard datasets. All **This project** metrics represent held-out writer evaluations under $K=5$ enrollment (mean $\pm$ standard deviation across 10 random enrollment draws, seed 42):

| Dataset                 | Published EER / Acc | Published Protocol & Setup                               | Source | This Project (Mean $\pm$ SD) |
| ----------------------- | :-----------------: | -------------------------------------------------------- | :----: | :--------------------------: |
| **CEDAR**               |    **0.00%** EER    | Writer-Independent Siamese CNN (50 train / 5 test)       |  [1]   |    **0.00% $\pm$ 0.00%**     |
| **CEDAR**               |      4.63% EER      | Writer-Dependent SVM on CNN features ($K=12$)            |  [2]   |    **0.00% $\pm$ 0.00%**     |
| **BHSig260 Bengali**    |   **13.89%** EER    | Writer-Independent Siamese CNN (50 train / 50 test)      |  [1]   |    **8.17% $\pm$ 1.07%**     |
| **BHSig260 Hindi**      |     15.36% EER      | Writer-Independent Siamese CNN (100 train / 60 test)     |  [1]   |    **5.33% $\pm$ 0.50%**     |
| **UTSig (Persian)**     |    **9.80%** EER    | Writer-Dependent SVM, Persian transfer learning ($K=10$) |  [3]   |      28.10% $\pm$ 3.21%      |
| **UTSig (Persian)**     |     11.16% EER      | Writer-Dependent SVM, Persian transfer learning ($K=5$)  |  [3]   |      28.10% $\pm$ 3.21%      |
| **UTSig (Persian)**     |     17.45% EER      | Writer-Dependent DMML on HOG features ($K=12$)           |  [4]   |      28.10% $\pm$ 3.21%      |
| **UTSig (Persian)**     |     29.71% EER      | Writer-Dependent SVM on geometric features ($K=12$)      |  [5]   |      28.10% $\pm$ 3.21%      |
| **SVC2004 Task 2**      |    **2.89%** EER    | Competition winner, user-specific thresholds ($K=5$)     |  [6]   |      13.58% $\pm$ 2.21%      |
| **SVC2004 Task 2**      |     77.00% Acc      | Writer-Independent Siamese CNN (35 train / 5 test)       |  [7]   |      13.58% $\pm$ 2.21%      |
| **SigComp2011 Dutch**   |    _Competition_    | Multi-system reference benchmark                         |  [8]   |    **4.11% $\pm$ 1.03%**     |
| **SigComp2011 Chinese** |    _Competition_    | Multi-system reference benchmark                         |  [8]   |    **5.28% $\pm$ 0.85%**     |
| **MOBISIG (Touch)**     |      5.81% EER      | DTW with user-specific thresholds                        |  [9]   |       21.26% $\dagger$       |
| **MOBISIG (Touch)**     |     14.31% EER      | Global features with global threshold                    |  [9]   |       21.26% $\dagger$       |

$\dagger$ _Note: MOBISIG figures reflect a separate touchscreen evaluation model. The primary shipped temporal weights are trained exclusively on digitizer pen corpora (SVC2004 Task 2 + SigComp2011)._

#### Analysis of Protocol Differences & Accuracy Gaps

- **UTSig Protocol Gap:** Published results achieving 9–17% EER on UTSig rely on **writer-dependent** training (training per-user classifiers with $K=10$ to $12$ references). In our strictly writer-independent, few-shot enrollment setup, Persian signatures exhibit complex stroke loops that yield 28.10% EER.
- **SVC2004 Intrinsic Hardness:** Published SVC2004 benchmarks below 3% utilize _a posteriori_ user-specific thresholds. Under a global threshold, SVC2004 is intrinsically challenging: DTW alignment confirms an intra-genuine distance ratio of 1.44 (vs 1.61 for Dutch) and a median forgery separation $z$-score of 1.37 (vs 2.23 for Dutch).
- **SigComp2011 Target Tier:** On true pen digitizers (SigComp2011 Dutch), our temporal model achieves **4.11% EER**, satisfying the $\le 5\%$ target for pen-tablet hardware.
- **Fusion is not unconditionally better.** Late fusion helps when the two branches are of comparable strength, and *hurts* when one pulls ahead. Measured on writers neither branch trained on: fusion improves the single temporal model (3.96% → 3.62% in §15; 4.41% → 3.50% on the second paired corpus, SigComp2009 / NFI, §25) and degrades the six-seed ensemble (2.61% → 4.63%, losing 22 of 25 draws). For the pen-tablet tier the ensemble is served **unimodally**; fusion is used where the spatial branch is competitive. See `SUMMARY.md` Part 9 §15.
- **Corpus matters when quoting temporal EER.** The ensemble reads **2.61%** on the 10-writer SigComp pen-tablet cohort and **6.94%** on DeepSignDB Evaluation's 372 writers. Both are honest; they measure different populations. Quote the corpus with the number.

---

### Temporal Branch Extended Pre-Training with DeepSignDB

Incorporating DeepSignDB Development's 498 stylus signers (expanding the training pool from 62 to 311 signers) was evaluated on the 3-channel baseline:

| Temporal Model                | Train Signers |      Pooled EER       |     SigComp Dutch     |    SigComp Chinese    |        SVC2004         |      BiosecurID       |         MCYT          | Cohen's d |
| ----------------------------- | :-----------: | :-------------------: | :-------------------: | :-------------------: | :--------------------: | :-------------------: | :-------------------: | :-------: |
| `xyp` baseline (Open corpora) |      62       |   9.41% $\pm$ 0.64%   |   3.95% $\pm$ 0.93%   |   7.81% $\pm$ 1.34%   |   14.21% $\pm$ 1.53%   |        10.58%         |        13.25%         |   1.633   |
| **+ DeepSignDB Development**  |    **311**    | **7.57% $\pm$ 0.29%** | **3.54% $\pm$ 0.43%** | **8.01% $\pm$ 1.50%** | **10.93% $\pm$ 1.23%** | **5.35% $\pm$ 0.46%** | **6.34% $\pm$ 0.36%** | **1.737** |

_Ablation Insight:_ When repeated on top of the 4-channel `dt` model with pinned test writers, scaling training writers 62 $\to$ 394 produced $-0.13$ points difference ($t = 0.28$), demonstrating that temporal verification at this resolution is governed by timing dynamics rather than data volume scarcity.

---

### Datasets and Device Targets

Device support is architected in tiers based on hardware capability:

| Tier          | Hardware Target          | Primary Dataset Proxy | Modality & Script       |  Target EER  |     Measured EER      |
| ------------- | ------------------------ | --------------------- | ----------------------- | :----------: | :-------------------: |
| **Primary**   | XPPen Deco 640 Tablet    | SigComp2011 Dutch     | Online Pen Trajectory   | $\le 5.0\%$  | **4.11% $\pm$ 1.03%** |
| **Secondary** | Document Scanners / Cams | CEDAR / BHSig260      | Offline Scanned Image   | $\le 10.0\%$ | **8.72% $\pm$ 0.42%** |
| **Tertiary**  | Mobile / Touchscreen     | MOBISIG               | Capacitive Finger Touch | $\le 20.0\%$ |        21.26%         |

---

### Evaluation Figures & Benchmark Metrics

Shipped weights evaluated on open corpora using `scripts/benchmark.py --enroll 5 --enroll-trials 10 --seed 42`:

| Branch       | Dataset             | Writers |   Mean $\pm$ SD EER   | Pooled EER | Cohen's d |
| ------------ | ------------------- | :-----: | :-------------------: | :--------: | :-------: |
| **Spatial**  | **All Combined**    | **65**  | **8.72% $\pm$ 0.42%** | **8.55%**  | **1.970** |
| Spatial      | CEDAR               |    8    |   0.00% $\pm$ 0.00%   |   0.00%    |   6.674   |
| Spatial      | BHSig260 Hindi      |   23    |   5.33% $\pm$ 0.50%   |   5.43%    |   2.313   |
| Spatial      | BHSig260 Bengali    |   16    |   8.17% $\pm$ 1.07%   |   8.65%    |   2.139   |
| Spatial      | UTSig               |   18    |  28.10% $\pm$ 3.21%   |   28.75%   |   0.947   |
| **Temporal** | **All Combined**    | **31**  | **7.77% $\pm$ 0.67%** | **7.67%**  | **1.710** |
| Temporal     | SigComp2011 Dutch   |   15    |   4.11% $\pm$ 1.03%   |   4.18%    |   2.450   |
| Temporal     | SigComp2011 Chinese |    6    |   5.28% $\pm$ 0.85%   |   5.23%    |   1.782   |
| Temporal     | SVC2004 Task 2      |   10    |  13.58% $\pm$ 2.21%   |   14.00%   |   1.504   |
| **Fused**    | SigComp2009 / NFI, paired, skilled forgeries | **14** | **3.50% $\pm$ 0.70%** | n/a ‡ | n/a ‡ |
| Fused        | SigComp2009 / NFI, paired, skilled + random  |   14    |   2.03% $\pm$ 0.55%   |   n/a ‡    |   n/a ‡   |

‡ Fused rows: shipped weights on both modalities of the same signing events, the 14 NFI writers clean of every training corpus, 10 enrolment draws; pooled EER and Cohen's d were not computed for this run. Protocol and the SigComp2011 paired result in `SUMMARY.md` Part 9 §25.

---

### Operating Point & Confusion Matrices

Confusion matrices evaluated at fixed production thresholds (`SPATIAL_DECISION_THRESHOLD = 0.7731`, `TEMPORAL_DECISION_THRESHOLD = 0.7900`):

> The notebook is committed **without stored cell outputs**. They were 88% of a
> 843 KB file and made every re-run an unreadable diff. The figures it produces
> are the ones shown below, tracked under `results/`; regenerate them with
> `python tools/run_notebook_cells.py`.

|                           Spatial Branch                           |                           Temporal Branch                            |
| :----------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![Confusion matrix, spatial branch](results/confusion_spatial.png) | ![Confusion matrix, temporal branch](results/confusion_temporal.png) |

| Modality     | True Reject | False Accept | True Accept | False Reject |    FAR     |    FRR     |
| ------------ | ----------: | -----------: | ----------: | -----------: | :--------: | :--------: |
| **Spatial**  |       1,345 |          125 |       1,177 |          112 | **8.50%**  | **8.69%**  |
| **Temporal** |         564 |           63 |         540 |           32 | **10.05%** | **5.59%**  |

_Error Complementarity:_ Spatial and temporal modalities exhibit complementary error distributions ($\phi < 0$). Spatial models tend toward false rejections (rejecting genuine variations), while temporal models tend toward false acceptances. Multimodal late fusion effectively balances both operating regimes.

---

### ROC and DET Curves

|   Modality   |                 ROC Curve                 |                 DET Curve                 |
| :----------: | :---------------------------------------: | :---------------------------------------: |
| **Spatial**  |  ![ROC spatial](results/roc_spatial.png)  |  ![DET spatial](results/det_spatial.png)  |
| **Temporal** | ![ROC temporal](results/roc_temporal.png) | ![DET temporal](results/det_temporal.png) |

---

## Standards & Compliance

| Standard                 | Application in System Architecture                                       |     Status      |
| ------------------------ | ------------------------------------------------------------------------ | :-------------: |
| **ISO/IEC 23053:2022**   | Writer-independent data splits, reproducible training configuration      |     **Met**     |
| **IEEE 730-2014**        | Configuration management, deviation tracking, verification audit logging |   **Partial**   |
| **ISO/IEC 24745:2022**   | Biometric template protection (only derived embeddings persisted)        | **Architected** |
| **ISO/IEC 9075-1:2023**  | Structured relational SQLite database schema                             | **Architected** |
| **ISO/IEC 18033-1:2021** | Application-layer AES-256-GCM encryption of stored templates             | **Architected** |

Configuration control and the deviation log are in place; IEEE 730-2014's verification audit log depends on `src/database/`, which is a stub, so that standard is Partial rather than Met. The three "Architected" rows are design targets for the application layer (`src/database/`, currently a stub) — not properties the current codebase has, since nothing is persisted yet. Only ISO/IEC 23053:2022 is demonstrable from the code as it stands.

---

## Hardware

Digitizer hardware requirements for deployment:

| Specification       | Requirement           | System Handling                                              |
| ------------------- | --------------------- | ------------------------------------------------------------ |
| **Report Rate**     | $\ge 200\text{ Hz}$   | Decimated to 100 Hz model rate on ingest                     |
| **Resolution**      | $\ge 2540\text{ LPI}$ | Coordinate normalization via per-signature z-score           |
| **Pressure Levels** | $\ge 1024$ levels     | Scaled to $[0.0, 1.0]$; flat 1.0 imputation in degraded mode |
| **Interface**       | USB HID               | Windows (Wintab), Linux (`evdev` / OpenTabletDriver)         |

---

## References

### How the Reference Architecture Differs

The spatial Siamese CNN was benchmarked directly against the reference architecture of Dey et al. (SigNet) [1]:

| Hyperparameter / Component | Dey et al. SigNet [1]                            | This Project (Shipped Spatial)                            | Rationale & Empirical Impact                            |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------- |
| **Input Geometry**         | $155 \times 220$, divided by std                 | $155 \times 220$, divided by 255                          | Retains natural pixel contrast with inverted polarity   |
| **Convolutional Stack**    | $96 \times 11 \times 11 \to 256 \to 384 \to 256$ | $96 \times 11 \times 11 \to 256 \to 384 \to 256$ (`wide`) | Wide trunk improved pooled EER from 12.31% to 11.78%    |
| **Normalization**          | Local Response Normalization (LRN)               | GroupNorm (`num_groups=32`)                               | GroupNorm avoids batch-size dependency and pair leakage |
| **Embedding Head**         | FC-1024 (Dropout 0.5) $\to$ FC-128               | $4 \times 4$ Avg+Max Pool (8192) $\to$ FC-512             | FC-1024 head caused metric collapse on 301 writers      |
| **Dropout Schedule**       | 0.3 after pool 2 & 4; 0.5 after FC               | None (`CONV_DROPOUT = 0.0`)                               | Dropout cost +6.78 EER points due to data scarcity      |
| **Loss Function**          | Contrastive Loss (Margin 1.0)                    | Contrastive Loss + Hard-Negative Mining                   | Mines challenging impostor pairs dynamically            |
| **Optimization**           | RMSprop ($10^{-4}$), batch 128                   | Adam ($10^{-3}$, plateau LR schedule), batch 32           | Adam + plateau LR tightened draw variance by 40%        |

---

### Bibliography

#### Methods

1. S. Dey, A. Dutta, J. I. Toledo, S. K. Ghosh, J. Lladós, U. Pal. _SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification._ arXiv:1707.02131, 2017. [arXiv:1707.02131](https://arxiv.org/abs/1707.02131)
2. L. G. Hafemann, R. Sabourin, L. S. Oliveira. _Learning features for offline handwritten signature verification using deep convolutional neural networks._ Pattern Recognition 70:163–176, 2017. doi:10.1016/j.patcog.2017.05.012.
3. O. Mersa, F. Etaati, S. Masoudnia, B. N. Araabi. _Learning Representations from Persian Handwriting for Offline Signature Verification, a Deep Transfer Learning Approach._ arXiv:1903.06249, 2019. [arXiv:1903.06249](https://arxiv.org/abs/1903.06249)
4. A. Soleimani, B. N. Araabi, K. Fouladi. _Deep Multitask Metric Learning for Offline Signature Verification._ Pattern Recognition Letters 80:84–90, 2016.
5. A. Soleimani, K. Fouladi, B. N. Araabi. _UTSig: A Persian offline signature dataset._ IET Biometrics 6(1):1–8, 2017.
6. D.-Y. Yeung, H. Chang, Y. Xiong, S. George, R. Kashi, T. Matsumoto, G. Rigoll. _SVC2004: First International Signature Verification Competition._ ICBA 2004, LNCS 3072, pp. 16–22, 2004.
7. C. Sekhar, P. Mukherjee, D. S. Guru, V. Pulabaigari. _OSVNet: Convolutional Siamese Network for Writer Independent Online Signature Verification._ ICDAR 2019. [arXiv:1904.00240](https://arxiv.org/abs/1904.00240)
8. M. Liwicki, M. I. Malik, C. E. van den Heuvel, X. Chen, C. Berger, R. Stoel, M. Blumenstein, B. Found. _Signature Verification Competition for Online and Offline Skilled Forgeries (SigComp2011)._ ICDAR 2011, pp. 1480–1484, 2011.
9. M. Antal, L. Z. Szabó, T. Tordai. _Online Signature Verification on MOBISIG Finger-Drawn Signature Corpus._ Mobile Information Systems, vol. 2018, Article ID 3127042, 2018.

#### Datasets & Required Licensure Citations

⚖ marks the four references DeepSignDB's licence clause 4.5 requires citing in any published work that uses that dataset (see [DATASET_LICENSES.md](DATASET_LICENSES.md)).

10. M. K. Kalera, S. Srihari, A. Xu. _Offline signature verification and identification using distance statistics._ IJPRAI 18(7):1339–1360, 2004. (CEDAR)
11. S. Pal, A. Alaei, U. Pal, M. Blumenstein. _Performance of an off-line signature verification method based on texture features on a large indic-script signature dataset._ DAS 2016, pp. 72–77, 2016. (BHSig260)
12. ⚖ R. Tolosana, R. Vera-Rodriguez, J. Fierrez, J. Ortega-Garcia. _DeepSign: Deep On-Line Signature Verification._ IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(2):229–239, 2021. (DeepSignDB)
13. ⚖ J. Fierrez, J. Galbally, et al. _BiosecurID: A Multimodal Biometric Database._ Pattern Analysis and Applications, 13(2):235–246, 2010.
14. ⚖ J. Ortega-Garcia, J. Fierrez, et al. _The Multi-Scenario Multi-Environment BioSecure Multimodal Database (BMDB)._ IEEE TPAMI, 32(6):1097–1111, 2010.
15. ⚖ J. Ortega-Garcia, J. Fierrez, et al. _MCYT Baseline Corpus: A Bimodal Biometric Database._ IEE Proc. Vision, Image and Signal Processing, 150(6):395–401, 2003.

#### Standards

16. **ISO/IEC 23053:2022** Framework for Artificial Intelligence Systems Using Machine Learning.
17. **ISO/IEC 24745:2022** Information Technology Security Techniques Biometric Information Protection.
18. **ISO/IEC 18033-1:2021** Information Security Encryption Algorithms.
19. **IEEE 730-2014** IEEE Standard for Software Quality Assurance Processes.

---

## Team

**Team 06** — Senior Design Project, Department of Electrical and Computer Engineering, King Abdulaziz University.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- King Abdulaziz University, Faculty of Engineering, Department of Electrical and Computer Engineering.
- Authors of the CEDAR, BHSig260, UTSig, SVC2004, SigComp2011, and DeepSignDB biometric datasets.
