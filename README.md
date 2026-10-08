# Signature Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)]()

> **Senior Design Project (EE 499)** — Department of Electrical and Computer Engineering, King Abdulaziz University (KAU), Jeddah, Saudi Arabia — Team 06

A multimodal signature verification system built with dual Siamese deep neural networks. It verifies signatures using both **offline** (scanned images) and **online** (dynamic tablet pen strokes) inputs, combining their scores via late fusion to detect genuine signatures and skilled forgeries.

This document covers installation, quickstart, project structure, evaluation benchmarks, and architectural comparisons.

---

## Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [GUI](#gui)
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
- [Database & Capture Layer](#database--capture-layer)
  - [Database](#database)
  - [Signature Capture](#signature-capture)  
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
- **Ultra-Low Latency Inference** Fast CPU execution (0.5 s from pressing Enter to the fused result with 5 scan and 5 pen references, 0.08 s per check against stored references, after a 1.35 s app start that loads the models) and light memory footprint (~0.39 GB RAM), well within embedded budgets (< 3.0 s, ≤ 4 GB).
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

### GUI

A standalone PyQt5 desktop application (`src/gui/`) provides the
employee-facing interface for enrollment and verification. It is
implemented and tested on the `mo-gui` branch, pending merge into `main`.

**Employee workflow:**

1. **Register / Login** — Each employee has their own account (full name,
   User ID, password). Enrolled clients are scoped per employee: an
   employee only sees the clients they personally registered.
2. **Enrollment** — Register a new client by ID and attach exactly 5
   reference signature images. References can be previewed, replaced, or
   removed before saving, and re-opened for editing at any time afterward.
3. **Verification** — Enter a claimed client ID, select modality (image
   input currently; dynamic tablet input is scaffolded but disabled until
   `src/capture/` is implemented), submit a probe signature, and receive
   the result rendered exactly as `[Genuine/Forged, Confidence %]`.
4. **Dashboard** — Landing screen after login showing per-employee stats:
   clients enrolled this month, total enrolled clients, total
   verifications run, and a live genuine-rate indicator.
5. **Clients** — A table of the employee's enrolled clients (enrollment
   date, reference count, most recent verification), used to reopen an
   existing client for editing.
6. **Profile** — Employee account settings (name, User ID, password).

**Current integration status:** the GUI calls a single interface module
(`src/gui/interfaces.py`) for every ML and storage operation
(`get_spatial_embedding`, `get_references`, `store_embedding`, etc.). Those
functions are currently mocked; wiring them to the real `src/models/` and
`src/database/` implementations requires no change to the GUI layer itself
— only `interfaces.py`.

**Screenshots:**

|                             Login                                  |                           Dashboard                                  |
| :----------------------------------------------------------------: | :------------------------------------------------------------------: |
|         ![Login screen](assets/screenshots/login.png)                | ![Dashboard with per-employee stats](assets/screenshots/dashboard.png) |


|                           Enrollment                               |                            Verification                              |
| :----------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![Enrollment panel](assets/screenshots/enrollment.png)               | ![Verification result display](assets/screenshots/verification.png)    |


|                       Clients                                      |                          Profile                                     |
| :----------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![Enrolled clients table](assets/screenshots/clients.png)            | ![Employee profile editing](assets/screenshots/profile.png)            |

**Running the GUI:**

```bash
cd src/gui
python app.py
```

#### The `dt` Channel

`dt` represents the time interval between consecutive sampled points, providing a 4-channel input tensor `[X, Y, Pressure, dt]`.

Without timing deltas, uniform-rate models lose visibility into pen pauses and air-travel hesitations. Genuine signers execute signatures with continuous motor memory rhythm, whereas forgers hesitate and trace strokes slowly. `dt` remains ~10 ms during continuous pen-down contact and jumps significantly at pen lifts. Across six paired random seeds, adding `dt` improved pooled temporal EER from **10.45% to 8.97%** ($t = 3.25$).

---
---

### Enrollment and Decision

Verification evaluates a probe against an enrolled gallery using nearest-reference matching:

| Parameter                 | Value                                                                | Configuration Constant                                      |
| ------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| References per user ($K$) | **5**                                                                | `NUM_ENROLLMENT_REFERENCES = 5`                             |
| Matching Rule             | Nearest Reference: $d_{min} = \min_{i} \|e_{probe} - e_{ref, i}\|_2$ | `_nearest_reference_score`                                  |
| Similarity Score          | $S = \frac{1}{1 + d_{min}}$                                          | —                                                           |
| Decision Threshold        | Spatial $\tau = 0.7637$, Temporal $\tau = 0.7900$                    | `SPATIAL_DECISION_THRESHOLD`, `TEMPORAL_DECISION_THRESHOLD` |

More enrolled references give better accuracy without retraining, but the gain gets small after $K=5$. We swept $K$ from 1 to 15 with `benchmark.py --enroll K`, 10 draws each. Spatial pooled EER goes from 11.71% ($K=1$) to 8.72% ($K=5$), 7.86% ($K=10$) and 7.82% ($K=15$). These spatial numbers come from the previous spatial weight, not the one shipped now. Temporal EER goes from 13.72% ($K=1$) to 7.81% ($K=5$), 6.99% ($K=10$) and 6.31% ($K=15$). Both curves are almost flat after $K=10$. The app uses $K=5$.

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

Each published result sits beside this project measured under the same protocol, on writers the shipped weights never trained on, on CPU. One-reference rows score isolated pairs, and DeepSignDB rows use the dataset's official comparison lists; each is one EER over every trial, as those protocols define it. $K=5$ rows are mean $\pm$ standard deviation over 10 random enrollment draws (seed 42).

| Dataset                 | Protocol                                                   | Published EER | Source | This project                  |
| ----------------------- | ---------------------------------------------------------- | :-----------: | :----: | :---------------------------: |
| **CEDAR**               | Writer-independent, 1 reference                            |     0.00%     |  [1]   |             0.00%             |
| **CEDAR**               | Writer-independent, 1 reference                            |     3.51%     |  [10]  |             0.00%             |
| **CEDAR**               | Writer-dependent SVM on CNN features, $K=12$               |     4.63%     |  [2]   |  0.00% $\pm$ 0.00% ($K=5$) §  |
| **BHSig260 Bengali**    | Writer-independent, 1 reference                            |    13.89%     |  [1]   |            11.10%             |
| **BHSig260 Bengali**    | Writer-independent, 1 reference                            |     6.12%     |  [10]  |            11.10%             |
| **BHSig260 Hindi**      | Writer-independent, 1 reference                            |    15.36%     |  [1]   |             5.94%             |
| **BHSig260 Hindi**      | Writer-independent, 1 reference                            |     6.68%     |  [10]  |             5.94%             |
| **UTSig (Persian)**     | Writer-dependent SVM, Persian transfer learning, $K=10$    |     9.80%     |  [3]   | 26.34% $\pm$ 2.17% ($K=5$) §  |
| **UTSig (Persian)**     | Writer-dependent SVM, Persian transfer learning, $K=5$     |    11.16%     |  [3]   |    26.34% $\pm$ 2.17% §      |
| **UTSig (Persian)**     | Writer-dependent DMML on HOG features, $K=12$              |    17.45%     |  [4]   | 26.34% $\pm$ 2.17% ($K=5$) §  |
| **UTSig (Persian)**     | Writer-dependent SVM on geometric features, $K=12$         |    29.71%     |  [5]   | 26.34% $\pm$ 2.17% ($K=5$) §  |
| **SVC2004 Task 2**      | $K=5$, skilled forgeries, writer-specific thresholds       |     1.39%     |  [12]  | 13.58% $\pm$ 2.21% (global threshold) |
| **SVC2004 Task 2**      | $K=5$, skilled forgeries, competition winner               |     2.89%     |  [6]   |      13.58% $\pm$ 2.21%       |
| **SVC2004 Task 2**      | Writer-independent Siamese CNN, 35 train / 5 test writers  |   77.00% acc  |  [7]   |      13.58% $\pm$ 2.21%       |
| **DeepSignDB (stylus)** | Official lists, skilled forgeries, 4 references            |     2.44%     |  [12]  |           13.59% ‖            |
| **DeepSignDB (stylus)** | Official lists, skilled forgeries, 4 references            |     2.54%     |  [11]  |           13.59% ‖            |
| **DeepSignDB (stylus)** | Official lists, skilled forgeries, 4 references            |     3.30%     |  [15]  |           13.59% ‖            |
| **DeepSignDB (stylus)** | Official lists, skilled forgeries, 1 reference             |     4.20%     |  [15]  |           16.33% ‖            |
| **DeepSignDB (stylus)** | Official lists, random forgeries, 4 references             |     0.60%     |  [15]  |            6.51% ‖            |
| **SigComp2011 Dutch**   | Multi-system competition benchmark                         | _Competition_ |  [8]   |  4.11% $\pm$ 1.03% ($K=5$)    |
| **SigComp2011 Chinese** | Multi-system competition benchmark                         | _Competition_ |  [8]   |  5.28% $\pm$ 0.85% ($K=5$)    |
| **MOBISIG (Touch)**     | DTW with user-specific thresholds                          |     5.81%     |  [9]   |           21.26% †            |
| **MOBISIG (Touch)**     | Global features with global threshold                      |    14.31%     |  [9]   |           21.26% †            |

§ _This project is writer-independent: one model for everyone, nothing fitted per user. No writer-dependent result exists here, so the closest protocol, $K=5$, is shown._

‖ _Shipped temporal weights, never trained on DeepSignDB, on the three pen-tablet sub-databases they can load (MCYT, BiosecurID and BiosecureDS2, 372 writers). The published figures cover all five sub-databases. Published 4-reference scores average the four comparisons; this project scores the nearest reference._

$\dagger$ _Note: MOBISIG figures reflect a separate touchscreen evaluation model. The primary shipped temporal weights are trained exclusively on digitizer pen corpora (SVC2004 Task 2 + SigComp2011)._

#### Analysis of Protocol Differences & Accuracy Gaps

- **UTSig Protocol Gap:** Published results achieving 9–17% EER on UTSig rely on **writer-dependent** training (training per-user classifiers with $K=10$ to $12$ references). In our strictly writer-independent, few-shot enrollment setup, Persian signatures exhibit complex stroke loops that yield 26.34% EER.
- **SVC2004 Intrinsic Hardness:** Published SVC2004 benchmarks below 3% utilize _a posteriori_ user-specific thresholds. Under a global threshold, SVC2004 is intrinsically challenging: DTW alignment confirms an intra-genuine distance ratio of 1.44 (vs 1.61 for Dutch) and a median forgery separation $z$-score of 1.37 (vs 2.23 for Dutch).
- **SigComp2011 Target Tier:** On true pen digitizers (SigComp2011 Dutch), our temporal model achieves **4.11% EER**, satisfying the $\le 5\%$ target for pen-tablet hardware.
- **Fusion does not always help.** It helps when the two branches are about equally strong, and it hurts when one branch is much stronger. On writers that neither branch trained on, fusion improves the single temporal model (3.96% → 3.62%). On the second paired corpus (SigComp2009 / NFI) it went from 3.90% to 3.62% with the previous spatial weight, but from 3.90% to 4.09% with the current one, so there is no gain there now. Fusion makes the six-seed temporal ensemble worse (2.61% → 4.63%, worse on 22 of 25 draws). So for the pen-tablet tier the ensemble is used **alone**, and fusion is used only where the spatial branch is competitive.
- **Corpus matters when quoting temporal EER.** The ensemble reads **2.61%** on the 10-writer SigComp pen-tablet cohort and **6.94%** on DeepSignDB Evaluation's 372 writers. Both are honest; they measure different populations. Quote the corpus with the number. On DeepSignDB's official lists, where the references come from one session and the probe from a later one, the shipped single model reads **13.59%** and the ensemble **11.11%** (skilled forgeries, 4 references).

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
| **Primary**   | XPPen Deco 640 Tablet    | SigComp2011 / NFI, paired | Fused (scan + pen), tablet-only fallback | $\le 5.0\%$  | **4.09% $\pm$ 1.11%** fused on NFI, 3.09% $\pm$ 0.37% on SigComp2011; 4.11% $\pm$ 1.03% pen alone |
| **Secondary** | Document Scanners / Cams | CEDAR / BHSig260      | Offline Scanned Image   | $\le 10.0\%$ | **7.46% $\pm$ 0.46%** |
| **Tertiary**  | Mobile / Touchscreen     | MOBISIG               | Capacitive Finger Touch | $\le 20.0\%$ |        21.26%         |

---

### Evaluation Figures & Benchmark Metrics

Shipped weights evaluated on open corpora using `scripts/benchmark.py --enroll 5 --enroll-trials 10 --seed 42`. To reproduce any release's published numbers, run `python tools/reproduce_release.py <tag>` (`--list` shows the releases):

| Branch       | Dataset             | Writers |   Mean $\pm$ SD EER   | Pooled EER | Cohen's d |
| ------------ | ------------------- | :-----: | :-------------------: | :--------: | :-------: |
| **Spatial**  | **All Combined**    | **65**  | **7.46% $\pm$ 0.46%** | **7.72%**  | **2.195** |
| Spatial      | CEDAR               |    8    |   0.00% $\pm$ 0.00%   |   0.00%    |   8.941   |
| Spatial      | BHSig260 Hindi      |   23    |   3.47% $\pm$ 0.90%   |   3.34%    |   2.575   |
| Spatial      | BHSig260 Bengali    |   16    |   6.82% $\pm$ 1.10%   |   6.89%    |   2.185   |
| Spatial      | UTSig               |   18    |  26.34% $\pm$ 2.17%   |   27.65%   |   1.277   |
| **Temporal** | **All Combined**    | **31**  | **7.77% $\pm$ 0.67%** | **7.67%**  | **1.710** |
| Temporal     | SigComp2011 Dutch   |   15    |   4.11% $\pm$ 1.03%   |   4.18%    |   2.450   |
| Temporal     | SigComp2011 Chinese |    6    |   5.28% $\pm$ 0.85%   |   5.23%    |   1.782   |
| Temporal     | SVC2004 Task 2      |   10    |  13.58% $\pm$ 2.21%   |   14.00%   |   1.504   |
| **Fused**    | SigComp2009 / NFI, paired, skilled forgeries | **14** | **4.09% $\pm$ 1.11%** | n/a ‡ | n/a ‡ |
| Fused        | SigComp2009 / NFI, paired, skilled + random  |   14    |   2.03% $\pm$ 0.55%   |   n/a ‡    |   n/a ‡   |

‡ Fused rows: shipped weights on both modalities of the same signing events, 14 NFI writers that are not in any training corpus, 10 enrollment draws. Pooled EER and Cohen's d were not computed for this run.

---

### Operating Point & Confusion Matrices

Confusion matrices evaluated at fixed production thresholds (`SPATIAL_DECISION_THRESHOLD = 0.7637`, `TEMPORAL_DECISION_THRESHOLD = 0.7900`):

> The notebook is committed **without stored cell outputs**. They were 88% of a
> 843 KB file and made every re-run an unreadable diff. The figures it produces
> are the ones shown below, tracked under `results/`; regenerate them with
> `python tools/run_notebook_cells.py`.

|                           Spatial Branch                           |                           Temporal Branch                            |
| :----------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![Confusion matrix, spatial branch](results/confusion_spatial.png) | ![Confusion matrix, temporal branch](results/confusion_temporal.png) |

| Modality     | True Reject | False Accept | True Accept | False Reject |    FAR     |    FRR     |
| ------------ | ----------: | -----------: | ----------: | -----------: | :--------: | :--------: |
| **Spatial**  |       1,360 |          110 |       1,181 |          108 | **7.48%**  | **8.38%**  |
| **Temporal** |         556 |           71 |         548 |           24 | **11.32%** | **4.20%**  |

_Error Complementarity:_ Spatial and temporal modalities exhibit complementary error distributions ($\phi < 0$). Spatial models tend toward false rejections (rejecting genuine variations), while temporal models tend toward false acceptances. Multimodal late fusion effectively balances both operating regimes.

---

### ROC and DET Curves

|   Modality   |                 ROC Curve                 |                 DET Curve                 |
| :----------: | :---------------------------------------: | :---------------------------------------: |
| **Spatial**  |  ![ROC spatial](results/roc_spatial.png)  |  ![DET spatial](results/det_spatial.png)  |
| **Temporal** | ![ROC temporal](results/roc_temporal.png) | ![DET temporal](results/det_temporal.png) |

---
## Database & Capture Layer 

### Database

Encrypted embedding storage tested end-to-end — AES-256-GCM encryption,
decryption, and audit logging all verified successfully via
`src/database/db_manager.py`.

![Database test result](assets/screenshots/Database_test_success.png)
![Database encryption](assets/screenshots/Database_encryption.png)


Test confirms: encrypted spatial and temporal embeddings match their
decrypted values, and verification events are correctly recorded in
the audit log.

### Signature Capture

Full pen-down to pen-up signature capture implemented via
`src/capture/tablet.py`, using PyQt5 Tablet Events on the XPPen Deco 640.

![Screen before signature](assets/screenshots/Screen_before_sign.png)
![Screen After signature](assets/screenshots/Screen_after_sign.png)
![Tablet capture result](assets/screenshots/Tablet_capture_result.png)


The captured trajectory passes both quality gates (pre- and
post-normalization) and produces the final `[L, 4]` array —
X, Y, Pressure, dt — matching the capture contract defined in
`AGENTS.md`.

**Status:** Windows path complete and tested. Linux (Arch) path is
pending.

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
10. F.-H. Huang, H.-M. Lu. _Multiscale Feature Learning Using Co-Tuplet Loss for Offline Handwritten Signature Verification._ 2023. [arXiv:2308.00428](https://arxiv.org/abs/2308.00428)
11. J. Jiang, S. Lai, L. Jin, Y. Zhu. _DsDTW: Local Representation Learning with Deep Soft-DTW for Dynamic Signature Verification._ IEEE Transactions on Information Forensics and Security 17:2198–2212, 2022.
12. R. Li, D. Yang. _Online Signature Verification Using Augmented Path Signature and T-Mamba._ 2026. [arXiv:2609.08276](https://arxiv.org/abs/2609.08276)

#### Datasets & Required Licensure Citations

⚖ marks the four references DeepSignDB's licence clause 4.5 requires citing in any published work that uses that dataset (see [DATASET_LICENSES.md](DATASET_LICENSES.md)).

13. M. K. Kalera, S. Srihari, A. Xu. _Offline signature verification and identification using distance statistics._ IJPRAI 18(7):1339–1360, 2004. (CEDAR)
14. S. Pal, A. Alaei, U. Pal, M. Blumenstein. _Performance of an off-line signature verification method based on texture features on a large indic-script signature dataset._ DAS 2016, pp. 72–77, 2016. (BHSig260)
15. ⚖ R. Tolosana, R. Vera-Rodriguez, J. Fierrez, J. Ortega-Garcia. _DeepSign: Deep On-Line Signature Verification._ IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(2):229–239, 2021. (DeepSignDB)
16. ⚖ J. Fierrez, J. Galbally, et al. _BiosecurID: A Multimodal Biometric Database._ Pattern Analysis and Applications, 13(2):235–246, 2010.
17. ⚖ J. Ortega-Garcia, J. Fierrez, et al. _The Multi-Scenario Multi-Environment BioSecure Multimodal Database (BMDB)._ IEEE TPAMI, 32(6):1097–1111, 2010.
18. ⚖ J. Ortega-Garcia, J. Fierrez, et al. _MCYT Baseline Corpus: A Bimodal Biometric Database._ IEE Proc. Vision, Image and Signal Processing, 150(6):395–401, 2003.

#### Standards

19. **ISO/IEC 23053:2022** Framework for Artificial Intelligence Systems Using Machine Learning.
20. **ISO/IEC 24745:2022** Information Technology Security Techniques Biometric Information Protection.
21. **ISO/IEC 18033-1:2021** Information Security Encryption Algorithms.
22. **IEEE 730-2014** IEEE Standard for Software Quality Assurance Processes.

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
