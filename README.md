<div align="center">

# [GRU-DDoS-Detection-CICDDoS2019](https://github.com/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019) <img src="https://cdn.simpleicons.org/github" width="3%" height="3%">

</div>

<div align="center">

---

**Methodology source:** Mahrukh Ramzan, Muhammad Shoaib, Ayesha Altaf, Shazia Arshad, Faiza Iqbal, Ángel Kuc Castilla, and Imran Ashraf, *Distributed Denial of Service Attack Detection in Network Traffic Using Deep Learning Algorithm* (2023), DOI: [10.3390/s23208642](https://doi.org/10.3390/s23208642).

A reproducible, memory-aware reconstruction of the paper's CICDDoS2019 multiclass GRU experiment, organized for bounded Apple-Silicon runs and full-source Linux GPU execution. This repository is an independent reproduction and is not an official repository of the paper's authors.

---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Contributors](https://img.shields.io/github/contributors/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![GitHub Created At](https://img.shields.io/github/created-at/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/1aa1df4d0d9e67c798ca62e117122c24191e4922.svg "Repobeats analytics image")

</div>

## Table of Contents

- [GRU-DDoS-Detection-CICDDoS2019 ](#gru-ddos-detection-cicddos2019-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Original Paper](#original-paper)
  - [Target Experiment](#target-experiment)
  - [Paper Methodology](#paper-methodology)
  - [Reconstruction Methodology](#reconstruction-methodology)
  - [Pipeline](#pipeline)
  - [Published Top-20 Features](#published-top-20-features)
  - [Target Classes](#target-classes)
  - [GRU Architecture and Training](#gru-architecture-and-training)
  - [Sampling Profiles](#sampling-profiles)
    - [`cap-per-class`](#cap-per-class)
    - [`figure6-inferred`](#figure6-inferred)
    - [`all`](#all)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Setup](#setup)
    - [Clone the repository](#clone-the-repository)
  - [Makefile Automation](#makefile-automation)
  - [Installation and Execution](#installation-and-execution)
    - [macOS Apple Silicon — bounded-memory execution](#macos-apple-silicon--bounded-memory-execution)
    - [Linux SSH server — full-source execution](#linux-ssh-server--full-source-execution)
  - [Runtime Logging and Completion Sound](#runtime-logging-and-completion-sound)
  - [Configuration](#configuration)
  - [Generated Outputs](#generated-outputs)
  - [Reproducibility Notes and Limitations](#reproducibility-notes-and-limitations)
  - [Results Target](#results-target)
  - [References](#references)
  - [How to Cite](#how-to-cite)
  - [Contributing](#contributing)
  - [Author](#author)
  - [License](#license)
    - [MIT License](#mit-license)

## Introduction

This repository reconstructs the **CICDDoS2019 multiclass GRU experiment** from Ramzan et al., *Distributed Denial of Service Attack Detection in Network Traffic Using Deep Learning Algorithm* (Sensors, 2023).

The paper evaluates RNN, LSTM, and GRU models for DDoS detection. For CICDDoS2019 multiclass classification, Table 4 reports the GRU as the strongest model in accuracy, with **99.54% accuracy**, **98% precision**, **99% recall**, and **98% F1-score**. The project therefore targets the GRU multiclass configuration rather than the binary experiment or the other recurrent architectures.

The repository separates **paper-specified settings** from **reconstruction choices**. This distinction is important because the publication gives considerably more detail than many related works, but still omits the exact source-row sampling procedure, recurrent sequence construction, dropout rate, validation partition construction, random seeds, scaler fitting scope, and repeated-run protocol.

## Original Paper

> M. Ramzan, M. Shoaib, A. Altaf, S. Arshad, F. Iqbal, Á. Kuc Castilla, and I. Ashraf, “Distributed Denial of Service Attack Detection in Network Traffic Using Deep Learning Algorithm,” *Sensors*, vol. 23, no. 20, article 8642, 2023. DOI: [10.3390/s23208642](https://doi.org/10.3390/s23208642).

The paper uses CICDDoS2019 for training/evaluation and also compares results with CICIDS2017. This repository focuses specifically on reproducing the **CICDDoS2019 GRU multiclass result**.

## Target Experiment

Table 4 of the paper reports:

| Model | Accuracy | Precision | Recall | F1-score | Execution time |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNN | 99.15% | 97% | 97% | 97% | 4 min |
| LSTM | 99.43% | 98% | 99% | 98% | 16 min 30 s |
| **GRU** | **99.54%** | **98%** | **99%** | **98%** | **7 min 3 s** |

The configured numerical targets are therefore:

```text
Accuracy = 0.9954
F1       = 0.9800
```

The paper reports F1 as the rounded value `98%`; `0.9800` is the decimal representation used by the reproduction code, not an unrounded value published by the authors.

## Paper Methodology

The paper describes the following high-level methodology for the CICDDoS2019 recurrent experiments:

1. Use CICDDoS2019 network-flow records.
2. Remove records with missing/null values.
3. Convert categorical values to numerical representations.
4. Normalize inputs to zero mean and unit standard deviation.
5. Use Extra Trees feature selection and retain the published top 20 features.
6. Split the data into 70% training and 30% testing.
7. Train recurrent deep-learning models.
8. For the GRU configuration, use two GRU layers with 8 units per recurrent layer.
9. Use two hidden layers with 16 and 8 neurons.
10. Use ReLU in hidden layers and Softmax for multiclass output.
11. Train with Adam, learning rate `0.001`, categorical cross-entropy, batch size `1000`, and up to `100` epochs.
12. Stop early based on validation loss using `min_delta=0.001` and `patience=5`.
13. Evaluate accuracy, precision, recall, F1, and confusion matrices.

The paper does not explicitly define all implementation details required to replay those steps bit-for-bit; the next section documents how this repository resolves them.

## Reconstruction Methodology

| Item | Publication status | Project implementation |
| --- | --- | --- |
| Dataset | Given | CICDDoS2019. |
| Source day/files | Not fully specified for the exact reported run | Defaults to `01-12`, which contains the Figure 6(c) target classes used by the reproduction. |
| Missing/null handling | Given | Invalid rows are removed during streamed cleaning. |
| Infinity handling | Not specified | Non-finite numeric values are rejected as invalid. |
| Categorical inputs | General numerical encoding described | `LabelEncoder` for `Timestamp` and `Flow ID`. |
| Output labels | One-hot encoding described | Label IDs are encoded and converted to one-hot arrays before model fitting. |
| Feature selection | Extra Trees + final top 20 given | The final published top-20 list is used directly; the project does not refit Extra Trees for the final model. |
| Split | 70/30 given | `train_test_split(..., test_size=0.30)`. |
| Split seed | Missing | Default `42`. |
| Stratification | Missing | High-fidelity command uses `--no-stratify`; stratification remains available. |
| StandardScaler | Given | `separate` mode fits train and test scalers independently for the high-fidelity reconstruction; `train-only` is available for rigorous evaluation. |
| GRU layers | Given | 2. |
| GRU units | Given | 8 + 8. |
| Dense hidden layers | Given | 16 + 8. |
| Input sequence construction | Missing | Reshape `(N, 20)` to `(N, 1, 20)`. |
| Dropout | Usage reported, rate/placement missing | `0.10` after each GRU by default. |
| Optimizer | Given | Adam. |
| Learning rate | Given | `0.001`. |
| Loss | Given | Categorical cross-entropy. |
| Batch size | Given | `1000` in the paper-faithful Linux command. |
| Maximum epochs | Given | `100`. |
| Early stopping | Given | `val_loss`, `min_delta=0.001`, `patience=5`. |
| Validation split | Missing despite validation curves | `validation-mode=test` reproduces the likely test-as-validation behavior; `holdout` provides a rigorous alternative. |
| Random model seed | Missing | Default `1337`. |
| Run count / CV | Not reported for the GRU result | Default one independent run; no cross-validation is invented. |
| F1 averaging | Missing | Macro, weighted, and micro F1 are all reported by the project. |

`scaling-mode=separate` and `validation-mode=test` deliberately remain available because the purpose of the high-fidelity path is to investigate the historical result. They are methodologically leaky and should not be interpreted as recommended practice for a new experiment.

## Pipeline

```mermaid
flowchart TD
    A[CICDDoS2019<br/>source day 01-12] --> B[Discover source CSV files]
    B --> C[PASS 1<br/>stream, clean, count valid class rows]
    C --> D{Sampling profile}
    D -->|cap-per-class| E[Bounded per-class sample]
    D -->|figure6-inferred| F[Figure 6(c)-inferred quotas]
    D -->|all| G[Every cleaned target-class row]
    E --> H[PASS 2<br/>exact hypergeometric sampling]
    F --> H
    G --> H
    H --> I[sampled_selected_top20.csv.gz]
    I --> J[LabelEncoder fit<br/>categorical features + output]
    J --> K[encoded_sample_X.npy<br/>encoded_sample_y.npy]
    K --> L[70% train / 30% test]
    L --> M[StandardScaler]
    M --> N[Reshape to N × 1 × 20]
    N --> O[GRU 8<br/>return sequences]
    O --> P[Dropout 0.10]
    P --> Q[GRU 8]
    Q --> R[Dropout 0.10]
    R --> S[Dense 16 + ReLU]
    S --> T[Dense 8 + ReLU]
    T --> U[Dense 12 + Softmax]
    U --> V[Adam 0.001<br/>categorical cross-entropy]
    V --> W[Early stopping on val_loss]
    W --> X[Final test prediction + metrics]
```

Compact representation:

```text
CICDDoS2019
→ clean invalid/null records
→ published top-20 features
→ exact streamed sampling
→ LabelEncoder / output encoding
→ 70/30 split
→ StandardScaler
→ reshape (N, 1, 20)
→ GRU(8)
→ Dropout(0.10)
→ GRU(8)
→ Dropout(0.10)
→ Dense(16, ReLU)
→ Dense(8, ReLU)
→ Dense(12, Softmax)
→ Adam 0.001 + categorical cross-entropy
→ early stopping
→ test metrics and confusion matrix
```

## Published Top-20 Features

The final selected features reproduced by this project are:

| # | Feature |
| ---: | --- |
| 1 | Timestamp |
| 2 | Source Port |
| 3 | Min Packet Length |
| 4 | Fwd Packet Length Min |
| 5 | Flow ID |
| 6 | Packet Length Mean |
| 7 | Fwd Packet Length Max |
| 8 | Average Packet Size |
| 9 | ACK Flag Count |
| 10 | Avg Fwd Segment Size |
| 11 | Fwd Packet Length Mean |
| 12 | Flow Bytes/s |
| 13 | Max Packet Length |
| 14 | Protocol |
| 15 | Fwd Packets/s |
| 16 | Flow Packets/s |
| 17 | Total Length of Fwd Packets |
| 18 | Subflow Fwd Bytes |
| 19 | Destination Port |
| 20 | act_data_pkt_fwd |

`Timestamp` and `Flow ID` are intentionally retained because the goal is to reproduce the published selected-feature experiment, even though identifier-like fields would normally deserve additional leakage analysis in a new study.

## Target Classes

The reproduction uses the 12 labels visible in the paper's GRU Figure 6(c):

1. `BENIGN`
2. `DrDoS_DNS`
3. `DrDoS_LDAP`
4. `DrDoS_MSSQL`
5. `DrDoS_NTP`
6. `DrDoS_NetBIOS`
7. `DrDoS_SNMP`
8. `DrDoS_SSDP`
9. `DrDoS_UDP`
10. `Syn`
11. `TFTP`
12. `UDP-lag`

This class set is important because the prose class description in the publication is not perfectly aligned with the labels visible in the GRU confusion matrix. The reproduction follows the matrix labels for the target experiment.

## GRU Architecture and Training

The model implemented in `gru_ddos_detection/model.py` is:

```text
Input: (1 timestep, 20 features)
        ↓
GRU(8, activation=ReLU, recurrent_activation=sigmoid)
return_sequences=True
        ↓
Dropout(0.10)
        ↓
GRU(8, activation=ReLU, recurrent_activation=sigmoid)
return_sequences=False
        ↓
Dropout(0.10)
        ↓
Dense(16, ReLU)
        ↓
Dense(8, ReLU)
        ↓
Dense(12, Softmax)
```

Training configuration:

| Parameter | Value |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical cross-entropy |
| Paper batch size | 1000 |
| Maximum epochs | 100 |
| Early-stopping monitor | `val_loss` |
| `min_delta` | 0.001 |
| Patience | 5 |

## Sampling Profiles

The project provides three explicit profiles because the exact original source-row sampling procedure is not published.

### `cap-per-class`

Used by `make run-mac` to bound memory use:

```text
MAC_SAMPLING_PROFILE=cap-per-class
MAC_PER_CLASS_CAP=100000
MAC_BATCH_SIZE=256
```

The cap is applied independently to each of the 12 target classes before the 70/30 split. Values can be overridden without editing source code:

```bash
make run-mac MAC_PER_CLASS_CAP=50000 MAC_BATCH_SIZE=128
```

### `figure6-inferred`

Uses class quotas inferred from the published Figure 6(c) test supports divided by the reported `0.30` test proportion. This is the closest matrix-sized reconstruction, not a source sample size explicitly stated by the authors.

### `all`

Used by `make run-linux` for the requested full-source server execution:

```text
LINUX_SAMPLING_PROFILE=all
LINUX_BATCH_SIZE=1000
```

Every cleaned target-class row from the selected `01-12` source day is retained. Batch size controls GRU training batches and does not cap source records. To prioritize the paper-matrix-sized reconstruction instead of all valid rows, run:

```bash
make run-linux LINUX_SAMPLING_PROFILE=figure6-inferred
```

## Project Structure

```text
GRU-DDoS-Detection-CICDDoS2019/
├── .assets/
│   └── Sounds/
│       └── NotificationSound.wav
├── logs/
│   └── .gitkeep
├── .gitignore
├── LICENSE
├── Logger.py
├── Makefile
├── README.md
├── main.bib
├── main.py
├── requirements.txt
└── gru_ddos_detection/
    ├── __init__.py
    ├── audit.py
    ├── cli.py
    ├── config.py
    ├── constants.py
    ├── encoding.py
    ├── evaluation.py
    ├── experiment.py
    ├── model.py
    ├── persistence.py
    ├── preprocessing.py
    ├── sampling.py
    ├── schema.py
    ├── system.py
    ├── tensorflow_runtime.py
    ├── timing.py
    └── workflow.py
```

`main.py` remains the orchestrator. It now configures the repository-root `Logger.py`, registers the bundled completion sound with `atexit`, parses and validates the CLI, and delegates to the package workflow. Dataset preparation, model construction, training, evaluation, and paper-audit logic remain modular.

## Requirements

- Python **3.11 or 3.12**.
- GNU Make or a compatible `make` implementation.
- CICDDoS2019 available locally.
- macOS Apple Silicon or Linux.
- For Linux GPU execution: a working NVIDIA driver visible to TensorFlow.
- Sufficient storage for the compressed sampled CSV, encoded memory-mapped arrays, standardized splits, model checkpoints, predictions, reports, and logs.
- macOS completion playback uses the built-in `afplay` command.
- Linux completion playback optionally uses `aplay`; missing utilities or headless audio devices do not fail the experiment.

The platform-aware `requirements.txt` installs TensorFlow 2.18.1, `tensorflow-metal` 1.2.0 on Apple Silicon, TensorFlow CUDA user-space dependencies on Linux x86_64, NumPy, pandas, scikit-learn, matplotlib, joblib, and psutil. `Logger.py` and sound playback use only the Python standard library and operating-system commands.

## Setup

### Clone the repository

```bash
git clone https://github.com/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019.git
cd GRU-DDoS-Detection-CICDDoS2019
```

No separate installation script is required. The Makefile creates `.venv`, upgrades `pip`, `setuptools`, and `wheel`, installs the platform-aware `requirements.txt`, verifies TensorFlow GPU availability, and executes the experiment.

A manual environment remains supported:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py --help
```
