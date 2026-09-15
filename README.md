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

## Makefile Automation

The former `run-mac.txt` and `run-linux.txt` files have been replaced by one Makefile with explicit operating-system targets:

| Target | Purpose |
| --- | --- |
| `make run` | Detect macOS/Linux and dispatch to the corresponding target. |
| `make run-mac` | Run bounded per-class sampling on Apple Silicon. |
| `make run-linux` | Run every cleaned `01-12` target row on a Linux GPU server. |
| `make tail-log` | Follow `logs/main.log`. |
| `make run-linux DETACH=1` | Start a detached SSH execution and write `logs/main.pid`. |
| `make stop` | Stop the process recorded in `logs/main.pid`. |
| `make show-config` | Print effective paths and tunable Make variables. |
| `make clean` | Remove `.venv` and Python cache files without deleting results. |

All major controls are overridable inline. For example:

```bash
make run-linux LINUX_BATCH_SIZE=2048 CHUNKSIZE=100000 RUNS=1
```

## Installation and Execution

### macOS Apple Silicon — bounded-memory execution

The defaults expect the dataset at `/Users/brenofarias/Downloads/RAW Datasets/CICDDoS2019`, cap each class at 100,000 rows, use batch size 256, and keep output inside this repository:

```bash
make run-mac
```

```bash
make run-mac MAC_DATA_DIR="/another/path/CICDDoS2019" MAC_PER_CLASS_CAP=50000 MAC_BATCH_SIZE=128
```

### Linux SSH server — full-source execution

The Linux defaults expect:

```text
Dataset: ~/DDoS-Detector/Datasets/CICDDoS2019
Output:  <repository>/GRU-DDoS-Detection-Full
```

From the project directory:

```bash
make run-linux
```

For a long SSH execution:

```bash
make run-linux DETACH=1
make tail-log
```

The output directory is generated from the Makefile's absolute `PROJECT_DIR`, so it remains inside the repository regardless of the caller's current directory.

## Runtime Logging and Completion Sound

At startup, `main.py` creates a fresh `logs/main.log` and assigns the same `Logger` instance to `sys.stdout` and `sys.stderr`. Every raw-scan ETA, encoding message, training callback, prediction update, warning, and final metric remains visible locally while an ANSI-clean copy is flushed immediately to disk.

At interpreter shutdown, `atexit` invokes `.assets/Sounds/NotificationSound.wav` through:

- `afplay` on macOS;
- `aplay -q` on Linux.

The sound is attempted on successful completion and on exits caused by an exception or CLI termination. Playback is deliberately non-fatal: an unavailable Linux audio command, a headless server, or a missing audio device produces only a log warning and never changes the experiment result or exit status.

## Configuration

Important CLI controls include:

| Option | Purpose |
| --- | --- |
| `--data-dir` | Read-only CICDDoS2019 root. |
| `--output-dir` | Project-local generated output directory. |
| `--source-day` | `01-12`, `03-11`, or `both`. |
| `--sampling-profile` | `figure6-inferred`, `cap-per-class`, or `all`. |
| `--per-class-cap` | Limit used by `cap-per-class`. |
| `--chunksize` | Streamed CSV chunk size. |
| `--data-seed` | Exact sampling random seed. |
| `--split-seed` | 70/30 split random seed. |
| `--model-seed` | TensorFlow/model random seed. |
| `--batch-size` | Training batch size. |
| `--validation-mode` | `test` or rigorous `holdout`. |
| `--scaling-mode` | `separate` or rigorous `train-only`. |
| `--no-stratify` | Disable split stratification. |
| `--reuse-cache` | Reuse `encoded_sample_X.npy`/`encoded_sample_y.npy`. |
| `--allow-cpu` | Permit execution without a detected TensorFlow GPU. |

Run `python main.py --help` for all model and early-stopping options.

## Generated Outputs

The output root contains reproducibility, paper-audit, sampling, encoding, and aggregate artifacts such as:

```text
config.json
environment.json
reconstruction_assumptions.json
paper_internal_consistency_audit.json
published_top20_features.json
raw_dataset_snapshot_before.json
raw_dataset_snapshot_after.json
source_csv_files.txt
sampling_report.json
encoded_sample_X.npy
encoded_sample_y.npy
runs_summary.csv
aggregate_metrics.json
derived_dataset/
encoders/
```

Each run directory contains artifacts such as:

```text
train_indices.npy
test_indices.npy
validation_indices_within_train.npy   # holdout mode only
standard_scaler_train.joblib
standard_scaler_test.joblib           # separate mode only
model_summary.txt
best_model.keras
training_history.csv
test_class_support.json
confusion_matrix.csv
confusion_matrix.png
classification_report.json
metrics.json
test_predictions.csv
generated_dataset/
```

## Reproducibility Notes and Limitations

The paper provides many hyperparameters, but an exact independent reproduction is still not guaranteed. Important unresolved or contradictory items include:

- the exact source-file sampling/downsampling procedure;
- the exact population used to fit Extra Trees;
- the prose class list versus the labels visible in Figure 6(c);
- sequence/window length, stride, ordering, overlap, and grouping;
- dropout rate and exact placement;
- random seeds;
- stratification;
- StandardScaler fitting scope;
- how validation data were created despite the reported 70/30 train/test split;
- multiclass F1 averaging convention;
- number of independent runs;
- whether cross-validation was actually executed for the reported GRU result;
- software/CUDA/cuDNN versions.

The repository records these distinctions in `reconstruction_assumptions.json`.

The project also persists `paper_internal_consistency_audit.json`, which recomputes metrics from the transcribed Figure 6(c) GRU confusion matrix. This is intentionally separate from the model's own generated metrics so that publication-level values and reproduction results are not conflated.

For the matrix currently transcribed in the project, the 755,755 cells contain 737,109 correct predictions, corresponding to **97.5328% accuracy**; its macro F1 is approximately **0.89917** and weighted F1 approximately **0.96587**. These values do not match Table 4's 99.54% accuracy / 98% F1, so the project reports the discrepancy explicitly rather than treating the matrix and table as the same result.

## Results Target

The repository does not force the published values. It measures how closely an independent reconstruction approaches them.

```text
Paper target accuracy = 0.9954
Paper-reported F1     = 98% (represented as target 0.9800)
```

Because the publication does not state the multiclass F1 averaging convention, each run records macro, weighted, and micro F1.

## References

1. M. Ramzan, M. Shoaib, A. Altaf, S. Arshad, F. Iqbal, Á. Kuc Castilla, and I. Ashraf, “Distributed Denial of Service Attack Detection in Network Traffic Using Deep Learning Algorithm,” *Sensors*, vol. 23, no. 20, article 8642, 2023. [https://doi.org/10.3390/s23208642](https://doi.org/10.3390/s23208642)
2. Canadian Institute for Cybersecurity, University of New Brunswick, “DDoS 2019 (CICDDoS2019).” [https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)

## How to Cite

If you use this repository, cite both the reproduction software and the original paper. The root [`main.bib`](main.bib) contains both entries.

```bibtex
@misc{farias2026gruddosdetection,
  author       = {Breno Farias da Silva},
  title        = {GRU DDoS Detection on CICDDoS2019},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/BrenoFariasdaSilva/GRU-DDoS-Detection-CICDDoS2019},
  note         = {Reproduction implementation of the Ramzan et al. (2023) CICDDoS2019 multiclass GRU experiment}
}

@article{ramzan2023distributed,
  author  = {Ramzan, Mahrukh and Shoaib, Muhammad and Altaf, Ayesha and Arshad, Shazia and Iqbal, Faiza and Castilla, {\'A}ngel Kuc and Ashraf, Imran},
  title   = {Distributed Denial of Service Attack Detection in Network Traffic Using Deep Learning Algorithm},
  journal = {Sensors},
  volume  = {23},
  number  = {20},
  pages   = {8642},
  year    = {2023},
  doi     = {10.3390/s23208642},
  url     = {https://doi.org/10.3390/s23208642}
}
```

If you find the repository useful, consider starring it and opening issues or pull requests that improve reproducibility or documentation.

## Contributing

Contributions are welcome when they preserve a clear distinction between **paper-faithful reconstruction** and **methodologically improved alternatives**.

1. Fork the repository and create a focused branch.
2. Keep paper-derived and inferred settings explicitly separated.
3. Preserve the existing source/function documentation rules.
4. Validate both constrained and full-source paths where practical.
5. Use clear commit messages, for example:
   - `FEAT: Add ...`
   - `FIX: Resolve ...`
   - `DOCS: Update ...`
   - `REFACTOR: Improve ...`
6. Open a pull request explaining any scientific or reproducibility impact.

## Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/BrenoFariasdaSilva" title="Breno Farias da Silva on GitHub">
        <img src="https://github.com/BrenoFariasdaSilva.png" width="100px;" alt="Breno Farias da Silva"/><br>
        <sub><b>Breno Farias da Silva</b></sub>
      </a>
    </td>
  </tr>
</table>
