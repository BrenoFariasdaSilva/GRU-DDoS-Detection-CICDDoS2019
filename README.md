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
