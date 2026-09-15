"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 SINGLE-RUN GRU EXPERIMENT EXECUTION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Coordinates one complete GRU CICDDoS2019 reproduction run from split/standardization through
    validation selection, GRU training, best-checkpoint loading, prediction, and test artifacts.

    Key features include:
        - Preserves per-run model/split seed progression and output-directory naming.
        - Supports test-as-validation and rigorous within-training holdout validation modes.
        - Preserves early stopping, checkpointing, CSV history, ETA callback, and cleanup behavior.

Usage:
    1. Prepare validated encoded sample arrays through gru_ddos_detection.workflow.
    2. Call run_experiment() for each one-based run index.
    3. Aggregate returned metric dictionaries after all requested runs complete.

Outputs:
    - Per-run generated data, split indices, scalers, model checkpoint/summary/history,
      test support, confusion artifacts, classification report, metrics, and predictions.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - psutil indirectly through training callback.
    - scikit-learn.
    - tensorflow.
    - Python standard library.
    - gru_ddos_detection evaluation, model, persistence, preprocessing, system, and timing modules.

Assumptions & Notes:
    - validation-mode=test intentionally reuses the final test partition for early stopping,
      matching the supplied high-fidelity reconstruction mode despite test leakage.
================================================================================
"""

from __future__ import annotations

import gc
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .config import Config
from .constants import PAPER_FIGURE6_CLASSES, PAPER_TOP20
from .evaluation import compute_metrics, persist_evaluation_artifacts, predict_with_eta
from .model import build_gru, make_dataset
from .persistence import json_dump
from .preprocessing import split_and_scale
from .system import set_seeds
from .tensorflow_runtime import tf
from .timing import create_epoch_eta, human_bytes


def select_validation_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_train_onehot: np.ndarray, y_test_onehot: np.ndarray, cfg: Config, run_index: int, run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select model-fit and validation arrays according to the configured validation mode.

    :param X_train: Model-ready one-timestep training feature array.
    :param X_test: Model-ready one-timestep final test feature array.
    :param y_train: Integer training labels used only for optional validation stratification.
    :param y_train_onehot: One-hot categorical training labels.
    :param y_test_onehot: One-hot categorical final test labels.
    :param cfg: Immutable experiment configuration.
    :param run_index: One-based repeated-run index.
    :param run_dir: Current run directory receiving optional validation indices.
    :return: Fit features, fit one-hot labels, validation features, and validation one-hot labels.
    """

    if cfg.validation_mode == "test":  # Verify if high-fidelity test-as-validation behavior was requested
        X_fit, y_fit_onehot = X_train, y_train_onehot  # Train on the complete 70% training partition
        X_validation, y_validation_onehot = X_test, y_test_onehot  # Reuse the final 30% test partition for validation and early stopping
        print("[WARNING] validation-mode=test: held-out test set is also used for early stopping (reproduction mode).")  # Preserve the original leakage warning
        return X_fit, y_fit_onehot, X_validation, y_validation_onehot  # Return the high-fidelity validation construction
    indices = np.arange(len(y_train), dtype=np.int64)  # Build within-training indices for rigorous validation splitting
    fit_indices, validation_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=cfg.split_seed + run_index - 1,
        shuffle=True,
        stratify=y_train if cfg.stratify else None,
    )  # Preserve the original optional-stratified 10% within-training validation split
    X_fit, X_validation = X_train[fit_indices], X_train[validation_indices]  # Materialize fit and validation recurrent features
    y_fit_onehot, y_validation_onehot = y_train_onehot[fit_indices], y_train_onehot[validation_indices]  # Materialize aligned categorical labels
    np.save(run_dir / "validation_indices_within_train.npy", validation_indices, allow_pickle=False)  # Preserve within-training validation-index artifact
    print(f"[VALIDATION] rigorous holdout: fit={len(fit_indices):,}, val={len(validation_indices):,}, final_test={len(X_test):,}")  # Preserve the original rigorous validation message
    return X_fit, y_fit_onehot, X_validation, y_validation_onehot  # Return rigorous fit and validation arrays


def write_model_summary_line(handle: TextIO, line: str) -> None:
    """
    Write one Keras model-summary line to the open text destination.

    :param handle: Open text file receiving model-summary lines.
    :param line: One model-summary line supplied by Keras.
    :return: None.
    """

    handle.write(line + "\n")  # Preserve one newline after every Keras model-summary line
