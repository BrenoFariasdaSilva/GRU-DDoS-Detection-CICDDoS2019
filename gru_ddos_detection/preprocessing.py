"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 TRAIN/TEST SPLIT AND STANDARDIZATION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Implements the per-run 70/30 split, configurable StandardScaler scope, generated-dataset
    persistence, one-timestep reshape, and one-hot output encoding used before GRU training.

    Key features include:
        - Preserves optional stratification and per-run split-seed progression.
        - Supports the supplied separate and train-only StandardScaler modes.
        - Saves standardized train/test arrays and integer labels for auditability.

Usage:
    1. Call split_and_scale() once per experiment run.
    2. Pass returned sequence arrays and one-hot labels to experiment training.
    3. Inspect generated_dataset and scaler artifacts in the run directory.

Outputs:
    - Split-index .npy files, scaler joblib files, and generated standardized dataset copies.

TODOs:
    - None identified.

Dependencies:
    - joblib.
    - numpy.
    - scikit-learn.
    - tensorflow.
    - Python standard library.
    - gru_ddos_detection.config and constants.

Assumptions & Notes:
    - Separate train/test scaler fitting is intentionally retained as the default reproduction
      mode because it is part of the supplied implementation's high-fidelity reconstruction.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from .tensorflow_runtime import tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import PAPER_FIGURE6_CLASSES, PAPER_TEST_FRACTION


def split_run_arrays(X: np.ndarray, y: np.ndarray, cfg: Config, run_index: int, run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split encoded sampled arrays into materialized 70% train and 30% test partitions.

    :param X: Complete encoded sampled feature matrix.
    :param y: Complete encoded sampled integer-label vector.
    :param cfg: Immutable experiment configuration.
    :param run_index: One-based repeated-run index.
    :param run_dir: Directory receiving split-index artifacts.
    :return: Materialized float32 train/test features and int16 train/test labels.
    """

    indices = np.arange(len(y), dtype=np.int64)  # Create original-row indices for reproducible split persistence
    split_seed = cfg.split_seed + run_index - 1  # Preserve per-run split-seed progression
    stratify_values = np.asarray(y) if cfg.stratify else None  # Preserve optional stratification against encoded class IDs
    train_indices, test_indices = train_test_split(
        indices,
        test_size=PAPER_TEST_FRACTION,
        random_state=split_seed,
        shuffle=True,
        stratify=stratify_values,
    )  # Preserve the exact scikit-learn 70/30 split call
    np.save(run_dir / "train_indices.npy", train_indices, allow_pickle=False)  # Persist original-row training indices
    np.save(run_dir / "test_indices.npy", test_indices, allow_pickle=False)  # Persist original-row test indices
    X_train = np.asarray(X[train_indices], dtype=np.float32)  # Materialize only sampled training rows as float32
    X_test = np.asarray(X[test_indices], dtype=np.float32)  # Materialize only sampled test rows as float32
    y_train = np.asarray(y[train_indices], dtype=np.int16)  # Materialize aligned training labels as int16
    y_test = np.asarray(y[test_indices], dtype=np.int16)  # Materialize aligned test labels as int16
    return X_train, X_test, y_train, y_test  # Return split arrays before standardization
