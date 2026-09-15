"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 HELD-OUT TEST EVALUATION AND ARTIFACTS
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Performs batch-wise GRU prediction with ETA reporting, computes the supplied multi-class
    metrics, and persists confusion, classification, metrics, and prediction artifacts.

    Key features include:
        - Predicts the final 30% test partition in explicit batches with ETA messages.
        - Computes macro, weighted, and micro metrics plus distance to paper targets.
        - Saves confusion matrix CSV/PNG, classification report JSON, metrics JSON, and predictions.

Usage:
    1. Call predict_with_eta() on the trained best model and test dataset.
    2. Compute scalar metrics with compute_metrics().
    3. Persist final artifacts with persist_evaluation_artifacts().

Outputs:
    - Per-run held-out test metrics, confusion artifacts, report JSON, and predictions CSV.

TODOs:
    - None identified.

Dependencies:
    - matplotlib.
    - numpy.
    - pandas.
    - scikit-learn.
    - tensorflow.
    - Python standard library.
    - gru_ddos_detection.constants, persistence, and timing.

Assumptions & Notes:
    - Metric averaging conventions are intentionally all reported because the paper does not
      specify which multiclass F1 convention produced its rounded Table 4 value.
================================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .config import Config
from .constants import PAPER_FIGURE6_CLASSES, PAPER_TARGET_ACCURACY, PAPER_TARGET_F1
from .persistence import json_dump
from .tensorflow_runtime import tf
from .timing import create_eta


def save_confusion(confusion: np.ndarray, labels: Sequence[str], path: Path) -> None:
    """
    Save the multi-class confusion matrix as a labeled PNG image.

    :param confusion: Square confusion matrix array.
    :param labels: Ordered class labels for both matrix axes.
    :param path: Destination PNG path.
    :return: None.
    """

    figure, axes = plt.subplots(figsize=(12, 10))  # Create the original confusion-matrix figure size
    image = axes.imshow(confusion)  # Render raw confusion counts without changing values
    figure.colorbar(image, ax=axes)  # Preserve the original color scale display
    axes.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")  # Label predicted-class axis in canonical class order
    axes.set_yticks(np.arange(len(labels)), labels=labels)  # Label actual-class axis in canonical class order
    axes.set_xlabel("Predicted")  # Preserve the original x-axis label
    axes.set_ylabel("Actual")  # Preserve the original y-axis label
    axes.set_title("GRU DDoS Detection on CICDDoS2019 — Ramzan et al. (2023) reproduction")  # Identify the method-centric project and source paper
    figure.tight_layout()  # Fit long class labels into the saved figure bounds
    figure.savefig(path, dpi=180)  # Preserve the original PNG resolution
    plt.close(figure)  # Release matplotlib resources after persistence
