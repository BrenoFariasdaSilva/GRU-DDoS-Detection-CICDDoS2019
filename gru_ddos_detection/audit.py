"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 PAPER AUDIT AND RECONSTRUCTION METADATA
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Computes the supplied Figure 6(c) confusion-matrix audit and builds the reconstruction
    assumptions manifest persisted by the GRU CICDDoS2019 reproduction workflow.

    Key features include:
        - Recomputes accuracy, precision, recall, and F1 from the transcribed paper matrix.
        - Reports the mathematical discrepancy between Table 4 and the transcribed Figure 6(c) matrix.
        - Builds the explicit published-versus-inferred reconstruction assumptions manifest.

Usage:
    1. Call paper_matrix_audit() before training.
    2. Call build_reconstruction_assumptions() with the resolved Config.
    3. Persist returned dictionaries through gru_ddos_detection.persistence.json_dump().

Outputs:
    - In-memory dictionaries for paper_internal_consistency_audit.json and reconstruction_assumptions.json.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - gru_ddos_detection.config.
    - gru_ddos_detection.constants.

Assumptions & Notes:
    - Audit metrics are computed directly from the transcribed Figure 6(c) matrix; the human-readable
      note explicitly distinguishes those values from the Table 4 targets.
================================================================================
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import Config
from .constants import (
    FIGURE6_INFERRED_CLASS_QUOTAS,
    PAPER_FIGURE6_CLASSES,
    PAPER_FIGURE6_GRU_CM,
    PAPER_TARGET_ACCURACY,
    PAPER_TARGET_F1,
)

# Functions Definitions:


def paper_matrix_audit() -> Dict[str, object]:
    """
    Compute the same publication confusion-matrix audit as the supplied implementation.

    :return: Dictionary containing Table 4 targets and Figure 6(c) recomputed statistics.
    """

    confusion = PAPER_FIGURE6_GRU_CM  # Use the exact transcribed matrix embedded in the supplied implementation
    support = confusion.sum(axis=1)  # Compute per-class true support from matrix row sums
    total = int(confusion.sum())  # Compute the total number of matrix observations
    diagonal = np.diag(confusion).astype(np.float64)  # Extract correct predictions as floating-point values
    predicted_totals = confusion.sum(axis=0).astype(np.float64)  # Compute per-class predicted totals from matrix columns
    precision = np.divide(diagonal, predicted_totals, out=np.zeros_like(diagonal), where=predicted_totals != 0)  # Compute class precision with zero-safe division
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support != 0)  # Compute class recall with zero-safe division
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diagonal), where=(precision + recall) != 0)  # Compute class F1 with zero-safe division
    accuracy = float(np.trace(confusion) / total)  # Compute matrix accuracy from diagonal trace over total observations
    macro_f1 = float(np.mean(f1))  # Compute unweighted mean class F1
    weighted_f1 = float(np.sum(f1 * support) / total)  # Compute support-weighted class F1
    macro_precision = float(np.mean(precision))  # Compute unweighted mean class precision
    macro_recall = float(np.mean(recall))  # Compute unweighted mean class recall
    return {
        "table4_reported_accuracy": PAPER_TARGET_ACCURACY,
        "table4_reported_precision": 0.98,
        "table4_reported_recall": 0.99,
        "table4_reported_f1": PAPER_TARGET_F1,
        "figure6c_total_test_rows": total,
        "figure6c_correct_predictions_trace": int(np.trace(confusion)),
        "figure6c_accuracy_computed_from_cells": accuracy,
        "figure6c_macro_precision_computed_from_cells": macro_precision,
        "figure6c_macro_recall_computed_from_cells": macro_recall,
        "figure6c_macro_f1_computed_from_cells": macro_f1,
        "figure6c_weighted_f1_computed_from_cells": weighted_f1,
        "figure6c_test_support": {class_name: int(count) for class_name, count in zip(PAPER_FIGURE6_CLASSES, support)},
        "consistency_note": (
            "Figure 6(c) is NOT mathematically consistent with the Table 4 values: "
            f"matrix accuracy={accuracy:.8f} versus Table 4 accuracy={PAPER_TARGET_ACCURACY:.4f}; "
            f"matrix macro precision={macro_precision:.8f}, macro recall={macro_recall:.8f}, "
            f"macro F1={macro_f1:.8f}, and weighted F1={weighted_f1:.8f} versus "
            f"Table 4 reported F1={PAPER_TARGET_F1:.4f}."
        ),
    }  # Return the audit values with a mathematically accurate comparison to Table 4


def build_reconstruction_assumptions(cfg: Config) -> Dict[str, object]:
    """
    Build the explicit published-versus-inferred reconstruction assumptions manifest.

    :param cfg: Immutable resolved experiment configuration.
    :return: Dictionary persisted as reconstruction_assumptions.json.
    """

    return {
        "paper_specified": {
            "missing_null_handling": "remove records",
            "normalization": "StandardScaler / mean 0 std 1",
            "feature_selection": "ExtraTrees, final published top 20",
            "split": "70% train / 30% test",
            "gru_layers": 2,
            "gru_units_per_layer": 8,
            "hidden_dense_units": [16, 8],
            "activation": "ReLU hidden, Softmax multiclass",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "categorical cross entropy",
            "batch_size": 1000,
            "max_epochs": 100,
            "early_stopping": {"monitor": "val_loss", "min_delta": 0.001, "patience": 5},
            "dropout": "used, rate and location not published",
        },
        "publication_missing_or_ambiguous": [
            "exact source-file sampling/downsampling procedure and sample counts",
            "class list conflicts between prose and Figure 6(c)",
            "dropout rate and exact placement",
            "validation partition creation despite only a 70/30 train/test split being specified",
            "GRU input sequence/window construction, ordering, stride, overlap, and timestep count",
            "random seed(s)",
            "whether train_test_split was stratified",
            "whether scaler was fit on train only or separately/before split",
            "exact ExtraTrees hyperparameters/training population",
            "exact number of independent runs",
            "whether any cross-validation was actually performed (none is reported)",
            "metric averaging convention for multiclass precision/recall/F1",
            "TensorFlow/Keras/Python/CUDA/cuDNN versions",
        ],
        "reconstruction_defaults": {
            "classes": list(PAPER_FIGURE6_CLASSES),
            "source_day": cfg.source_day,
            "sampling_profile": cfg.sampling_profile,
            "inferred_class_quotas": FIGURE6_INFERRED_CLASS_QUOTAS,
            "dropout": cfg.dropout,
            "input_shape": [1, 20],
            "split_seed": cfg.split_seed,
            "model_seed": cfg.model_seed,
            "validation_mode": cfg.validation_mode,
            "scaling_mode": cfg.scaling_mode,
            "stratify": cfg.stratify,
            "external_clue_warning": (
                "Some defaults are informed by a pre-existing, non-author CICDDoS2019 notebook with the same exact "
                "top-20 feature list. It is not an official implementation of Ramzan et al."
            ),
        },
    }  # Preserve the supplied assumptions manifest exactly while moving construction out of main.py
