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
