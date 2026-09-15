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


def predict_with_eta(model: tf.keras.Model, test_dataset: tf.data.Dataset, test_rows: int, batch_size: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict test probabilities batch by batch while reporting progress ETA.

    :param model: Trained best Keras model used for inference.
    :param test_dataset: Batched final test dataset.
    :param test_rows: Number of final test observations.
    :param batch_size: Configured inference batch size.
    :param device: TensorFlow device selected for model execution.
    :return: Concatenated probability matrix and integer predicted class IDs.
    """

    probability_parts: List[np.ndarray] = []  # Collect per-batch prediction arrays before final concatenation
    total_batches = math.ceil(test_rows / batch_size)  # Preserve the original expected prediction batch count
    prediction_eta = create_eta("PREDICT", total_batches)  # Initialize the original prediction ETA reporter
    for batch_index, (features, _) in enumerate(test_dataset, 1):  # Iterate final test batches in deterministic dataset order
        with tf.device(device):  # Preserve explicit model inference device placement
            probabilities = model(features, training=False).numpy()  # Execute direct inference without model.predict opacity
        probability_parts.append(probabilities)  # Preserve each prediction block for final row-order concatenation
        prediction_eta.report(batch_index, f"batch={batch_index}/{total_batches}")  # Preserve the original prediction progress detail
    prediction_eta.report(total_batches, "prediction complete", force=True)  # Preserve the forced prediction completion message
    all_probabilities = np.concatenate(probability_parts, axis=0)  # Reassemble prediction rows in test-dataset order
    predictions = all_probabilities.argmax(axis=1).astype(np.int16)  # Preserve integer class selection and output dtype
    return all_probabilities, predictions  # Return probabilities and predicted class IDs for metrics/artifacts


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, cfg: Config, run_index: int, train_seconds: float, epochs_completed: int) -> Dict[str, object]:
    """
    Compute scalar test metrics and distances to the paper targets for one run.

    :param y_test: Integer ground-truth labels for the final test partition.
    :param y_pred: Integer predicted labels aligned with y_test.
    :param cfg: Immutable experiment configuration.
    :param run_index: One-based repeated-run index.
    :param train_seconds: Completed model-fit wall-clock duration.
    :param epochs_completed: Number of epochs actually completed before stopping.
    :return: Metrics dictionary matching the supplied main.py fields.
    """

    accuracy = float(accuracy_score(y_test, y_pred))  # Compute final test accuracy once for reporting and target distance
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))  # Compute unweighted multiclass F1
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))  # Compute support-weighted multiclass F1
    model_seed = cfg.model_seed + run_index - 1  # Preserve per-run model-seed progression in metrics metadata
    split_seed = cfg.split_seed + run_index - 1  # Preserve per-run split-seed progression in metrics metadata
    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": f1_macro,
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": f1_weighted,
        "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "paper_target_accuracy": PAPER_TARGET_ACCURACY,
        "paper_target_f1": PAPER_TARGET_F1,
        "distance_accuracy": abs(accuracy - PAPER_TARGET_ACCURACY),
        "distance_f1_macro": abs(f1_macro - PAPER_TARGET_F1),
        "distance_f1_weighted": abs(f1_weighted - PAPER_TARGET_F1),
        "training_seconds": train_seconds,
        "epochs_completed": epochs_completed,
        "epochs_requested": cfg.epochs,
        "run": run_index,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "validation_mode": cfg.validation_mode,
        "scaling_mode": cfg.scaling_mode,
        "stratify": cfg.stratify,
    }  # Preserve the supplied per-run metrics field names and meanings


def persist_evaluation_artifacts(run_dir: Path, y_test: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, metrics: Dict[str, object]) -> None:
    """
    Persist all final test evaluation artifacts for one experiment run.

    :param run_dir: Current experiment run directory.
    :param y_test: Integer ground-truth labels for the final test partition.
    :param y_pred: Integer predicted labels aligned with y_test.
    :param probabilities: Softmax probability matrix aligned with y_test.
    :param metrics: Scalar metrics dictionary for the current run.
    :return: None.
    """

    class_ids = np.arange(len(PAPER_FIGURE6_CLASSES))  # Build canonical integer class IDs in Figure 6(c) order
    confusion = confusion_matrix(y_test, y_pred, labels=class_ids)  # Compute the complete 12-class confusion matrix
    pd.DataFrame(confusion, index=PAPER_FIGURE6_CLASSES, columns=PAPER_FIGURE6_CLASSES).to_csv(run_dir / "confusion_matrix.csv")  # Persist raw confusion counts with canonical row/column labels
    save_confusion(confusion, PAPER_FIGURE6_CLASSES, run_dir / "confusion_matrix.png")  # Persist the matching visual confusion matrix
    report = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=PAPER_FIGURE6_CLASSES,
        output_dict=True,
        zero_division=0,
    )  # Compute the same structured per-class classification report
    json_dump(run_dir / "classification_report.json", report)  # Persist the structured classification report
    json_dump(run_dir / "metrics.json", metrics)  # Persist scalar run metrics
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "true_label": [PAPER_FIGURE6_CLASSES[class_id] for class_id in y_test],
        "predicted_label": [PAPER_FIGURE6_CLASSES[class_id] for class_id in y_pred],
        "predicted_probability": probabilities.max(axis=1),
    }).to_csv(run_dir / "test_predictions.csv", index=False)  # Preserve the original per-observation prediction artifact columns and order
