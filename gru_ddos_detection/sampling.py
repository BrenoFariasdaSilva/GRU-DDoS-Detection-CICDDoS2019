"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 TWO-PASS MEMORY-SAFE SAMPLING
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Implements the supplied two-pass CICDDoS2019 sampling procedure: first counting valid
    target rows and then writing an exact class-quota sample to compressed local storage.

    Key features include:
        - Counts cleaned target-class populations without loading the corpus into memory.
        - Computes Figure 6-inferred, cap-per-class, or all-row sampling quotas.
        - Uses the original hypergeometric conditional allocation for exact uniform sampling.

Usage:
    1. Call count_valid_rows() with validated FileSchema objects.
    2. Compute class quotas with determine_quotas().
    3. Call exact_sample_to_disk() to create sampled_selected_top20.csv.gz.

Outputs:
    - A gzip-compressed sampled selected-feature CSV and in-memory sampling reports.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - Python standard library.
    - gru_ddos_detection.constants, schema, and timing.

Assumptions & Notes:
    - Random-number calls and class iteration order preserve the supplied implementation's
      exact sampling procedure for a fixed NumPy version, seed, and source-file ordering.
================================================================================
"""

from __future__ import annotations

import gc
import gzip
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, TextIO, Tuple

import numpy as np
import pandas as pd

from .constants import FIGURE6_INFERRED_CLASS_QUOTAS, PAPER_FIGURE6_CLASSES, PAPER_TOP20, CATEGORICAL_SELECTED
from .schema import FileSchema, canonical_labels, read_usecols, row_validity_mask
from .timing import ETA, create_eta


def count_labels_for_chunk(labels: pd.Series, valid: pd.Series, file_counts: Counter[str], file_omitted: Counter[str]) -> None:
    """
    Accumulate valid canonical target and omitted label counts for one source chunk.

    :param labels: Canonical label series for the current source chunk.
    :param valid: Boolean cleaning mask aligned with labels.
    :param file_counts: Mutable target-class counter for the current source file.
    :param file_omitted: Mutable omitted-class counter for the current source file.
    :return: None.
    """

    good_labels = labels[valid]  # Restrict counting to rows that survive the original cleaning rules
    value_counts = good_labels.value_counts(dropna=True)  # Preserve exclusion of unknown labels that canonicalize to missing values
    for label, count in value_counts.items():  # Accumulate each canonical label present in the cleaned chunk
        class_name = str(label)  # Normalize pandas scalar labels to ordinary strings for dictionary keys
        if class_name in PAPER_FIGURE6_CLASSES:  # Verify if the label belongs to the Figure 6(c) target set
            file_counts[class_name] += int(count)  # Add cleaned target rows to the current file count
        else:  # Handle canonicalized labels intentionally outside the Figure 6(c) target set
            file_omitted[class_name] += int(count)  # Preserve omitted-label accounting from the supplied implementation
