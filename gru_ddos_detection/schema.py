"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 CICDDOS2019 SCHEMA AND CLEANING UTILITIES
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Discovers CICDDoS2019 source CSVs, validates published top-20 features, canonicalizes
    labels, and applies the supplied missing/null/non-finite row-validity rules.

    Key features include:
        - Resolves source CSVs for 01-12, 03-11, or both days.
        - Maps exact CSV headers onto the published 20 selected features.
        - Builds row-validity masks that preserve the original cleaning semantics.

Usage:
    1. Discover source files with find_source_csvs().
    2. Inspect exact headers with inspect_schemas().
    3. Use canonical_labels(), row_validity_mask(), and read_usecols() during streamed passes.

Outputs:
    - In-memory FileSchema objects, canonical label series, and row-validity masks.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - Python standard library.
    - gru_ddos_detection.constants.

Assumptions & Notes:
    - Missing/null removal is evaluated across all non-index, non-label source fields before
      downstream use of the published selected features, matching the supplied main.py.
================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import BASE_LABEL_ALIASES, CATEGORICAL_SELECTED, PAPER_TOP20


@dataclass(frozen=True)
class FileSchema:
    """Store exact source-column mappings and validity columns for one CSV file."""

    path: Path
    label_col: str
    selected_actual: Mapping[str, str]
    validity_cols: Tuple[str, ...]


def normalize_token(value: object) -> str:
    """
    Normalize a raw label token to uppercase alphanumeric form.

    :param value: Raw label-like value to normalize.
    :return: Uppercase alphanumeric token used for alias lookup.
    """

    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())  # Preserve the original label token normalization regex


def normalize_column(value: object) -> str:
    """
    Normalize a source column name to lowercase alphanumeric form.

    :param value: Raw source-column value to normalize.
    :return: Lowercase alphanumeric key used for header matching.
    """

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())  # Preserve the original source-column normalization regex


def infer_label(columns: Sequence[str]) -> str:
    """
    Return the exact source header corresponding to the Label column.

    :param columns: Ordered source CSV column names.
    :return: Exact source header spelling for the label column.
    """

    for column in columns:  # Search headers in their original order
        if normalize_column(column) == "label":  # Verify if the normalized header identifies the label field
            return column  # Return the exact source header spelling for pandas usecols compatibility
    raise ValueError(f"Could not find Label column in {list(columns)[:20]}")  # Preserve the original missing-label failure


def find_source_csvs(data_dir: Path, source_day: str) -> List[Path]:
    """
    Discover source CSV files for the configured CICDDoS2019 day selection.

    :param data_dir: Root CICDDoS2019 directory.
    :param source_day: Source-day selector: 01-12, 03-11, or both.
    :return: Sorted list of matching source CSV paths.
    """

    if source_day.lower() == "both":  # Verify if both source days were requested
        files = sorted(path for path in data_dir.rglob("*.csv") if path.is_file())  # Discover every source CSV recursively
    else:  # Handle one explicitly requested source day
        day_dir = data_dir / source_day  # Resolve the conventional direct day subdirectory
        if day_dir.exists():  # Verify if the requested day exists directly below the dataset root
            files = sorted(path for path in day_dir.rglob("*.csv") if path.is_file())  # Discover CSVs below the direct day directory
        else:  # Fall back to matching the day component anywhere in recursive paths
            files = sorted(
                path for path in data_dir.rglob("*.csv")
                if path.is_file() and source_day in path.parts
            )  # Preserve the supplied fallback day-discovery behavior
    if not files:  # Verify if discovery produced at least one source CSV
        raise FileNotFoundError(f"No CSVs found for source day {source_day!r} under {data_dir}")  # Preserve the original discovery failure
    return files  # Return deterministic source ordering
