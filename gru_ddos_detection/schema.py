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
