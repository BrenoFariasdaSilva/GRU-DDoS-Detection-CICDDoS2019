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


def inspect_schemas(files: Sequence[Path]) -> List[FileSchema]:
    """
    Read CSV headers and map every file onto the published top-20 feature names.

    :param files: Ordered source CSV files to inspect.
    :return: Ordered FileSchema objects for streamed processing.
    """

    schemas: List[FileSchema] = []  # Collect one validated schema per source CSV
    wanted = {normalize_column(name): name for name in PAPER_TOP20}  # Build normalized lookup keys for every published selected feature
    for path in files:  # Inspect each source CSV header without loading data rows
        header = pd.read_csv(path, nrows=0)  # Read only column names from the current source file
        raw_columns = [str(column) for column in header.columns]  # Preserve exact source header spellings
        label = infer_label(raw_columns)  # Resolve the exact label header
        by_key = {normalize_column(column): column for column in raw_columns if normalize_column(column)}  # Map normalized keys back to exact source headers
        missing = [display for key, display in wanted.items() if key not in by_key]  # Identify any published selected features absent from this source file
        if missing:  # Verify if the current source file lacks required published features
            raise ValueError(f"{path.name} is missing published top-20 columns: {missing}")  # Preserve the original schema failure
        selected = {display: by_key[key] for key, display in wanted.items()}  # Map canonical selected-feature names to exact source headers
        validity = tuple(
            column for column in raw_columns
            if column != label and not normalize_column(column).startswith("unnamed")
        )  # Preserve row-validity coverage across every non-index, non-label source field
        schemas.append(FileSchema(path, label, selected, validity))  # Store the validated source schema
    return schemas  # Return schemas in source-file order


def canonical_labels(series: pd.Series) -> pd.Series:
    """
    Canonicalize raw source labels through the supplied alias mapping.

    :param series: Raw source label series.
    :return: Series containing canonical labels or missing values for unknown labels.
    """

    return series.astype("string").fillna("").map(normalize_token).map(BASE_LABEL_ALIASES)  # Preserve the original vectorized label canonicalization chain


def row_validity_mask(chunk: pd.DataFrame, schema: FileSchema) -> pd.Series:
    """
    Build the original row-validity mask for one streamed source chunk.

    :param chunk: Source CSV chunk containing validity and label columns.
    :param schema: Exact source-column mapping for the current CSV file.
    :return: Boolean Series identifying rows that survive cleaning.
    """

    validity = chunk.loc[:, list(schema.validity_cols)]  # Inspect every non-index, non-label source field for paper-aligned row removal
    valid = ~validity.isna().any(axis=1)  # Reject rows containing pandas-recognized missing values
    numeric_columns = list(validity.select_dtypes(include=[np.number]).columns)  # Identify already-numeric columns for efficient finite-value testing
    if numeric_columns:  # Verify if any validity columns have numeric dtypes
        numeric = validity[numeric_columns].to_numpy(copy=False)  # Access numeric validity values without an unnecessary copy
        try:  # Attempt vectorized finite-value validation across numeric source fields
            valid &= np.isfinite(numeric).all(axis=1)  # Reject rows containing positive or negative infinity in numeric fields
        except TypeError:  # Preserve the original tolerance for mixed/object numeric arrays that cannot be tested directly
            pass  # Leave the existing missing-value mask unchanged for this vectorized check
    for display in PAPER_TOP20:  # Apply the stronger downstream-validity guard to each selected feature
        if display in CATEGORICAL_SELECTED:  # Verify if the selected feature is categorical in the supplied reconstruction
            column = schema.selected_actual[display]  # Resolve the exact source header for the categorical feature
            values = chunk[column].astype("string")  # Convert categorical values to pandas string dtype
            valid &= values.notna() & (values.str.strip() != "")  # Reject missing and empty categorical values
        else:  # Handle selected features expected to be numeric downstream
            column = schema.selected_actual[display]  # Resolve the exact source header for the numeric feature
            numeric_values = pd.to_numeric(chunk[column], errors="coerce")  # Coerce non-numeric selected values to missing
            array = numeric_values.to_numpy(dtype=np.float64, copy=False)  # Materialize a float64 view for finite-value testing
            valid &= np.isfinite(array)  # Reject non-numeric, NaN, and infinite selected numeric values
    return valid  # Return the complete cleaning mask for the current chunk


def read_usecols(schema: FileSchema) -> List[str]:
    """
    Build the deduplicated source-column list required for streamed cleaning and labeling.

    :param schema: Exact source-column mapping for one CSV file.
    :return: Ordered source headers passed to pandas read_csv(usecols=...).
    """

    return list(dict.fromkeys(list(schema.validity_cols) + [schema.label_col]))  # Preserve source validity-column order and append the label once
