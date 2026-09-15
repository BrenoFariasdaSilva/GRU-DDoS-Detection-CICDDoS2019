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


def count_one_file(schema: FileSchema, root: Path, chunksize: int, eta: ETA, completed_bytes: int, file_index: int, file_total: int) -> Tuple[Counter[str], Counter[str]]:
    """
    Count cleaned target and omitted rows for one source CSV file.

    :param schema: Exact schema mapping for the current source CSV.
    :param root: Raw dataset root used for relative progress paths.
    :param chunksize: Number of source rows read per pandas chunk.
    :param eta: Shared pass-one byte-progress reporter.
    :param completed_bytes: Bytes belonging to source files completed before this file.
    :param file_index: One-based index of the current file.
    :param file_total: Total number of source files in the pass.
    :return: Target-class and omitted-class counters for the current file.
    """

    file_counts: Counter[str] = Counter()  # Count cleaned target labels for this file only
    file_omitted: Counter[str] = Counter()  # Count cleaned non-target canonical labels for this file only
    file_size = schema.path.stat().st_size  # Read source size for byte-based ETA progress
    print(f"[DATA][PASS1] file {file_index}/{file_total}: {schema.path.relative_to(root)}")  # Preserve the original per-file count-pass message
    with schema.path.open("rb") as raw_handle:  # Open the raw CSV strictly for binary reading
        reader = pd.read_csv(raw_handle, usecols=read_usecols(schema), chunksize=chunksize, low_memory=False)  # Stream only columns required by cleaning and labels
        for chunk_index, chunk in enumerate(reader, 1):  # Process the current source file chunk by chunk
            labels = canonical_labels(chunk[schema.label_col])  # Canonicalize raw labels before target filtering
            valid = row_validity_mask(chunk, schema)  # Apply the supplied missing/null/non-finite cleaning rules
            count_labels_for_chunk(labels, valid, file_counts, file_omitted)  # Accumulate cleaned canonical labels for this chunk
            if chunk_index % 5 == 0:  # Verify if the original five-chunk progress interval was reached
                position = min(raw_handle.tell(), file_size)  # Bound buffered file position to the current source-file size
                eta.report(
                    completed_bytes + position,
                    f"file={schema.path.name} chunk={chunk_index} valid_target={sum(file_counts.values()):,}",
                )  # Preserve the original PASS1 progress detail
            del chunk, labels, valid  # Release large chunk-local objects before the next iteration
            gc.collect()  # Preserve explicit garbage collection used by the supplied memory-safe implementation
    return file_counts, file_omitted  # Return per-file counts for global aggregation and reporting


def count_valid_rows(schemas: Sequence[FileSchema], root: Path, chunksize: int) -> Tuple[Counter[str], Dict[str, object]]:
    """
    Count all cleaned target rows across source CSVs during the first streaming pass.

    :param schemas: Ordered validated source-file schemas.
    :param root: Raw dataset root used for relative paths.
    :param chunksize: Number of source rows read per pandas chunk.
    :return: Global target counter and detailed first-pass report.
    """

    total_bytes = sum(schema.path.stat().st_size for schema in schemas)  # Sum source sizes for byte-based pass-one progress
    completed_bytes = 0  # Track complete source-file bytes already scanned
    eta = create_eta("PASS1-COUNT", total_bytes)  # Initialize the original pass-one ETA reporter
    counts: Counter[str] = Counter()  # Accumulate cleaned target populations across all source files
    omitted: Counter[str] = Counter()  # Accumulate cleaned canonical omitted populations across all source files
    file_reports: List[Dict[str, object]] = []  # Preserve one detailed count report per source file
    for file_index, schema in enumerate(schemas, 1):  # Scan source files in their validated deterministic order
        file_counts, file_omitted = count_one_file(schema, root, chunksize, eta, completed_bytes, file_index, len(schemas))  # Count one source file without loading the corpus at once
        counts.update(file_counts)  # Merge this file's target counts into the global population
        omitted.update(file_omitted)  # Merge this file's omitted counts into the global population
        completed_bytes += schema.path.stat().st_size  # Advance completed-byte accounting after finishing the file
        eta.report(completed_bytes, f"completed {schema.path.name}", force=True)  # Preserve forced end-of-file progress output
        file_reports.append({
            "file": str(schema.path.relative_to(root)),
            "valid_target_counts": dict(file_counts),
            "valid_omitted_counts": dict(file_omitted),
        })  # Preserve the original per-file sampling-audit fields
    return counts, {
        "valid_target_counts": dict(counts),
        "valid_omitted_counts": dict(omitted),
        "files": file_reports,
    }  # Return global populations and the complete pass-one audit report
