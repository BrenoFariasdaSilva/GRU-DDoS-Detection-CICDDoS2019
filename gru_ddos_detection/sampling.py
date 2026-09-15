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


def determine_quotas(counts: Counter[str], profile: str, per_class_cap: int) -> Dict[str, int]:
    """
    Determine per-class source sample quotas using the configured sampling profile.

    :param counts: Available cleaned target-row populations by canonical class.
    :param profile: Sampling profile name: figure6-inferred, cap-per-class, or all.
    :param per_class_cap: Requested per-class cap used by cap-per-class mode.
    :return: Actual per-class sampling quotas bounded by available populations.
    """

    quotas: Dict[str, int] = {}  # Build quotas in the published Figure 6(c) class order
    for class_name in PAPER_FIGURE6_CLASSES:  # Process every required target class exactly once
        available = int(counts.get(class_name, 0))  # Resolve cleaned source population for the current class
        if available <= 0:  # Verify if the required class is absent after cleaning
            raise RuntimeError(f"Required Figure-6 class {class_name!r} has zero valid source rows.")  # Preserve the original required-class failure
        if profile == "figure6-inferred":  # Verify if quotas should follow Figure 6(c) support divided by 30%
            desired = FIGURE6_INFERRED_CLASS_QUOTAS[class_name]  # Use the supplied inferred source quota
        elif profile == "cap-per-class":  # Verify if a fixed per-class cap was requested
            desired = per_class_cap  # Use the configured fixed cap
        elif profile == "all":  # Verify if every cleaned source row should be retained
            desired = available  # Request the complete cleaned class population
        else:  # Handle unexpected profile values outside CLI validation
            raise ValueError(profile)  # Preserve the original defensive profile failure
        quotas[class_name] = min(available, desired)  # Bound the requested quota by rows actually available
        if desired > available:  # Verify if the requested or inferred quota exceeds the cleaned source population
            print(
                f"[DATA] WARNING: {class_name}: inferred/requested {desired:,}, but only "
                f"{available:,} valid rows are available; using all available."
            )  # Preserve the original quota-shortfall warning
    return quotas  # Return actual exact quotas for the second pass


def normalize_selected_chunk(chunk: pd.DataFrame, schema: FileSchema, mask: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """
    Normalize selected sampled rows into canonical published feature columns plus Label.

    :param chunk: Original source chunk containing selected rows.
    :param schema: Exact source-column mapping for the current file.
    :param mask: Boolean mask selecting sampled rows in the chunk.
    :param labels: Canonical label series aligned with the source chunk.
    :return: DataFrame containing canonical top-20 columns and Label for selected rows.
    """

    output = pd.DataFrame(index=chunk.index[mask])  # Create an output frame aligned only to selected source rows
    for display in PAPER_TOP20:  # Materialize selected features in the published order
        source = schema.selected_actual[display]  # Resolve the exact source header for the canonical feature
        if display in CATEGORICAL_SELECTED:  # Verify if the selected feature is categorical in this reconstruction
            output[display] = chunk.loc[mask, source].astype("string").str.strip()  # Preserve stripped string encoding input values
        else:  # Handle numeric selected features
            output[display] = pd.to_numeric(chunk.loc[mask, source], errors="coerce")  # Preserve numeric conversion used by the supplied implementation
    output["Label"] = labels.loc[mask].astype("string")  # Append canonical labels after the 20 selected features
    return output.reset_index(drop=True)  # Return compact zero-based sampled rows before CSV persistence


def select_chunk_indices(valid_indices: np.ndarray, valid_labels: np.ndarray, remaining_population: Dict[str, int], remaining_need: Dict[str, int], selected_counts: Counter[str], rng: np.random.Generator) -> List[int]:
    """
    Select exact sampled source-row positions for one cleaned chunk using hypergeometric allocation.

    :param valid_indices: Absolute positional indices of cleaned target rows within the chunk.
    :param valid_labels: Canonical target labels aligned with valid_indices.
    :param remaining_population: Mutable class populations not yet passed in the stream.
    :param remaining_need: Mutable exact class quotas still required.
    :param selected_counts: Mutable cumulative selected-row counter.
    :param rng: NumPy random generator seeded for exact sampling.
    :return: Absolute positional indices selected from the current chunk.
    """

    chosen_absolute: List[int] = []  # Collect current-chunk positional rows selected across all target classes
    for class_name in PAPER_FIGURE6_CLASSES:  # Preserve the original deterministic target-class iteration order
        local_positions = np.flatnonzero(valid_labels == class_name)  # Locate cleaned current-chunk rows belonging to this class
        chunk_population = int(local_positions.size)  # Count current-chunk population for the class
        if chunk_population == 0:  # Verify if the class has no rows in this chunk
            continue  # Preserve RNG sequence by making no sampling calls for absent classes
        remaining_total = remaining_population[class_name]  # Read class population remaining before the current chunk
        remaining_quota = remaining_need[class_name]  # Read exact rows still required for this class
        if remaining_total < chunk_population or remaining_quota > remaining_total:  # Verify internal exact-sampling bookkeeping invariants
            raise RuntimeError(
                f"Sampling bookkeeping failure for {class_name}: N={remaining_total} m={chunk_population} K={remaining_quota}"
            )  # Preserve the original bookkeeping failure
        if remaining_quota == 0:  # Verify if this class quota has already been fully satisfied
            take_count = 0  # Select no additional rows without consuming RNG state
        elif remaining_quota == remaining_total:  # Verify if every remaining row must be selected to meet the exact quota
            take_count = chunk_population  # Select every class row in the current chunk without a hypergeometric draw
        else:  # Handle the normal conditional exact-sampling case
            take_count = int(rng.hypergeometric(chunk_population, remaining_total - chunk_population, remaining_quota))  # Preserve the exact hypergeometric draw and argument order
        if take_count > 0:  # Verify if at least one row from this class must be selected in the current chunk
            local_chosen = rng.choice(local_positions, size=take_count, replace=False)  # Preserve the original uniform without-replacement row choice
            chosen_absolute.extend(valid_indices[local_chosen].tolist())  # Convert class-local positions back to absolute chunk positions
            selected_counts[class_name] += take_count  # Accumulate selected rows for progress and reporting
            remaining_need[class_name] -= take_count  # Decrease the exact quota remaining for this class
        remaining_population[class_name] -= chunk_population  # Remove the complete observed chunk population from rows still unseen
    return chosen_absolute  # Return all selected absolute positions for output persistence


def write_selected_rows(gzip_handle: TextIO, chunk: pd.DataFrame, schema: FileSchema, labels: pd.Series, chosen_absolute: Sequence[int], wrote_header: bool) -> bool:
    """
    Write selected current-chunk rows to the compressed derived sample.

    :param gzip_handle: Open text-mode gzip destination handle.
    :param chunk: Current raw source chunk.
    :param schema: Exact schema mapping for the current source CSV.
    :param labels: Canonical labels aligned with the source chunk.
    :param chosen_absolute: Absolute positional indices selected within the chunk.
    :param wrote_header: Whether a CSV header has already been written to the destination.
    :return: Updated header-written state.
    """

    if not chosen_absolute:  # Verify if this chunk contributed no sampled rows
        return wrote_header  # Preserve header state without writing an empty frame
    ordered_positions = sorted(chosen_absolute)  # Preserve original source-row order among selected positions before persistence
    chosen_mask = pd.Series(False, index=chunk.index)  # Build a boolean mask aligned with the original chunk index
    chosen_mask.iloc[ordered_positions] = True  # Mark each sampled positional row for normalization and writing
    selected_labels = labels.copy()  # Preserve the original defensive label-series copy before normalization
    normalized = normalize_selected_chunk(chunk, schema, chosen_mask, selected_labels)  # Convert sampled rows to canonical top-20 columns plus Label
    normalized.to_csv(gzip_handle, index=False, header=not wrote_header)  # Append sampled rows and write the header only on the first non-empty chunk
    del normalized, chosen_mask  # Release selected chunk materialization before continuing the raw scan
    return True  # Record that the compressed sample now contains its header


def exact_sample_to_disk(schemas: Sequence[FileSchema], root: Path, chunksize: int, population_counts: Mapping[str, int], quotas: Mapping[str, int], seed: int, out_gz: Path) -> Dict[str, object]:
    """
    Stream an exact uniform without-replacement class sample to compressed local disk.

    :param schemas: Ordered validated source-file schemas.
    :param root: Raw dataset root used for relative paths.
    :param chunksize: Number of source rows read per pandas chunk.
    :param population_counts: Cleaned target populations measured by the first pass.
    :param quotas: Exact target rows requested for each Figure 6(c) class.
    :param seed: NumPy random seed for second-pass exact sampling.
    :param out_gz: Destination gzip CSV path for sampled top-20 rows.
    :return: Sampling-result dictionary with quotas, selected counts, path, and row total.
    """

    rng = np.random.default_rng(seed)  # Initialize the same NumPy generator from the configured data seed
    remaining_population = {class_name: int(population_counts[class_name]) for class_name in PAPER_FIGURE6_CLASSES}  # Track exact class rows still unseen
    remaining_need = {class_name: int(quotas[class_name]) for class_name in PAPER_FIGURE6_CLASSES}  # Track exact class rows still required
    selected_counts: Counter[str] = Counter()  # Accumulate selected rows per target class
    total_bytes = sum(schema.path.stat().st_size for schema in schemas)  # Sum source sizes for pass-two byte progress
    completed_bytes = 0  # Track bytes belonging to source files already completed
    eta = create_eta("PASS2-SAMPLE", total_bytes)  # Initialize the original pass-two ETA reporter
    out_gz.parent.mkdir(parents=True, exist_ok=True)  # Ensure the derived-dataset directory exists before opening gzip output
    with gzip.open(out_gz, "wt", encoding="utf-8", newline="") as gzip_handle:  # Create the compressed derived sample in text mode
        wrote_header = False  # Delay header writing until the first non-empty sampled chunk
        for file_index, schema in enumerate(schemas, 1):  # Stream every validated source file in deterministic order
            file_size = schema.path.stat().st_size  # Read current source size for bounded file-position progress
            print(f"[DATA][PASS2] file {file_index}/{len(schemas)}: {schema.path.relative_to(root)}")  # Preserve the original pass-two file message
            with schema.path.open("rb") as raw_handle:  # Open the raw source strictly for binary reading
                reader = pd.read_csv(raw_handle, usecols=read_usecols(schema), chunksize=chunksize, low_memory=False)  # Stream cleaning/label columns using the same pandas settings
                for chunk_index, chunk in enumerate(reader, 1):  # Process the current source file one bounded chunk at a time
                    labels = canonical_labels(chunk[schema.label_col])  # Canonicalize source labels before target filtering
                    valid = row_validity_mask(chunk, schema) & labels.isin(PAPER_FIGURE6_CLASSES)  # Restrict exact sampling to cleaned Figure 6(c) target rows
                    if valid.any():  # Verify if the current chunk contains at least one cleaned target row
                        valid_indices = np.flatnonzero(valid.to_numpy())  # Resolve positional target-row indices within the original chunk
                        valid_labels = labels.iloc[valid_indices].to_numpy(dtype=object)  # Materialize canonical target labels aligned to valid indices
                        chosen = select_chunk_indices(valid_indices, valid_labels, remaining_population, remaining_need, selected_counts, rng)  # Perform exact conditional class sampling in original RNG order
                        wrote_header = write_selected_rows(gzip_handle, chunk, schema, labels, chosen, wrote_header)  # Append sampled canonical rows to the compressed derived dataset
                    if chunk_index % 5 == 0:  # Verify if the original five-chunk progress interval was reached
                        position = min(raw_handle.tell(), file_size)  # Bound buffered source position to the current file size
                        eta.report(
                            completed_bytes + position,
                            f"file={schema.path.name} chunk={chunk_index} selected={sum(selected_counts.values()):,}",
                        )  # Preserve the original pass-two progress detail
                    del chunk, labels, valid  # Release chunk-local source data before the next iteration
                    gc.collect()  # Preserve explicit garbage collection for bounded memory usage
            completed_bytes += file_size  # Advance completed-byte accounting after finishing the current file
            eta.report(completed_bytes, f"completed {schema.path.name}", force=True)  # Preserve forced end-of-file progress output
    unmet = {class_name: count for class_name, count in remaining_need.items() if count != 0}  # Identify any class quota not satisfied by the complete second pass
    if unmet:  # Verify if exact class quotas were not fully met
        raise RuntimeError(f"Exact sampling did not satisfy quotas: {unmet}")  # Preserve the original exact-sampling failure
    return {
        "quotas": dict(quotas),
        "selected_counts": dict(selected_counts),
        "sample_file": str(out_gz),
        "sample_rows": int(sum(selected_counts.values())),
    }  # Return the same second-pass report fields as the supplied implementation
