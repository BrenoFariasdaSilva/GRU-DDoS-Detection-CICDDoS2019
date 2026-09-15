"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 COMPLETE REPRODUCTION WORKFLOW
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Coordinates configuration persistence, paper audit metadata, raw-source integrity,
    accelerator setup, two-pass sampling/cache preparation, repeated GRU runs, and aggregation.

    Key features include:
        - Preserves the supplied main.py execution order and artifact names.
        - Reuses encoded_sample_X.npy/y.npy only when --reuse-cache is explicitly requested.
        - Verifies the complete raw CSV metadata snapshot after all configured runs finish.

Usage:
    1. Parse and validate arguments through gru_ddos_detection.cli.
    2. Pass the validated namespace to run_workflow().
    3. Inspect all generated artifacts below the configured project-local output directory.

Outputs:
    - Configuration, paper audit, assumptions, environment, raw snapshots, sampling/encoding
      caches, per-run artifacts, runs_summary.csv, and aggregate_metrics.json.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - Python standard library.
    - gru_ddos_detection audit, config, constants, encoding, experiment, persistence, sampling,
      schema, system, and timing modules.

Assumptions & Notes:
    - The output path has already been normalized and constrained by gru_ddos_detection.cli.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .audit import build_reconstruction_assumptions, paper_matrix_audit
from .config import Config
from .constants import (
    FIGURE6_INFERRED_CLASS_QUOTAS,
    PAPER_FIGURE6_CLASSES,
    PAPER_FIGURE6_TEST_SUPPORT,
    PAPER_TARGET_ACCURACY,
    PAPER_TARGET_F1,
    PAPER_TOP20,
)
from .encoding import encode_sample_to_npy, fit_sample_encoders
from .experiment import run_experiment
from .persistence import json_dump, raw_snapshot, verify_raw_snapshot
from .sampling import count_valid_rows, determine_quotas, exact_sample_to_disk
from .schema import find_source_csvs, inspect_schemas
from .system import configure_accelerator, environment_info
from .timing import format_seconds


def build_config(args: argparse.Namespace) -> Config:
    """
    Build the immutable Config from validated command-line arguments.

    :param args: Validated and normalized command-line namespace.
    :return: Immutable experiment configuration preserving the original field mapping.
    """

    return Config(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        source_day=args.source_day,
        chunksize=args.chunksize,
        sampling_profile=args.sampling_profile,
        per_class_cap=args.per_class_cap,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        runs=args.runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gru_units=args.gru_units,
        dense1=args.dense1,
        dense2=args.dense2,
        dropout=args.dropout,
        early_patience=args.early_patience,
        early_min_delta=args.early_min_delta,
        scaling_mode=args.scaling_mode,
        validation_mode=args.validation_mode,
        stratify=not args.no_stratify,
        allow_cpu=args.allow_cpu,
        reuse_cache=args.reuse_cache,
    )  # Preserve the original command-line-to-configuration field mapping exactly


def persist_paper_metadata(cfg: Config, output_dir: Path) -> None:
    """
    Persist paper audit, reconstruction assumptions, and published selected features.

    :param cfg: Immutable resolved experiment configuration.
    :param output_dir: Root generated-output directory.
    :return: None.
    """

    audit = paper_matrix_audit()  # Recompute the supplied Figure 6(c) audit before training
    json_dump(output_dir / "paper_internal_consistency_audit.json", audit)  # Preserve the original audit artifact filename
    print("[PAPER] Table 4 target: accuracy=0.9954, F1=0.9800")  # Preserve the original Table 4 target message
    print(
        "[PAPER] Figure 6(c) audit: "
        f"test_rows={audit['figure6c_total_test_rows']:,}, "
        f"accuracy_from_cells={audit['figure6c_accuracy_computed_from_cells']:.6f}, "
        f"macro_F1={audit['figure6c_macro_f1_computed_from_cells']:.6f}, "
        f"weighted_F1={audit['figure6c_weighted_f1_computed_from_cells']:.6f}"
    )  # Preserve the original concise audit message
    print(
        "[PAPER] Figure 6(c) is NOT mathematically consistent with Table 4: "
        f"trace={audit['figure6c_correct_predictions_trace']:,}/"
        f"{audit['figure6c_total_test_rows']:,}, "
        f"matrix_accuracy={audit['figure6c_accuracy_computed_from_cells']:.8f}, "
        f"table4_accuracy={PAPER_TARGET_ACCURACY:.4f}, "
        f"matrix_macro_F1={audit['figure6c_macro_f1_computed_from_cells']:.8f}, "
        f"table4_F1={PAPER_TARGET_F1:.4f}."
    )  # Report the computed matrix values and published Table 4 targets without conflating them
    json_dump(output_dir / "reconstruction_assumptions.json", build_reconstruction_assumptions(cfg))  # Persist the original assumptions/missing-information structure
    json_dump(output_dir / "published_top20_features.json", list(PAPER_TOP20))  # Preserve the published selected-feature artifact


def discover_and_sample(args: argparse.Namespace, sample_gz: Path, sample_report_path: Path) -> Tuple[int, Dict[str, object]]:
    """
    Discover source files, count valid rows, determine quotas, and write the exact disk sample.

    :param args: Validated command-line namespace controlling source discovery and sampling.
    :param sample_gz: Destination compressed selected-feature sample path.
    :param sample_report_path: Destination sampling_report.json path.
    :return: Exact sampled row count and completed sampling report dictionary.
    """

    source_files = find_source_csvs(args.data_dir, args.source_day)  # Discover the configured CICDDoS2019 source-day CSV files
    print(f"[DATA] Source day={args.source_day}; discovered {len(source_files)} CSV files.")  # Preserve source discovery summary
    for path in source_files:  # Report each discovered source path in deterministic order
        print("       ", path.relative_to(args.data_dir))  # Preserve the original indented relative-path output
    (args.output_dir / "source_csv_files.txt").write_text(
        "\n".join(str(path.relative_to(args.data_dir)) for path in source_files), encoding="utf-8"
    )  # Preserve the original source inventory filename and contents
    schemas = inspect_schemas(source_files)  # Validate labels and published top-20 feature availability in every source file
    counts, count_report = count_valid_rows(schemas, args.data_dir, args.chunksize)  # Execute the first pass over cleaned source rows
    quotas = determine_quotas(counts, args.sampling_profile, args.per_class_cap)  # Resolve exact class quotas from the configured sampling profile
    print("[DATA] Valid target rows:", dict(counts))  # Preserve cleaned source population diagnostics
    print("[DATA] Sampling quotas:", quotas)  # Preserve actual class-quota diagnostics
    print(f"[DATA] Total local derived sample rows: {sum(quotas.values()):,}")  # Preserve derived sample-size diagnostics
    sample_report: Dict[str, object] = {
        "count_pass": count_report,
        "sampling_profile": args.sampling_profile,
        "figure6_test_support": PAPER_FIGURE6_TEST_SUPPORT,
        "figure6_inferred_70plus30_source_quotas": FIGURE6_INFERRED_CLASS_QUOTAS,
        "actual_quotas": quotas,
    }  # Build the original pre-second-pass sampling report fields
    sample_report.update(exact_sample_to_disk(
        schemas,
        args.data_dir,
        args.chunksize,
        counts,
        quotas,
        args.data_seed,
        sample_gz,
    ))  # Execute exact second-pass hypergeometric sampling and merge its report fields
    json_dump(sample_report_path, sample_report)  # Persist the complete sampling report after successful exact sampling
    return int(sum(quotas.values())), sample_report  # Return exact row count for encoder-array allocation plus the completed report


def build_encoded_cache(args: argparse.Namespace, x_cache: Path, y_cache: Path, sample_gz: Path, sample_report_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the sampled compressed dataset, fit encoders, and create encoded NumPy caches.

    :param args: Validated command-line namespace controlling dataset preparation.
    :param x_cache: Destination encoded_sample_X.npy path.
    :param y_cache: Destination encoded_sample_y.npy path.
    :param sample_gz: Destination sampled_selected_top20.csv.gz path.
    :param sample_report_path: Destination sampling_report.json path.
    :return: Read-only memory-mapped encoded feature and label arrays.
    """

    total_rows, _ = discover_and_sample(args, sample_gz, sample_report_path)  # Build exact compressed sampled selected-feature dataset first
    encoder_dir = args.output_dir / "encoders"  # Resolve the original encoder artifact directory
    encoder_dir.mkdir(parents=True, exist_ok=True)  # Ensure the encoder directory exists before joblib persistence
    feature_encoders, label_encoder = fit_sample_encoders(sample_gz, args.chunksize, total_rows, encoder_dir)  # Fit and persist categorical and output encoders
    X, y = encode_sample_to_npy(
        sample_gz,
        args.chunksize,
        total_rows,
        feature_encoders,
        label_encoder,
        x_cache,
        y_cache,
    )  # Stream the sampled dataset into the original disk-backed encoded cache files
    json_dump(encoder_dir / "output_class_order.json", label_encoder.classes_.tolist())  # Preserve encoded canonical class-order artifact
    return X, y  # Return memory-mapped encoded arrays for validation and repeated runs


def load_or_build_encoded_cache(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reuse complete encoded caches when explicitly requested, otherwise rebuild them.

    :param args: Validated command-line namespace controlling cache reuse and dataset preparation.
    :return: Read-only memory-mapped or freshly generated encoded feature and label arrays.
    """

    x_cache = args.output_dir / "encoded_sample_X.npy"  # Resolve the original encoded feature-cache path
    y_cache = args.output_dir / "encoded_sample_y.npy"  # Resolve the original encoded label-cache path
    sample_gz = args.output_dir / "derived_dataset" / "sampled_selected_top20.csv.gz"  # Resolve the original compressed sampled-data path
    sample_report_path = args.output_dir / "sampling_report.json"  # Resolve the original sampling report path
    if args.reuse_cache and x_cache.exists() and y_cache.exists():  # Verify if explicit cache reuse was requested and both required arrays exist
        print("[DATA] Reusing local encoded sample cache; raw dataset will not be rescanned.")  # Preserve the original cache-reuse message
        return np.load(x_cache, mmap_mode="r"), np.load(y_cache, mmap_mode="r")  # Reopen existing arrays as read-only memory maps without raw rescanning
    return build_encoded_cache(args, x_cache, y_cache, sample_gz, sample_report_path)  # Rebuild sampling, encoders, and encoded arrays when reuse is unavailable
