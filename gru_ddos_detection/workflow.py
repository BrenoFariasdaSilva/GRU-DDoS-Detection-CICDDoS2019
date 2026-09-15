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
