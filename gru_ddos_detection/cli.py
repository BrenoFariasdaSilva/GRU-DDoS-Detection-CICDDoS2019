"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 COMMAND-LINE INTERFACE
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines and validates the command-line interface for the GRU CICDDoS2019
    reproduction while preserving the supplied main.py option names, defaults, and checks.

    Key features include:
        - Parses source-day, sampling, split, scaling, validation, and model options.
        - Validates positive numeric arguments and dropout bounds.
        - Constrains generated output to the directory containing the top-level main.py.

Usage:
    1. Call parse_args() from the top-level orchestrator.
    2. Call validate_args() before building Config.
    3. Use the normalized data_dir and output_dir values stored back into the namespace.

Outputs:
    - Parsed and validated command-line configuration values.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.
    - gru_ddos_detection.constants.

Assumptions & Notes:
    - Relative output paths resolve from the current working directory and must remain inside the top-level main.py directory.
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .constants import PROJECT_ROOT

# Functions Definitions:


def enforce_output_inside_script_dir(output_dir: Path) -> Path:
    """
    Resolve and constrain generated output to the top-level main.py directory.

    :param output_dir: User-provided output directory path.
    :return: Resolved output directory inside the project root.
    """

    output = output_dir.resolve()  # Preserve the original current-working-directory resolution before project-root containment validation
    try:  # Verify if the requested output remains inside the project root
        output.relative_to(PROJECT_ROOT)  # Preserve the original containment requirement using the modular project root
    except ValueError as exc:  # Handle output paths that escape the project root
        raise ValueError(
            f"--output-dir must be inside the directory containing main.py ({PROJECT_ROOT}). "
            f"Requested: {output}"
        ) from exc  # Preserve the original error semantics while referencing the correct top-level main.py directory
    output.mkdir(parents=True, exist_ok=True)  # Preserve automatic output-directory creation
    return output  # Return the resolved output directory


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate and normalize command-line arguments exactly as the supplied implementation.

    :param args: Parsed command-line namespace to validate and normalize in place.
    :return: None.
    """

    if not args.data_dir.exists():  # Verify if the raw dataset path exists before execution
        raise FileNotFoundError(args.data_dir)  # Preserve the original missing-data error
    for name in ("chunksize", "per_class_cap", "runs", "epochs", "batch_size", "gru_units", "dense1", "dense2"):  # Validate the same strictly positive options as the original script
        if getattr(args, name) < 1:  # Verify if the current positive-only argument is invalid
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")  # Preserve the original argument-specific validation message
    if not (0.0 <= args.dropout < 1.0):  # Verify if dropout falls outside the original valid interval
        raise ValueError("--dropout must be in [0,1)")  # Preserve the original dropout validation message
    args.data_dir = args.data_dir.resolve()  # Preserve resolution of the source dataset path before workflow execution
    args.output_dir = enforce_output_inside_script_dir(args.output_dir)  # Resolve and create the project-local output directory
