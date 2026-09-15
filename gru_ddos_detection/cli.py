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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the GRU CICDDoS2019 reproduction pipeline.

    :param argv: Optional explicit argument sequence; uses process arguments when None.
    :return: Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="GRU DDoS Detection on CICDDoS2019: independent reproduction of Ramzan et al. (2023) for macOS Apple Silicon and Linux.")  # Describe the method-centric project and source paper
    parser.add_argument("--data-dir", type=Path, required=True)  # Preserve the required raw dataset path option
    parser.add_argument("--output-dir", type=Path, default=Path("GRU-DDoS-Detection"))  # Use the method-centric default output directory
    parser.add_argument("--source-day", choices=("01-12", "03-11", "both"), default="01-12")  # Preserve the original source-day choices and default
    parser.add_argument("--chunksize", type=int, default=50_000)  # Preserve the original streaming chunk size
    parser.add_argument(
        "--sampling-profile",
        choices=("figure6-inferred", "cap-per-class", "all"),
        default="figure6-inferred",
        help="Default infers source class counts from Figure 6(c) test supports / 0.30.",
    )  # Preserve the original sampling-profile contract
    parser.add_argument("--per-class-cap", type=int, default=100_000)  # Preserve the cap-per-class fallback limit
    parser.add_argument("--data-seed", type=int, default=42)  # Preserve the raw-sampling random seed
    parser.add_argument("--split-seed", type=int, default=42)  # Preserve the train/test split seed
    parser.add_argument("--model-seed", type=int, default=1337)  # Preserve the neural-network seed
    parser.add_argument("--runs", type=int, default=1, help="Paper does not report repeated independent runs; default 1.")  # Preserve the original run-count default
    parser.add_argument("--epochs", type=int, default=100)  # Preserve the paper-aligned maximum epoch count
    parser.add_argument("--batch-size", type=int, default=1000)  # Preserve the paper-aligned batch size
    parser.add_argument("--learning-rate", type=float, default=0.001)  # Preserve the paper-aligned Adam learning rate
    parser.add_argument("--gru-units", type=int, default=8)  # Preserve the paper-aligned recurrent width
    parser.add_argument("--dense1", type=int, default=16)  # Preserve the first hidden-layer width
    parser.add_argument("--dense2", type=int, default=8)  # Preserve the second hidden-layer width
    parser.add_argument("--dropout", type=float, default=0.10, help="Paper says dropout is used but omits rate; 0.10 is an external reconstruction clue.")  # Preserve the inferred dropout default
    parser.add_argument("--early-patience", type=int, default=5)  # Preserve the published early-stopping patience
    parser.add_argument("--early-min-delta", type=float, default=0.001)  # Preserve the published early-stopping minimum delta
    parser.add_argument(
        "--scaling-mode",
        choices=("separate", "train-only"),
        default="separate",
        help="'separate' fits a StandardScaler separately to train and test (high-fidelity clue but leaky); 'train-only' is rigorous.",
    )  # Preserve the original high-fidelity and rigorous scaler modes
    parser.add_argument(
        "--validation-mode",
        choices=("test", "holdout"),
        default="test",
        help="Paper has only 70/30 train/test yet reports validation curves. 'test' is high-fidelity but leaks test information.",
    )  # Preserve the original validation-mode choices
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratification. Paper does not state stratification.")  # Preserve the original stratification switch
    parser.add_argument("--allow-cpu", action="store_true")  # Preserve explicit CPU fallback behavior
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse encoded_sample_X.npy/y.npy in output-dir and skip the raw 2-pass scan.")  # Preserve explicit encoded-cache reuse
    return parser.parse_args(argv)  # Return parsed options without starting pipeline work


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
