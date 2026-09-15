"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 EXPERIMENT CONFIGURATION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines the immutable runtime configuration used by the modular GRU CICDDoS2019
    reproduction pipeline.

    Key features include:
        - Stores dataset and output paths.
        - Stores sampling, split, scaling, and validation settings.
        - Stores GRU architecture and training hyperparameters.

Usage:
    1. Build Config from validated command-line arguments in gru_ddos_detection.workflow.
    2. Pass the immutable configuration to preprocessing, model, and experiment modules.
    3. Serialize with dataclasses.asdict when persisting config.json.

Outputs:
    - None directly produced.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - Field names and values preserve the supplied main.py configuration contract.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    """Store immutable settings for one complete GRU CICDDoS2019 reproduction invocation."""

    data_dir: str
    output_dir: str
    source_day: str
    chunksize: int
    sampling_profile: str
    per_class_cap: int
    data_seed: int
    split_seed: int
    model_seed: int
    runs: int
    epochs: int
    batch_size: int
    learning_rate: float
    gru_units: int
    dense1: int
    dense2: int
    dropout: float
    early_patience: int
    early_min_delta: float
    scaling_mode: str
    validation_mode: str
    stratify: bool
    allow_cpu: bool
    reuse_cache: bool
