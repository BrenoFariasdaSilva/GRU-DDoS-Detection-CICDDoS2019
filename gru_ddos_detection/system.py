"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 SYSTEM, ACCELERATOR, AND REPRODUCIBILITY SETUP
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Configures TensorFlow execution, verifies GPU availability, applies random seeds, and
    reports environment metadata for macOS Apple Silicon and Linux reproduction runs.

    Key features include:
        - Reports platform, Python, TensorFlow, and optional tensorflow-metal versions.
        - Performs the original GPU matrix-multiplication smoke test.
        - Applies Python, NumPy, and TensorFlow seeds for each experiment run.

Usage:
    1. Call configure_accelerator() before model construction.
    2. Call environment_info() after selecting the device.
    3. Call set_seeds() at the beginning of each experiment run.

Outputs:
    - System diagnostics written to standard output and an environment metadata dictionary.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - psutil.
    - tensorflow.
    - Python standard library.

Assumptions & Notes:
    - CPU execution is rejected unless --allow-cpu is explicitly supplied.
================================================================================
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psutil
from .tensorflow_runtime import tf


def package_version(name: str) -> Optional[str]:
    """
    Return an installed package version when available.

    :param name: Distribution package name understood by importlib.metadata.
    :return: Installed version string or None when the package is absent.
    """

    try:  # Attempt to resolve the installed package distribution metadata
        return importlib.metadata.version(name)  # Return the exact installed distribution version
    except importlib.metadata.PackageNotFoundError:  # Handle optional packages that are not installed
        return None  # Preserve the original unavailable-version behavior
