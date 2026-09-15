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


def configure_accelerator(allow_cpu: bool) -> str:
    """
    Require and verify the TensorFlow GPU unless CPU execution was explicitly allowed.

    :param allow_cpu: Whether execution may continue without a detected GPU.
    :return: TensorFlow device string selected for model execution.
    """

    print("[SYSTEM] platform:", platform.platform())  # Preserve platform diagnostics
    print("[SYSTEM] machine:", platform.machine())  # Preserve machine-architecture diagnostics
    print("[SYSTEM] Python:", sys.version.split()[0])  # Preserve concise Python version diagnostics
    print("[SYSTEM] TensorFlow:", tf.__version__)  # Preserve TensorFlow version diagnostics
    print("[SYSTEM] tensorflow-metal:", package_version("tensorflow-metal"))  # Preserve Metal plugin version diagnostics
    tf.config.set_soft_device_placement(True)  # Preserve TensorFlow soft device placement behavior
    gpus = tf.config.list_physical_devices("GPU")  # Discover TensorFlow-visible physical GPU devices
    print("[SYSTEM] TensorFlow GPU devices:", gpus)  # Preserve GPU discovery output
    if not gpus:  # Verify if TensorFlow failed to expose a GPU
        if not allow_cpu:  # Verify if the caller did not explicitly allow CPU fallback
            raise RuntimeError(
                "No TensorFlow GPU detected. On Apple Silicon verify tensorflow-metal; on Linux "
                "verify the NVIDIA driver and TensorFlow CUDA dependencies, or pass --allow-cpu intentionally."
            )  # Preserve the original accelerator failure message
        print("[SYSTEM] WARNING: CPU execution explicitly allowed.")  # Preserve the original CPU fallback warning
        return "/CPU:0"  # Preserve the original early return when CPU fallback is used
    with tf.device("/GPU:0"):  # Force the smoke-test operations onto the first visible GPU
        left = tf.random.uniform((256, 256), dtype=tf.float32)  # Create the first smoke-test matrix
        right = tf.random.uniform((256, 256), dtype=tf.float32)  # Create the second smoke-test matrix
        product = tf.linalg.matmul(left, right)  # Execute a real matrix multiplication on the requested device
        checksum = float(tf.reduce_sum(product).numpy())  # Materialize a checksum so the operation actually executes
    print(f"[SYSTEM] GPU smoke test: device={product.device} checksum={checksum:.3f}")  # Preserve the original smoke-test output
    if "GPU" not in product.device.upper():  # Verify if the operation was actually placed on a GPU
        raise RuntimeError(f"GPU listed, but smoke test executed on {product.device}")  # Preserve the original placement failure
    tf.keras.mixed_precision.set_global_policy("float32")  # Preserve explicit float32 policy on the GPU path
    return "/GPU:0"  # Return the verified GPU device


def set_seeds(model_seed: int) -> None:
    """
    Apply the supplied model seed to Python, NumPy, and TensorFlow.

    :param model_seed: Integer seed for the current model run.
    :return: None.
    """

    os.environ["PYTHONHASHSEED"] = str(model_seed)  # Preserve the original Python hash-seed assignment
    random.seed(model_seed)  # Seed the Python pseudo-random generator
    np.random.seed(model_seed)  # Seed NumPy's legacy global pseudo-random generator
    tf.keras.utils.set_random_seed(model_seed)  # Seed TensorFlow/Keras reproducibility utilities
