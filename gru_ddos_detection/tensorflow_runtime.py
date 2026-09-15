"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 TENSORFLOW RUNTIME IMPORT
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Centralizes TensorFlow import behavior for the modular GRU CICDDoS2019 reproduction so every
    TensorFlow-dependent module retains the supplied startup log setting and import error.

    Key features include:
        - Sets TF_CPP_MIN_LOG_LEVEL before TensorFlow import.
        - Imports TensorFlow once through the package runtime namespace.
        - Raises a platform-aware RuntimeError when TensorFlow import fails.

Usage:
    1. Import tf from gru_ddos_detection.tensorflow_runtime in TensorFlow-dependent modules.
    2. Install the pinned requirements.txt before running the project.
    3. Execute through the top-level main.py entry point.

Outputs:
    - None directly produced during successful import.

TODOs:
    - None identified.

Dependencies:
    - tensorflow.
    - Python standard library.

Assumptions & Notes:
    - TensorFlow startup verbosity and import-failure wording are preserved from the supplied main.py.
================================================================================
"""

from __future__ import annotations

import os


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
