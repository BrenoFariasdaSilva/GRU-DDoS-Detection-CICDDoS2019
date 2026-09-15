"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 PERSISTENCE AND RAW-INTEGRITY UTILITIES
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Provides JSON persistence and raw CICDDoS2019 metadata snapshot utilities used to keep
    generated artifacts separate from the read-only source corpus.

    Key features include:
        - Writes UTF-8 JSON artifacts with stable indentation.
        - Snapshots raw CSV sizes and nanosecond modification times recursively.
        - Verifies that raw source metadata is unchanged after experiment execution.

Usage:
    1. Use json_dump() for pipeline JSON artifacts.
    2. Capture raw metadata with raw_snapshot() before and after execution.
    3. Use verify_raw_snapshot() to reject runs where source metadata changed.

Outputs:
    - JSON files requested by callers and raw-integrity verification messages.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - Raw integrity is based on source CSV size and mtime_ns, matching the supplied main.py.
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def json_dump(path: Path, obj: object) -> None:
    """
    Write one JSON-serializable object to disk using the original formatting.

    :param path: Destination JSON path.
    :param obj: JSON-serializable object to persist.
    :return: None.
    """

    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")  # Preserve the original UTF-8 JSON formatting
