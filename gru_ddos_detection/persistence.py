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


def raw_snapshot(root: Path) -> Dict[str, Dict[str, int]]:
    """
    Snapshot size and nanosecond modification time for every raw CSV under a root.

    :param root: Raw CICDDoS2019 root directory.
    :return: Mapping from relative CSV path to size and mtime_ns metadata.
    """

    snapshot: Dict[str, Dict[str, int]] = {}  # Collect metadata keyed by paths relative to the raw dataset root
    for path in sorted(root.rglob("*.csv")):  # Traverse every raw CSV in deterministic path order
        if path.is_file():  # Verify if the discovered path is a regular file
            stat = path.stat()  # Read filesystem metadata without opening the CSV for writing
            snapshot[str(path.relative_to(root))] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}  # Preserve the original integrity fields
    return snapshot  # Return the complete raw-source metadata snapshot


def verify_raw_snapshot(before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]) -> None:
    """
    Verify that raw CSV metadata is identical before and after execution.

    :param before: Raw-source metadata captured before pipeline execution.
    :param after: Raw-source metadata captured after pipeline execution.
    :return: None.
    """

    if before != after:  # Verify if any raw CSV size, mtime, addition, or removal changed during execution
        raise RuntimeError(
            "RAW DATASET METADATA CHANGED DURING EXECUTION. The script itself only opens raw CSVs read-only; "
            "investigate another process before trusting results."
        )  # Preserve the original integrity failure message
    print("[RAW] Verified: source CSV sizes and mtimes are unchanged.")  # Preserve the original successful integrity message
