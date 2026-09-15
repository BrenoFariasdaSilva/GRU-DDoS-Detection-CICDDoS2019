"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 PROGRESS, ETA, AND RESOURCE REPORTING
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Implements duration and byte formatting, generic ETA reporting, and Keras epoch-level
    timing/resource reporting used throughout the GRU CICDDoS2019 reproduction pipeline.

    Key features include:
        - Formats elapsed and estimated durations with the original display convention.
        - Reports long-stage progress at the original five-second minimum interval.
        - Reports training epoch metrics, ETA, process RSS, and system memory usage.

Usage:
    1. Use format_seconds() and human_bytes() for progress messages.
    2. Create stage ETA state with create_eta().
    3. Create the Keras callback with create_epoch_eta().

Outputs:
    - Progress and resource messages written to standard output.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - psutil.
    - tensorflow.
    - Python standard library.

Assumptions & Notes:
    - ETA values are estimates derived from observed progress rates and completed epoch times.
================================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
from .tensorflow_runtime import tf


def format_seconds(seconds: Optional[float]) -> str:
    """
    Format seconds using the compact duration convention from the supplied implementation.

    :param seconds: Duration in seconds or None when unavailable.
    :return: Human-readable duration string.
    """

    if seconds is None or not np.isfinite(seconds) or seconds < 0:  # Verify if the duration is unavailable or invalid
        return "unavailable"  # Preserve the original unavailable marker
    rounded_seconds = int(round(seconds))  # Round to whole seconds as in the original implementation
    days, remainder = divmod(rounded_seconds, 86400)  # Split complete days from remaining seconds
    hours, remainder = divmod(remainder, 3600)  # Split complete hours from remaining seconds
    minutes, secs = divmod(remainder, 60)  # Split complete minutes from remaining seconds
    parts: List[str] = []  # Build the compact duration from largest to smallest unit
    if days:  # Verify if at least one complete day is present
        parts.append(f"{days}d")  # Preserve day formatting without zero padding
    if hours or days:  # Verify if hours should be shown directly or because days are present
        parts.append(f"{hours}h")  # Preserve hour formatting without zero padding
    if minutes or hours or days:  # Verify if minutes should be included in the compact representation
        parts.append(f"{minutes}m")  # Preserve minute formatting without zero padding
    parts.append(f"{secs}s")  # Always include seconds in the final representation
    return " ".join(parts)  # Return the original space-separated duration format


def human_bytes(byte_count: float) -> str:
    """
    Format a byte count using binary units from bytes through tebibytes.

    :param byte_count: Number of bytes to format.
    :return: Human-readable binary byte-size string.
    """

    units = ("B", "KiB", "MiB", "GiB", "TiB")  # Preserve the original ordered binary unit list
    value = float(byte_count)  # Convert input to floating point for repeated division
    for unit in units:  # Test each unit from bytes through tebibytes
        if abs(value) < 1024.0 or unit == units[-1]:  # Verify if the current unit is suitable or the final unit was reached
            return f"{value:.2f} {unit}"  # Return the original two-decimal size format
        value /= 1024.0  # Convert the value to the next binary unit
    return f"{value:.2f} TiB"  # Preserve the defensive final fallback from the supplied implementation


@dataclass
class ETA:
    """Store generic progress-reporting state for one long-running stage."""

    label: str
    total: float
    started: float
    last_print: float

    def report(self: "ETA", done: float, detail: str = "", force: bool = False) -> None:
        """
        Report stage progress and ETA using the supplied implementation's formula.

        :param self: Current ETA state object.
        :param done: Amount of work completed so far.
        :param detail: Optional detail suffix appended to the progress message.
        :param force: Whether to bypass the five-second output throttle.
        :return: None.
        """

        now = time.time()  # Capture one timestamp for all progress calculations
        if not force and now - self.last_print < 5.0:  # Verify if a non-forced report occurs before the original five-second interval
            return  # Skip the report to preserve the original output cadence
        self.last_print = now  # Record the current reporting timestamp
        bounded_done = max(0.0, min(float(done), self.total)) if self.total > 0 else float(done)  # Clamp progress only when a positive total exists
        elapsed = max(now - self.started, 1e-9)  # Protect the rate calculation from a zero elapsed duration
        if self.total > 0 and bounded_done > 0:  # Verify if enough progress exists to calculate a remaining ETA
            fraction = bounded_done / self.total  # Calculate completed fraction of total work
            eta = elapsed * (1.0 - fraction) / fraction  # Estimate remaining duration using the original rate formula
            percent = 100.0 * fraction  # Convert completed fraction to percentage
            speed = bounded_done / elapsed  # Calculate observed work units per second
            message = (
                f"[ETA][{self.label}] {percent:6.2f}% | elapsed={format_seconds(elapsed)} "
                f"| ETA={format_seconds(eta)} | rate={speed:,.2f}/s"
            )  # Preserve the original progress field order and formatting
        else:  # Handle stages that do not yet have measurable positive progress
            message = f"[ETA][{self.label}] elapsed={format_seconds(elapsed)} | ETA=unavailable"  # Preserve the original unavailable-ETA form
        if detail:  # Verify if the caller provided an additional progress detail
            message += f" | {detail}"  # Append the detail with the original separator
        print(message, flush=True)  # Emit the progress line immediately


def create_eta(label: str, total: float) -> ETA:
    """
    Create initialized generic ETA state without defining a custom underscore-prefixed initializer.

    :param label: Stage label displayed in ETA messages.
    :param total: Total expected work units for the stage.
    :return: Initialized ETA state object.
    """

    return ETA(label=label, total=float(total), started=time.time(), last_print=0.0)  # Preserve original initial state and start timestamp


class EpochETA(tf.keras.callbacks.Callback):
    """Report per-epoch training ETA, metrics, and process/system memory usage."""

    total_epochs: int
    started: float
    epoch_times: List[float]
    epoch_started: float
    process: psutil.Process

    def on_train_begin(self: "EpochETA", logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the start time of model training.

        :param self: Current EpochETA callback instance.
        :param logs: Optional Keras training log dictionary.
        :return: None.
        """

        self.started = time.time()  # Preserve training-start timing at the Keras callback boundary

    def on_epoch_begin(self: "EpochETA", epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the start time of one training epoch.

        :param self: Current EpochETA callback instance.
        :param epoch: Zero-based Keras epoch index.
        :param logs: Optional Keras training log dictionary.
        :return: None.
        """

        self.epoch_started = time.time()  # Start per-epoch wall-clock measurement
