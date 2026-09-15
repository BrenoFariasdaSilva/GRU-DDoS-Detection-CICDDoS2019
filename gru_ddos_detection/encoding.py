"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 SAMPLE ENCODING AND NUMPY CACHE CREATION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Fits categorical LabelEncoder objects on the compressed sampled dataset and encodes the
    published top-20 features plus output labels into memory-mappable NumPy arrays.

    Key features include:
        - Collects complete sampled vocabularies for Timestamp, Flow ID, and output labels.
        - Persists feature and output LabelEncoder objects with joblib.
        - Streams encoded feature/label blocks into encoded_sample_X.npy and encoded_sample_y.npy.

Usage:
    1. Call fit_sample_encoders() after sampled_selected_top20.csv.gz exists.
    2. Call encode_sample_to_npy() with the fitted encoders.
    3. Reuse the resulting .npy files with --reuse-cache on later executions.

Outputs:
    - LabelEncoder joblib files and memory-mappable encoded NumPy sample arrays.

TODOs:
    - None identified.

Dependencies:
    - joblib.
    - numpy.
    - pandas.
    - scikit-learn.
    - Python standard library.
    - gru_ddos_detection.constants, schema, and timing.

Assumptions & Notes:
    - Categorical vocabulary collection and class-order validation preserve the supplied main.py.
================================================================================
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Mapping, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .constants import CATEGORICAL_SELECTED, PAPER_FIGURE6_CLASSES, PAPER_TOP20
from .schema import normalize_column
from .timing import create_eta


def collect_vocabularies(sample_gz: Path, chunksize: int, total_rows: int) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    Collect complete sampled categorical feature and output-label vocabularies.

    :param sample_gz: Compressed sampled selected-feature CSV path.
    :param chunksize: Number of sampled rows read per pandas chunk.
    :param total_rows: Expected total sampled row count for progress reporting.
    :return: Categorical feature vocabularies and output-label vocabulary.
    """

    unique: Dict[str, Set[str]] = {column: set() for column in CATEGORICAL_SELECTED}  # Initialize one vocabulary set per categorical selected feature
    unique_labels: Set[str] = set()  # Collect every canonical output label present in the sampled dataset
    eta = create_eta("ENCODER-FIT", total_rows)  # Initialize the original encoder-fit row-progress reporter
    done = 0  # Track sampled rows processed during vocabulary collection
    for chunk in pd.read_csv(sample_gz, chunksize=chunksize, low_memory=False):  # Stream the compressed derived sample without loading it all at once
        for column in CATEGORICAL_SELECTED:  # Collect categorical values for each selected string feature
            unique[column].update(chunk[column].astype("string").fillna("").tolist())  # Preserve the original string conversion and missing-value fallback
        unique_labels.update(chunk["Label"].astype("string").tolist())  # Collect canonical output labels from this sampled chunk
        done += len(chunk)  # Advance encoder-fit progress by rows consumed
        eta.report(done, f"unique_flow_ids={len(unique['Flow ID']):,} unique_timestamps={len(unique['Timestamp']):,}")  # Preserve the original vocabulary progress detail
        del chunk  # Release sampled chunk memory before reading the next block
        gc.collect()  # Preserve explicit garbage collection from the supplied implementation
    eta.report(total_rows, "categorical vocabularies complete", force=True)  # Preserve the original forced completion report
    return unique, unique_labels  # Return complete sampled vocabularies for encoder fitting


def build_encoders(unique: Mapping[str, Set[str]], unique_labels: Set[str], out_dir: Path) -> Tuple[Dict[str, LabelEncoder], LabelEncoder]:
    """
    Fit and persist categorical feature encoders and the output label encoder.

    :param unique: Complete categorical feature vocabularies keyed by canonical feature name.
    :param unique_labels: Complete canonical output-label vocabulary.
    :param out_dir: Directory receiving persisted encoder joblib files.
    :return: Fitted feature-encoder mapping and fitted output label encoder.
    """

    feature_encoders: Dict[str, LabelEncoder] = {}  # Collect one fitted LabelEncoder per categorical selected feature
    for column in CATEGORICAL_SELECTED:  # Preserve iteration over the original categorical-feature set
        encoder = LabelEncoder()  # Create the same scikit-learn categorical encoder type
        encoder.fit(np.asarray(sorted(unique[column]), dtype=object))  # Fit on deterministically sorted sampled vocabulary values
        feature_encoders[column] = encoder  # Store the fitted encoder by canonical feature name
        joblib.dump(encoder, out_dir / f"label_encoder_{normalize_column(column)}.joblib")  # Preserve the original feature-encoder filename convention
    label_encoder = LabelEncoder()  # Create the output class encoder
    label_encoder.fit(np.asarray(sorted(unique_labels), dtype=object))  # Fit on deterministically sorted canonical output labels
    joblib.dump(label_encoder, out_dir / "label_encoder_output.joblib")  # Preserve the original output-encoder filename
    if tuple(label_encoder.classes_.tolist()) != PAPER_FIGURE6_CLASSES:  # Verify if encoded class IDs follow the Figure 6(c) order expected downstream
        raise RuntimeError(
            "Encoded class order differs from Figure 6 order. "
            f"Got {label_encoder.classes_.tolist()}"
        )  # Preserve the original class-order safety check
    return feature_encoders, label_encoder  # Return fitted encoders for sample-array construction


def fit_sample_encoders(sample_gz: Path, chunksize: int, total_rows: int, out_dir: Path) -> Tuple[Dict[str, LabelEncoder], LabelEncoder]:
    """
    Collect sampled vocabularies, fit LabelEncoder objects, and persist them.

    :param sample_gz: Compressed sampled selected-feature CSV path.
    :param chunksize: Number of sampled rows read per pandas chunk.
    :param total_rows: Expected total sampled row count.
    :param out_dir: Directory receiving persisted encoder joblib files.
    :return: Fitted categorical feature encoders and output label encoder.
    """

    unique, unique_labels = collect_vocabularies(sample_gz, chunksize, total_rows)  # Scan the sampled dataset for complete encoder vocabularies
    return build_encoders(unique, unique_labels, out_dir)  # Fit, validate, persist, and return the encoder objects


def encode_feature_block(chunk: pd.DataFrame, feature_encoders: Mapping[str, LabelEncoder]) -> np.ndarray:
    """
    Encode one sampled DataFrame chunk into the ordered float32 top-20 feature matrix.

    :param chunk: Sampled selected-feature DataFrame chunk.
    :param feature_encoders: Fitted categorical feature encoders by canonical feature name.
    :return: Float32 encoded feature matrix for the current chunk.
    """

    block = np.empty((len(chunk), len(PAPER_TOP20)), dtype=np.float32)  # Allocate the exact output block shape and dtype once
    for feature_index, column in enumerate(PAPER_TOP20):  # Encode selected features in the published order
        if column in CATEGORICAL_SELECTED:  # Verify if this feature requires its fitted LabelEncoder
            block[:, feature_index] = feature_encoders[column].transform(chunk[column].astype("string")).astype(np.float32)  # Preserve categorical integer encoding converted to float32 model features
        else:  # Handle numeric selected features
            values = pd.to_numeric(chunk[column], errors="raise").to_numpy(dtype=np.float32, copy=False)  # Preserve strict downstream numeric conversion
            if not np.isfinite(values).all():  # Verify if cleaning unexpectedly allowed a non-finite numeric selected value
                raise RuntimeError(f"Non-finite value survived cleaning in {column}")  # Preserve the original defensive failure
            block[:, feature_index] = values  # Copy numeric feature values into the correct published feature column
    return block  # Return the encoded feature block aligned with current sampled rows


def encode_sample_to_npy(sample_gz: Path, chunksize: int, total_rows: int, feature_encoders: Mapping[str, LabelEncoder], label_encoder: LabelEncoder, x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream the compressed sampled dataset into memory-mappable encoded NumPy arrays.

    :param sample_gz: Compressed sampled selected-feature CSV path.
    :param chunksize: Number of sampled rows read per pandas chunk.
    :param total_rows: Exact sampled row count expected in output arrays.
    :param feature_encoders: Fitted categorical feature encoders.
    :param label_encoder: Fitted canonical output label encoder.
    :param x_path: Destination encoded feature .npy path.
    :param y_path: Destination encoded integer-label .npy path.
    :return: Read-only memory-mapped encoded feature and label arrays.
    """

    feature_memmap = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(total_rows, len(PAPER_TOP20)))  # Create disk-backed feature array with the original shape and dtype
    label_memmap = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int16, shape=(total_rows,))  # Create disk-backed integer-label array
    eta = create_eta("ENCODE", total_rows)  # Initialize the original encoding row-progress reporter
    offset = 0  # Track the next destination row in both memory-mapped arrays
    for chunk in pd.read_csv(sample_gz, chunksize=chunksize, low_memory=False):  # Stream sampled selected rows from compressed disk storage
        row_count = len(chunk)  # Capture current sampled block size
        block = encode_feature_block(chunk, feature_encoders)  # Encode the published top-20 features for this block
        label_block = label_encoder.transform(chunk["Label"].astype("string")).astype(np.int16)  # Encode canonical output labels to the original int16 dtype
        feature_memmap[offset:offset + row_count] = block  # Persist encoded features directly into the disk-backed array
        label_memmap[offset:offset + row_count] = label_block  # Persist aligned encoded labels directly into the disk-backed array
        offset += row_count  # Advance the destination position by rows written
        eta.report(offset, f"rows={offset:,}/{total_rows:,}")  # Preserve the original encoding progress detail
        del chunk, block, label_block  # Release sampled and encoded block objects before the next chunk
        gc.collect()  # Preserve explicit garbage collection from the supplied implementation
    if offset != total_rows:  # Verify if the encoded sample contains exactly the expected number of rows
        raise RuntimeError(f"Encoded row count mismatch: expected {total_rows}, got {offset}")  # Preserve the original encoded-row mismatch failure
    feature_memmap.flush()  # Flush pending feature-array writes to disk
    label_memmap.flush()  # Flush pending label-array writes to disk
    del feature_memmap, label_memmap  # Close writable memory-map objects before reopening read-only views
    eta.report(total_rows, "encoded cache complete", force=True)  # Preserve the original forced encoding completion message
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")  # Reopen and return read-only memory-mapped caches exactly as supplied
