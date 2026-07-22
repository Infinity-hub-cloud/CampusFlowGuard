"""Leakage-resistant split and window utilities for trusted baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Original CSV row indices assigned to exactly one partition."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    @property
    def all_indices(self) -> np.ndarray:
        return np.concatenate((self.train, self.validation, self.test))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_stratified_split(
    labels: np.ndarray,
    *,
    pool_size: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitIndices:
    """Create one deterministic stratified pool and mutually exclusive splits."""

    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("labels must be a one-dimensional array with at least three rows")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1")
    if pool_size <= 0 or pool_size > len(labels):
        raise ValueError("pool_size must be positive and no larger than the dataset")

    all_indices = np.arange(len(labels), dtype=np.int64)
    if pool_size < len(labels):
        pool_indices, _ = train_test_split(
            all_indices,
            train_size=pool_size,
            random_state=seed,
            stratify=labels,
        )
    else:
        pool_indices = all_indices

    train_validation_indices, test_indices = train_test_split(
        pool_indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels[pool_indices],
    )
    validation_relative_fraction = validation_fraction / (train_fraction + validation_fraction)
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=validation_relative_fraction,
        random_state=seed,
        stratify=labels[train_validation_indices],
    )
    split = SplitIndices(
        train=np.sort(train_indices.astype(np.int64)),
        validation=np.sort(validation_indices.astype(np.int64)),
        test=np.sort(test_indices.astype(np.int64)),
    )
    assert_split_integrity(split, expected_pool_size=pool_size)
    return split


def assert_split_integrity(split: SplitIndices, *, expected_pool_size: int | None = None) -> None:
    arrays = (split.train, split.validation, split.test)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Every split index array must be one-dimensional")
    if any(len(array) == 0 for array in arrays):
        raise ValueError("Every split must contain at least one row")
    if any(len(array) != len(np.unique(array)) for array in arrays):
        raise ValueError("Duplicate indices exist within a split")
    if np.intersect1d(split.train, split.validation).size:
        raise ValueError("Train and validation indices overlap")
    if np.intersect1d(split.train, split.test).size:
        raise ValueError("Train and test indices overlap")
    if np.intersect1d(split.validation, split.test).size:
        raise ValueError("Validation and test indices overlap")
    if expected_pool_size is not None and len(split.all_indices) != expected_pool_size:
        raise ValueError("Split size does not match the configured experiment pool")


def build_partition_window_indices(
    partition_indices: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build source-row windows wholly inside one already separated partition."""

    indices = np.asarray(partition_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("partition_indices must be one-dimensional")
    if window_size < 2 or len(indices) < window_size:
        raise ValueError("window_size must be >= 2 and fit inside the partition")
    ordered = np.sort(indices)
    windows = np.lib.stride_tricks.sliding_window_view(ordered, window_size).copy()
    return windows, windows[:, -1].copy()


def save_split_manifest(
    split: SplitIndices,
    *,
    npz_path: Path,
    json_path: Path,
    metadata: dict[str, Any],
    labels: np.ndarray,
) -> None:
    assert_split_integrity(split, expected_pool_size=int(metadata["pool_size"]))
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        train=split.train,
        validation=split.validation,
        test=split.test,
    )
    binary_labels = np.asarray(labels, dtype=np.int8)
    document = {
        **metadata,
        "index_file": str(npz_path.resolve()),
        "partitions": {
            name: {
                "row_count": int(len(indices)),
                "benign_count": int(np.count_nonzero(binary_labels[indices] == 0)),
                "attack_count": int(np.count_nonzero(binary_labels[indices] == 1)),
                "minimum_source_index": int(indices.min()),
                "maximum_source_index": int(indices.max()),
            }
            for name, indices in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
        "integrity": {
            "indices_are_mutually_exclusive": True,
            "windows_built_after_split": True,
            "time_extrapolation_claimed": False,
        },
    }
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_split_manifest(npz_path: Path, *, expected_pool_size: int) -> SplitIndices:
    with np.load(npz_path, allow_pickle=False) as loaded:
        split = SplitIndices(
            train=loaded["train"].astype(np.int64),
            validation=loaded["validation"].astype(np.int64),
            test=loaded["test"].astype(np.int64),
        )
    assert_split_integrity(split, expected_pool_size=expected_pool_size)
    return split


def load_selected_rows(
    csv_path: Path,
    *,
    columns: list[str],
    selected_indices: np.ndarray,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Read only selected original rows while scanning a real CSV in chunks."""

    wanted = np.sort(np.asarray(selected_indices, dtype=np.int64))
    if len(wanted) == 0 or len(wanted) != len(np.unique(wanted)):
        raise ValueError("selected_indices must be non-empty and unique")
    selected_chunks: list[pd.DataFrame] = []
    offset = 0
    cursor = 0
    for chunk in pd.read_csv(csv_path, usecols=columns, chunksize=chunk_size):
        upper = offset + len(chunk)
        next_cursor = int(np.searchsorted(wanted, upper, side="left"))
        source_indices = wanted[cursor:next_cursor]
        if len(source_indices):
            local_indices = source_indices - offset
            selected = chunk.iloc[local_indices].copy()
            selected.index = source_indices
            selected.index.name = "source_index"
            selected_chunks.append(selected)
        cursor = next_cursor
        offset = upper
        if cursor == len(wanted):
            break
    if cursor != len(wanted):
        raise ValueError("Some split indices exceed the CSV row count")
    frame = pd.concat(selected_chunks).sort_index()
    if not np.array_equal(frame.index.to_numpy(dtype=np.int64), wanted):
        raise RuntimeError("Loaded source indices differ from the requested split pool")
    return frame
