from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.data_protocol import load_selected_rows, sha256_file


EXCLUDED_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "Label"}


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Document root must be a mapping: {path}")
    return loaded


def compose_demo_frames(
    validation_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    benign_label: str,
    class_count: int = 16,
    mixed_per_class: int = 8,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[int]]]:
    """Select deterministic validation rows and return feature-only demo frames."""

    if class_count < 8 or mixed_per_class < 8 or mixed_per_class > class_count:
        raise ValueError("Class counts must support at least one 8-row inference window")
    required = [*feature_columns, label_column]
    missing = [column for column in required if column not in validation_frame.columns]
    if missing:
        raise ValueError(f"Validation data is missing required columns: {missing}")
    if validation_frame[label_column].isna().any():
        raise ValueError("Validation labels contain missing values")

    ordered = validation_frame.sort_index()
    benign = ordered.loc[ordered[label_column] == benign_label]
    attack = ordered.loc[ordered[label_column] != benign_label]
    if len(benign) < class_count or len(attack) < class_count:
        raise ValueError("Validation split does not contain enough rows for both demo classes")

    benign = benign.iloc[:class_count]
    attack = attack.iloc[:class_count]
    mixed = pd.concat((benign.iloc[:mixed_per_class], attack.iloc[:mixed_per_class]))
    selected = {
        "benign_demo.csv": benign,
        "attack_demo.csv": attack,
        "mixed_demo.csv": mixed,
    }
    frames = {
        name: frame.loc[:, feature_columns].reset_index(drop=True)
        for name, frame in selected.items()
    }
    source_indices = {
        name: [int(index) for index in frame.index]
        for name, frame in selected.items()
    }
    return frames, source_indices


def prepare_demo_data(
    *,
    dataset_path: Path,
    schema_path: Path,
    split_path: Path,
    split_manifest_path: Path,
    frozen_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    schema = _read_mapping(schema_path)
    split_manifest = _read_mapping(split_manifest_path)
    frozen_manifest = _read_mapping(frozen_manifest_path)
    feature_columns = [str(column) for column in schema["feature_columns"]]
    label_column = str(schema["label_column"])
    benign_label = str(schema["benign_label"])
    if len(feature_columns) != 37 or len(set(feature_columns)) != 37:
        raise ValueError("The demo schema must contain 37 unique model features")
    if EXCLUDED_COLUMNS.intersection(feature_columns):
        raise ValueError("The feature whitelist contains an identifier or label")

    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != str(split_manifest["dataset_sha256"]):
        raise ValueError("Dataset SHA-256 differs from the frozen split manifest")
    split_entry = next(
        item
        for item in frozen_manifest["files"]
        if item["relative_path"] == "splits/split_indices.npz"
    )
    split_hash = sha256_file(split_path)
    if split_hash != str(split_entry["sha256"]):
        raise ValueError("Split index SHA-256 differs from the frozen baseline manifest")

    with np.load(split_path, allow_pickle=False) as loaded:
        validation_indices = loaded["validation"].astype(np.int64)
        test_indices = loaded["test"].astype(np.int64)
    if np.intersect1d(validation_indices, test_indices).size:
        raise ValueError("Frozen validation and test indices overlap")

    validation_frame = load_selected_rows(
        dataset_path,
        columns=[*feature_columns, label_column],
        selected_indices=validation_indices,
    )
    frames, source_indices = compose_demo_frames(
        validation_frame,
        feature_columns=feature_columns,
        label_column=label_column,
        benign_label=benign_label,
    )
    chosen = np.asarray(
        sorted({index for indices in source_indices.values() for index in indices}),
        dtype=np.int64,
    )
    if not np.isin(chosen, validation_indices).all() or np.isin(chosen, test_indices).any():
        raise ValueError("Demo selection is not validation-only")

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        files[name] = {
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "source_indices": source_indices[name],
        }

    document = {
        "version": 1,
        "purpose": "minimal_reproducible_local_inference_demo",
        "source": {
            "dataset_filename": dataset_path.name,
            "dataset_sha256": dataset_hash,
            "source_and_license_status": split_manifest["source_and_license_status"],
            "partition": "validation",
            "split_seed": int(split_manifest["split_seed"]),
            "split_indices_sha256": split_hash,
            "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
        },
        "selection": {
            "rule": "ascending_source_index_within_frozen_validation_partition",
            "benign_rule": f"{label_column} == {benign_label}",
            "attack_rule": f"{label_column} != {benign_label}",
            "mixed_composition": "first_8_benign_then_first_8_attack",
            "temporal_order_claimed": False,
            "test_rows_used": False,
        },
        "privacy": {
            "feature_only": True,
            "excluded_columns": sorted(EXCLUDED_COLUMNS),
        },
        "feature_columns": feature_columns,
        "files": files,
    }
    (output_dir / "demo_manifest.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare validation-only demo CSVs")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\NF-UNSW-NB15-v2.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "demo"
    )
    args = parser.parse_args()
    try:
        document = prepare_demo_data(
            dataset_path=args.dataset.expanduser().resolve(),
            schema_path=PROJECT_ROOT / "configs" / "datasets" / "dataset_schema.yaml",
            split_path=PROJECT_ROOT / "artifacts" / "baseline_freeze"
            / "unsw_trusted_baseline_v1" / "splits" / "split_indices.npz",
            split_manifest_path=PROJECT_ROOT / "artifacts" / "baseline_freeze"
            / "unsw_trusted_baseline_v1" / "splits" / "split_manifest.json",
            frozen_manifest_path=PROJECT_ROOT / "artifacts" / "baseline_freeze"
            / "unsw_trusted_baseline_v1" / "manifest.json",
            output_dir=args.output_dir.expanduser().resolve(),
        )
    except Exception as exc:
        print(f"演示数据准备失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "演示数据已生成: "
        + ", ".join(
            f"{name}={details['row_count']}行"
            for name, details in document["files"].items()
        )
    )
    print(f"输出目录: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
