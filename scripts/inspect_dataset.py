from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TIME_NAME_PATTERN = re.compile(r"(?:time|date|timestamp|datetime|_ts$)", re.IGNORECASE)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value_counts(series: pd.Series) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False)
    return [
        {"value": _json_value(value), "count": int(count)}
        for value, count in counts.items()
    ]


def _read_schema(schema_path: Path | None) -> dict[str, Any]:
    if schema_path is None:
        return {}
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file does not exist: {schema_path}")
    loaded = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Schema root must be a mapping: {schema_path}")
    return loaded


def _numeric_inf_counts(frame: pd.DataFrame) -> dict[str, int]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    mask = np.isinf(values)
    return {
        column: int(mask[:, index].sum())
        for index, column in enumerate(numeric.columns)
        if int(mask[:, index].sum()) > 0
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 of an existing file without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(
    csv_path: Path,
    *,
    schema_path: Path | None = None,
    label_column: str | None = None,
    time_column: str | None = None,
    nrows: int | None = None,
) -> dict[str, Any]:
    """Inspect a real CSV without changing it or fabricating missing statistics."""

    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV path, received: {csv_path}")
    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive when provided")

    schema = _read_schema(schema_path.resolve() if schema_path else None)
    frame = pd.read_csv(csv_path, nrows=nrows)
    fields = [str(column) for column in frame.columns]
    label = label_column or schema.get("label_column")
    configured_categories = schema.get("categorical_columns", [])
    categorical_columns = [
        str(column) for column in configured_categories if str(column) in frame.columns
    ]
    if not categorical_columns:
        categorical_columns = [
            str(column)
            for column in frame.select_dtypes(include=["object", "category", "bool"]).columns
        ]

    explicit_time = time_column or schema.get("time_column")
    time_candidates = [
        column for column in fields if TIME_NAME_PATTERN.search(column)
    ]
    if explicit_time and explicit_time in fields and explicit_time not in time_candidates:
        time_candidates.insert(0, explicit_time)

    schema_features = [str(column) for column in schema.get("feature_columns", [])]
    expected_schema_columns = schema_features + ([str(label)] if label else [])
    missing_schema_columns = [column for column in expected_schema_columns if column not in fields]
    extra_schema_columns = [column for column in fields if column not in expected_schema_columns]

    inf_by_column = _numeric_inf_counts(frame)
    constant_columns = [
        str(column)
        for column in frame.columns
        if int(frame[column].nunique(dropna=False)) <= 1
    ]
    label_distribution = None
    label_error = None
    if label:
        if label in frame.columns:
            label_distribution = _value_counts(frame[label])
        else:
            label_error = f"Configured label column is absent: {label}"

    benign_label = schema.get("benign_label")
    binary_label_summary = None
    if label and label in frame.columns and benign_label is not None:
        labels = frame[label]
        missing_label_count = int(labels.isna().sum())
        benign_count = int((labels == benign_label).sum())
        attack_count = int(len(labels) - benign_count - missing_label_count)
        non_missing_count = int(len(labels) - missing_label_count)
        binary_label_summary = {
            "benign_label": str(benign_label),
            "benign_count": benign_count,
            "attack_count_non_benign": attack_count,
            "missing_label_count": missing_label_count,
            "non_missing_label_count": non_missing_count,
            "benign_ratio_non_missing": (
                benign_count / non_missing_count if non_missing_count else None
            ),
            "attack_ratio_non_missing": (
                attack_count / non_missing_count if non_missing_count else None
            ),
        }

    category_distribution = {
        column: {
            "unique_count_in_loaded_rows": int(frame[column].nunique(dropna=True)),
            "value_counts": _value_counts(frame[column]),
        }
        for column in categorical_columns
    }

    result: dict[str, Any] = {
        "inspection_scope": "full_file" if nrows is None else f"first_{nrows}_rows",
        "csv_path": str(csv_path),
        "file_size_bytes": int(csv_path.stat().st_size),
        "file_size_mib": round(csv_path.stat().st_size / (1024 * 1024), 3),
        "file_sha256": _sha256(csv_path),
        "row_count_loaded": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "fields": fields,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
        },
        "missing_value_total": int(frame.isna().sum().sum()),
        "infinite_values_by_numeric_column": inf_by_column,
        "infinite_value_total": int(sum(inf_by_column.values())),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "constant_columns": constant_columns,
        "constant_column_count": len(constant_columns),
        "label_column": label,
        "label_distribution": label_distribution,
        "label_error": label_error,
        "binary_label_summary": binary_label_summary,
        "categorical_columns_checked": categorical_columns,
        "category_counts": category_distribution,
        "explicit_time_column": explicit_time,
        "time_column_exists": bool(explicit_time and explicit_time in fields),
        "time_column_candidates_by_name": time_candidates,
        "schema_missing_columns": missing_schema_columns,
        "schema_extra_columns": extra_schema_columns if schema else [],
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a real CSV dataset and emit JSON diagnostics."
    )
    parser.add_argument("csv_path", type=Path, help="Path to an existing CSV file")
    parser.add_argument("--schema", type=Path, help="Optional YAML dataset schema")
    parser.add_argument("--label-column", help="Override the schema label column")
    parser.add_argument("--time-column", help="Declare a candidate time column")
    parser.add_argument(
        "--nrows",
        type=int,
        help="Inspect only the first N rows; output is marked as partial",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = inspect_dataset(
            args.csv_path,
            schema_path=args.schema,
            label_column=args.label_column,
            time_column=args.time_column,
            nrows=args.nrows,
        )
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
