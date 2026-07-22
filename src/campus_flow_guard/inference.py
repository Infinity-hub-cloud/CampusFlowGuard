"""Local CSV inference for the selected Focal Loss model."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=r"urllib3 .*doesn't match a supported version!")
warnings.filterwarnings("ignore", message=r"Skipping variable loading for optimizer .*", category=UserWarning)

import numpy as np
import pandas as pd
import yaml

from campus_flow_guard.data_protocol import sha256_file
from campus_flow_guard.risk import RiskMapping, assess_risk, load_risk_mapping


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def validate_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")


def build_inference_windows(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Inference feature values must be a two-dimensional array")
    if window_size < 2 or len(values) < window_size:
        raise ValueError(f"Input CSV must contain at least {window_size} rows")
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2).copy()
    target_rows = np.arange(window_size - 1, len(values), dtype=np.int64)
    return windows, target_rows


def build_detection_output(
    probabilities: np.ndarray,
    target_rows: np.ndarray,
    *,
    decision_threshold: float,
    risk_mapping: RiskMapping,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target_rows = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    if len(probabilities) != len(target_rows):
        raise ValueError("Probability and target-row counts differ")
    assessments = [
        assess_risk(float(probability), decision_threshold, risk_mapping)
        for probability in probabilities
    ]
    detections = [
        {
            "window_index": int(index),
            "target_row_number": int(target_rows[index]),
            "attack_probability": assessment.attack_probability,
            "detected_attack": bool(assessment.attack_probability >= decision_threshold),
            "risk_level": assessment.risk_level,
            "decision_threshold": assessment.decision_threshold,
            "high_cutoff": assessment.high_cutoff,
        }
        for index, assessment in enumerate(assessments)
    ]
    risk_counts = {
        label: sum(item["risk_level"] == label for item in detections)
        for label in (risk_mapping.low_label, risk_mapping.medium_label, risk_mapping.high_label)
    }
    detected_count = sum(item["detected_attack"] for item in detections)
    summary = {
        "input_row_count": int(metadata["input_row_count"]),
        "window_count": len(detections),
        "detected_attack_count": int(detected_count),
        "detected_attack_ratio": float(detected_count / len(detections)) if detections else 0.0,
        "risk_level_counts": risk_counts,
        "maximum_attack_probability": float(probabilities.max()) if len(probabilities) else None,
        "mean_attack_probability": float(probabilities.mean()) if len(probabilities) else None,
        "decision_threshold": float(decision_threshold),
        "high_cutoff": assessments[0].high_cutoff if assessments else None,
    }
    return {"metadata": metadata, "detection_summary": summary, "detections": detections}


def _resolve(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def run(config_path: Path, input_path: Path) -> dict[str, Any]:
    from tensorflow import keras

    from campus_flow_guard.baseline import _build_record_projection_model

    config_path = config_path.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file() or input_path.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    project_root = config_path.parents[2]
    config = _read_yaml(config_path)
    model_config = config["model"]
    preprocessing_config = config["preprocessing"]
    decision_config = config["decision"]
    inference_config = config["inference"]

    weights_path = _resolve(project_root, str(model_config["weights_path"]))
    preprocessing_path = _resolve(project_root, str(preprocessing_config["artifact_path"]))
    frozen_model_config_path = _resolve(project_root, str(preprocessing_config["model_config_path"]))
    focal_result_path = _resolve(project_root, str(model_config["focal_experiment_result"]))
    focal_log_path = _resolve(project_root, str(model_config["focal_training_log"]))
    risk_mapping_path = _resolve(project_root, str(decision_config["risk_mapping_path"]))
    if sha256_file(weights_path) != str(model_config["weights_sha256"]):
        raise ValueError("Selected Focal weight SHA-256 does not match inference config")
    if sha256_file(preprocessing_path) != str(preprocessing_config["artifact_sha256"]):
        raise ValueError("Frozen preprocessing SHA-256 does not match inference config")

    focal_result = json.loads(focal_result_path.read_text(encoding="utf-8"))
    if focal_result.get("test_used_for_selection") is not False:
        raise ValueError("Focal experiment indicates test participated in model selection")
    seed = int(model_config["seed"])
    seed_result = next(
        (item for item in focal_result["focal_results"] if int(item["seed"]) == seed),
        None,
    )
    if seed_result is None:
        raise ValueError(f"Selected seed is absent from Focal experiment: {seed}")
    decision_threshold = float(decision_config["threshold"])
    if not np.isclose(decision_threshold, float(seed_result["threshold"])):
        raise ValueError("Inference threshold differs from the validation-selected Focal threshold")
    training_log = pd.read_csv(focal_log_path)
    best_validation_pr_auc = float(training_log["val_pr_auc"].max())
    if not np.isclose(best_validation_pr_auc, float(model_config["selection_metric_value"])):
        raise ValueError("Configured validation model-selection metric does not match training log")

    with preprocessing_path.open("rb") as handle:
        preprocessing = pickle.load(handle)
    frame = pd.read_csv(input_path)
    numeric_columns = [str(column) for column in preprocessing["numeric_columns"]]
    categorical_columns = [str(column) for column in preprocessing["categorical_columns"]]
    validate_required_columns(frame, [*numeric_columns, *categorical_columns])
    numeric = preprocessing["numeric_pipeline"].transform(frame[numeric_columns]).astype(np.float32)
    categorical = (
        preprocessing["categorical_encoder"]
        .transform(frame[categorical_columns].astype(str))
        .astype(np.int32)
        + 1
    )
    window_size = int(inference_config["window_size"])
    numeric_windows, target_rows = build_inference_windows(numeric, window_size)
    categorical_windows = [
        build_inference_windows(categorical[:, index : index + 1], window_size)[0].squeeze(-1)
        for index in range(categorical.shape[1])
    ]

    frozen_config = _read_yaml(frozen_model_config_path)
    if window_size != int(frozen_config["model"]["window_size"]):
        raise ValueError("Inference window size differs from trained model")
    keras.backend.clear_session()
    model = _build_record_projection_model(
        window_size=window_size,
        numeric_feature_count=len(numeric_columns),
        vocabulary_sizes=[
            len(categories) + 1 for categories in preprocessing["categorical_encoder"].categories_
        ],
        model_config=frozen_config["model"],
        learning_rate=float(frozen_config["training"]["learning_rate"]),
    )
    model.load_weights(weights_path)
    probabilities = model.predict(
        (numeric_windows, *categorical_windows),
        batch_size=int(inference_config["batch_size"]),
        verbose=0,
    ).reshape(-1)
    risk_mapping = load_risk_mapping(risk_mapping_path)
    metadata = {
        "input_csv": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_row_count": int(len(frame)),
        "window_size": window_size,
        "target_position": "last_row",
        "model": "Focal Loss Record Projection Transformer",
        "model_seed": seed,
        "model_selection_scope": "validation_only",
        "model_selection_metric": str(model_config["selection_metric"]),
        "model_selection_metric_value": best_validation_pr_auc,
        "weights_sha256": str(model_config["weights_sha256"]),
        "threshold_source": str(decision_config["threshold_source"]),
    }
    return build_detection_output(
        probabilities,
        target_rows,
        decision_threshold=decision_threshold,
        risk_mapping=risk_mapping,
        metadata=metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local CampusFlowGuard CSV inference")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        document = run(args.config, args.input)
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary = document["detection_summary"]
        print(
            f"推理完成: windows={summary['window_count']}, detected={summary['detected_attack_count']}, "
            f"max_probability={summary['maximum_attack_probability']:.6f}"
        )
        print(f"输出文件: {output_path}")
    except Exception as exc:
        print(f"本地推理失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
