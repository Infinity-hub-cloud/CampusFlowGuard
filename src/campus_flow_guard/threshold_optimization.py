"""Validation-only threshold optimization for a frozen trusted baseline."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from campus_flow_guard.data_protocol import SplitIndices, load_selected_rows, sha256_file


POLICY_FIXED = "fixed_0_5"
POLICY_F1 = "validation_f1_max"
POLICY_FAR = "validation_far_le_0_005_max_recall"


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate threshold metrics with FAR = FP / (FP + TN)."""

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be one-dimensional and equally sized")
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def scan_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    step: float,
) -> pd.DataFrame:
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("Threshold bounds must satisfy 0 <= minimum < maximum <= 1")
    if step <= 0 or step > maximum - minimum:
        raise ValueError("Threshold step must be positive and fit inside the configured range")
    count = int(round((maximum - minimum) / step))
    thresholds = np.linspace(minimum, maximum, count + 1, dtype=np.float64)
    return pd.DataFrame([binary_metrics(labels, probabilities, float(value)) for value in thresholds])


def select_validation_thresholds(search: pd.DataFrame, *, far_limit: float) -> dict[str, dict[str, Any]]:
    """Lock threshold policies using only a validation search table."""

    required = {"threshold", "precision", "recall", "f1", "far", "balanced_accuracy"}
    if not required <= set(search.columns):
        raise ValueError(f"Validation search is missing columns: {sorted(required - set(search.columns))}")
    fixed_rows = search[np.isclose(search["threshold"], 0.5)]
    if len(fixed_rows) != 1:
        raise ValueError("Threshold grid must contain exactly one 0.5 row")
    f1_row = search.sort_values(
        ["f1", "balanced_accuracy", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0]
    constrained = search[search["far"] <= far_limit + 1e-12]
    if constrained.empty:
        raise ValueError("No validation threshold satisfies the configured FAR limit")
    far_row = constrained.sort_values(
        ["recall", "f1", "balanced_accuracy", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0]

    def row_to_dict(row: pd.Series) -> dict[str, Any]:
        return {
            key: int(value) if key in {"TP", "FP", "TN", "FN"} else float(value)
            for key, value in row.items()
        }

    return {
        POLICY_FIXED: row_to_dict(fixed_rows.iloc[0]),
        POLICY_F1: row_to_dict(f1_row),
        POLICY_FAR: row_to_dict(far_row),
    }


def evaluate_locked_thresholds(
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    locked_validation_selection: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate already locked policies without changing their thresholds."""

    return [
        {"policy": policy, **binary_metrics(test_labels, test_probabilities, selection["threshold"])}
        for policy, selection in locked_validation_selection.items()
    ]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def _validate_frozen_input(freeze_root: Path, relative_path: str, manifest: dict[str, Any]) -> Path:
    path = freeze_root / relative_path
    records = {record["relative_path"]: record for record in manifest["files"]}
    if relative_path not in records or not path.is_file():
        raise FileNotFoundError(f"Frozen artifact is missing: {relative_path}")
    if sha256_file(path) != records[relative_path]["sha256"]:
        raise ValueError(f"Frozen artifact hash mismatch: {relative_path}")
    return path


def _plot_validation_search(search: pd.DataFrame, selection: dict[str, dict[str, Any]], path: Path, seed: int) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(search["threshold"], search["f1"], label="F1")
    axis.plot(search["threshold"], search["recall"], label="Recall")
    axis.plot(search["threshold"], search["far"], label="FAR")
    for policy, result in selection.items():
        axis.axvline(result["threshold"], linestyle="--", linewidth=1, label=f"{policy}: {result['threshold']:.3f}")
    axis.axhline(0.005, color="black", linestyle=":", linewidth=1, label="FAR limit 0.5%")
    axis.set(xlabel="Threshold", ylabel="Metric", title=f"Validation threshold search, seed {seed}")
    axis.set_ylim(-0.01, 1.01)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_test_comparison(frame: pd.DataFrame, path: Path) -> None:
    means = frame.groupby("policy", sort=False)[["f1", "recall", "far"]].mean()
    axis = means.plot(kind="bar", figsize=(7.2, 4.5))
    axis.set(title="Frozen test comparison after validation-only threshold locking", ylabel="Mean across seeds")
    axis.tick_params(axis="x", labelrotation=15)
    axis.legend()
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)


def _write_markdown_comparison(path: Path, selection_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Validation 阈值优化对比",
        "",
        "所有阈值仅由 validation 锁定。test 概率在阈值锁定后每个 seed 只推理一次，并用于三个已锁定策略的并列评估。",
        "",
        "## Validation 选择",
        "",
        "| Seed | Policy | Threshold | Precision | Recall | F1 | FAR | Balanced Accuracy |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['seed']} | {row['policy']} | {row['threshold']:.3f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['far']:.4f} | {row['balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Test 一次性对比",
            "",
            "| Seed | Policy | Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FAR | Balanced Accuracy |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| {row['seed']} | {row['policy']} | {row['threshold']:.3f} | {row['TP']} | {row['FP']} | "
            f"{row['TN']} | {row['FN']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['far']:.4f} | {row['balanced_accuracy']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Path]:
    from tensorflow import keras

    from campus_flow_guard.baseline import _build_record_projection_model, _prepare_partition

    config_path = config_path.expanduser().resolve()
    project_root = config_path.parents[2]
    config = _read_yaml(config_path)
    freeze_root = (project_root / str(config["run"]["frozen_baseline_dir"])).resolve()
    output_root = (project_root / str(config["outputs"]["directory"])).resolve()
    if freeze_root == output_root or freeze_root in output_root.parents:
        raise ValueError("Threshold outputs must not be written inside the frozen baseline archive")
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_root / "predictions"
    plots_dir = output_root / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = output_root / "threshold_optimization.json"
    test_comparison_path = output_root / "test_threshold_comparison.csv"
    markdown_path = output_root / "threshold_comparison.md"
    if final_json_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed threshold result: {final_json_path}")

    manifest_path = freeze_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frozen_config_path = _validate_frozen_input(freeze_root, "config/executed_config.yaml", manifest)
    split_path = _validate_frozen_input(freeze_root, "splits/split_indices.npz", manifest)
    preprocessing_path = _validate_frozen_input(freeze_root, "models/preprocessing.pkl", manifest)
    frozen_config = _read_yaml(frozen_config_path)
    with np.load(split_path, allow_pickle=False) as loaded:
        split = SplitIndices(
            train=loaded["train"].astype(np.int64),
            validation=loaded["validation"].astype(np.int64),
            test=loaded["test"].astype(np.int64),
        )
    with preprocessing_path.open("rb") as handle:
        preprocessing = pickle.load(handle)

    schema_path = (project_root / str(frozen_config["data"]["schema_path"])).resolve()
    frozen_fingerprint = json.loads(
        _validate_frozen_input(freeze_root, "metadata/data_fingerprint.json", manifest).read_text(encoding="utf-8")
    )
    if sha256_file(schema_path) != frozen_fingerprint["schema_sha256"]:
        raise ValueError("Current schema differs from the schema fingerprint used by the frozen baseline")
    schema = _read_yaml(schema_path)
    label_column = str(schema["label_column"])
    benign_label = str(schema["benign_label"])
    csv_path = Path(str(frozen_config["data"]["csv_path"])).resolve()
    if sha256_file(csv_path) != frozen_fingerprint["dataset_sha256"]:
        raise ValueError("Dataset differs from the frozen baseline fingerprint")

    selected_indices = np.sort(np.concatenate((split.validation, split.test)))
    selected_frame = load_selected_rows(
        csv_path,
        columns=[*preprocessing["numeric_columns"], *preprocessing["categorical_columns"], label_column],
        selected_indices=selected_indices,
        chunk_size=int(frozen_config["data"]["read_chunk_size"]),
    )
    model_config = frozen_config["model"]
    window_size = int(model_config["window_size"])
    common = {
        "numeric_columns": preprocessing["numeric_columns"],
        "categorical_columns": preprocessing["categorical_columns"],
        "label_column": label_column,
        "benign_label": benign_label,
        "numeric_pipeline": preprocessing["numeric_pipeline"],
        "categorical_encoder": preprocessing["categorical_encoder"],
        "window_size": window_size,
    }
    validation = _prepare_partition(selected_frame.loc[split.validation], **common)
    test = _prepare_partition(selected_frame.loc[split.test], **common)
    del selected_frame

    search_config = config["threshold_search"]
    validation_selection_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    seed_documents: list[dict[str, Any]] = []
    for seed in [int(value) for value in manifest["model_seeds"]]:
        cache_path = predictions_dir / f"seed{seed}_probabilities.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                validation_probabilities = cached["validation_probabilities"]
                test_probabilities = cached["test_probabilities"]
                if not np.array_equal(cached["validation_labels"], validation.labels):
                    raise ValueError(f"Cached validation labels differ for seed {seed}")
                if not np.array_equal(cached["test_labels"], test.labels):
                    raise ValueError(f"Cached test labels differ for seed {seed}")
            prediction_source = "reused_probability_cache"
        else:
            keras.backend.clear_session()
            model = _build_record_projection_model(
                window_size=window_size,
                numeric_feature_count=len(preprocessing["numeric_columns"]),
                vocabulary_sizes=[len(categories) + 1 for categories in preprocessing["categorical_encoder"].categories_],
                model_config=model_config,
                learning_rate=float(frozen_config["training"]["learning_rate"]),
            )
            model.load_weights(_validate_frozen_input(freeze_root, f"models/seed{seed}.weights.h5", manifest))
            validation_probabilities = model.predict(validation.inputs, verbose=0).reshape(-1)
            search_before_test = scan_thresholds(
                validation.labels,
                validation_probabilities,
                minimum=float(search_config["minimum"]),
                maximum=float(search_config["maximum"]),
                step=float(search_config["step"]),
            )
            select_validation_thresholds(search_before_test, far_limit=float(search_config["far_limit"]))
            test_probabilities = model.predict(test.inputs, verbose=0).reshape(-1)
            np.savez_compressed(
                cache_path,
                validation_probabilities=validation_probabilities.astype(np.float32),
                validation_labels=validation.labels.astype(np.int8),
                test_probabilities=test_probabilities.astype(np.float32),
                test_labels=test.labels.astype(np.int8),
            )
            prediction_source = "frozen_weights_single_validation_and_test_inference"

        search = scan_thresholds(
            validation.labels,
            validation_probabilities,
            minimum=float(search_config["minimum"]),
            maximum=float(search_config["maximum"]),
            step=float(search_config["step"]),
        )
        locked_selection = select_validation_thresholds(search, far_limit=float(search_config["far_limit"]))
        search.to_csv(output_root / f"validation_threshold_search_seed{seed}.csv", index=False)
        _plot_validation_search(search, locked_selection, plots_dir / f"validation_threshold_search_seed{seed}.png", seed)
        for policy, values in locked_selection.items():
            validation_selection_rows.append({"seed": seed, "policy": policy, **values})
        seed_test_rows = evaluate_locked_thresholds(test.labels, test_probabilities, locked_selection)
        for row in seed_test_rows:
            test_rows.append({"seed": seed, **row})
        seed_documents.append(
            {
                "seed": seed,
                "prediction_source": prediction_source,
                "validation_selection": locked_selection,
                "test_results": seed_test_rows,
                "test_inference_passes": 1,
            }
        )

    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(test_comparison_path, index=False)
    _plot_test_comparison(test_frame, plots_dir / "test_policy_comparison.png")
    _write_markdown_comparison(markdown_path, validation_selection_rows, test_rows)
    final_json_path.write_text(
        json.dumps(
            {
                "run_name": config["run"]["name"],
                "frozen_baseline_id": manifest["baseline_id"],
                "frozen_manifest_sha256": sha256_file(manifest_path),
                "selection_scope": "validation_only",
                "test_used_for_threshold_selection": False,
                "test_evaluated_after_threshold_lock": True,
                "threshold_grid": search_config,
                "seed_results": seed_documents,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"阶段结果: validation 阈值已锁定，test 未参与选择，输出 {output_root}")
    return {"final_json": final_json_path, "test_comparison": test_comparison_path, "comparison_markdown": markdown_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize frozen baseline thresholds on validation only")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"阶段失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
