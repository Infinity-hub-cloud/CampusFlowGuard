"""Evaluate classical model baselines against the frozen CampusFlowGuard split.

The experiment is deliberately independent of the frozen Transformer and Focal
artifacts. It uses the same source rows, partition-local windows, train-only
preprocessing, validation-only threshold selection, and one final test pass.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from campus_flow_guard.data_protocol import (
    SplitIndices,
    assert_split_integrity,
    build_partition_window_indices,
    load_selected_rows,
    load_split_manifest,
    sha256_file,
)


METRICS = ["precision", "recall", "f1", "far", "balanced_accuracy", "roc_auc", "pr_auc"]


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _window_features(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    benign_label: str,
    numeric_pipeline: Any,
    categorical_encoder: Any,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_index()
    windows, targets = build_partition_window_indices(indices, window_size)
    partition_set = set(ordered.index.to_numpy(dtype=np.int64).tolist())
    if any(not set(window.tolist()) <= partition_set for window in windows):
        raise RuntimeError("A generated window crossed a partition boundary")
    numeric = numeric_pipeline.transform(ordered[numeric_columns]).astype(np.float32)
    categorical = (
        categorical_encoder.transform(ordered[categorical_columns].astype(str)).astype(np.float32) + 1.0
    )
    record_features = np.column_stack((numeric, categorical)).astype(np.float32)
    raw_windows = np.lib.stride_tricks.sliding_window_view(
        record_features, window_size, axis=0
    )
    flattened = np.swapaxes(raw_windows, 1, 2).reshape(len(targets), -1).copy()
    labels = ordered[label_column].ne(benign_label).to_numpy(dtype=np.int32)
    return flattened, labels[window_size - 1 :], targets


def _validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(values))])


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int32)
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
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def _build_model(name: str, config: dict[str, Any], seed: int) -> Any:
    model_config = _section(_section(config, "models"), name)
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=int(model_config["max_iter"]),
            C=float(model_config["c"]),
            solver=str(model_config["solver"]),
            class_weight=str(model_config["class_weight"]),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_config["n_estimators"]),
            max_depth=int(model_config["max_depth"]),
            min_samples_leaf=int(model_config["min_samples_leaf"]),
            max_features=str(model_config["max_features"]),
            class_weight=str(model_config["class_weight"]),
            n_jobs=int(model_config["n_jobs"]),
            random_state=seed,
        )
    raise ValueError(f"Unknown classical model: {name}")


def _plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    models = summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.12
    metrics = ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "Balanced accuracy", "ROC-AUC", "PR-AUC"]
    colors = ["#2f6690", "#3a7d44", "#d17a22", "#6a4c93", "#4d908e", "#bc4749"]
    fig, axis = plt.subplots(figsize=(11, 5.5), dpi=300)
    for offset, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = summary[metric].to_numpy(dtype=float)
        errors = summary[f"{metric}_std"].to_numpy(dtype=float)
        axis.bar(x + (offset - 2.5) * width, values, width, yerr=errors, capsize=2,
                 label=label, color=color, edgecolor="white", linewidth=0.5)
    axis.set_xticks(x, models)
    plotted_values = summary[metrics].to_numpy(dtype=float)
    lower_bound = max(0.0, float(np.nanmin(plotted_values)) - 0.04)
    axis.set_ylim(lower_bound, 1.01)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.6), dpi=300)
    far = summary["far"].to_numpy(dtype=float) * 100.0
    far_std = summary["far_std"].to_numpy(dtype=float) * 100.0
    bars = axis.bar(models, far, yerr=far_std, capsize=3, color=["#2f6690", "#bc4749", "#d17a22", "#6a4c93"],
                    edgecolor="white", linewidth=0.5)
    axis.set_ylabel("False alarm rate (%)")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, far):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}%",
                  ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    far_output = output.with_name(output.stem + "_far")
    fig.savefig(far_output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(far_output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _write_report(path: Path, *, source_rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# 经典模型对照实验报告",
        "",
        "本实验为新增对照实验，不修改冻结的 BCE/Focal Loss、split manifest、阈值规则或历史归档。",
        "",
        "## 协议",
        "",
        "- 数据：NF-UNSW-NB15-v2，固定 120,000 行实验池；split manifest 与可信 baseline 相同。",
        "- train/validation/test 原始行数：96,000/12,000/12,000；窗口长度 8，窗口在分区内构造。",
        "- 预处理仅在 train 拟合；模型和阈值选择只使用 train/validation；test 每个 seed 仅最终评估一次。",
        "- 经典模型输入为同一 8 行窗口的展开特征，使用 class_weight 处理 train 类别不平衡。",
        "- 由于数据没有时间字段，本实验不声称时间外推能力。",
        "",
        "## 模型定位",
        "",
        "- Logistic Regression：经典线性判别基线（1958）。",
        "- Random Forest：经典非线性树集成基线（2001）。",
        "- BCE 与 Focal Loss：来自已有同结构 Transformer 实验，作为主要 loss 对照。",
        "",
        "## 三 seed 均值 ± 标准差",
        "",
        "| 模型 | Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['precision']:.4f} ± {row['precision_std']:.4f} | "
            f"{row['recall']:.4f} ± {row['recall_std']:.4f} | {row['f1']:.4f} ± {row['f1_std']:.4f} | "
            f"{row['far']:.4f} ± {row['far_std']:.4f} | {row['balanced_accuracy']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f} | {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "Focal Loss 相对于 BCE 的直接改进仍由同一 Transformer 结构下的 loss 对照支持。Logistic Regression 和 Random Forest 改变了模型结构，结果用于说明整体方案相对于传统模型的表现，不应把其差异归因于 Focal Loss 单一因素。",
        "",
        "## 原始结果",
        "",
        "完整逐 seed 结果见 `classical_by_seed.csv`；汇总表见 `classical_aggregate.csv`。BCE/Focal 来源为既有冻结实验目录，经典模型结果来源为本次新增实验。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_yaml(config_path)
    root = config_path.parents[2]
    run_config = _section(config, "run")
    data_config = _section(config, "data")
    output_config = _section(config, "outputs")
    output = (root / str(output_config["directory"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "experiment_config.yaml")

    frozen = (root / str(run_config["frozen_baseline_dir"])).resolve()
    split_manifest_path = frozen / "splits" / "split_manifest.json"
    split_npz_path = frozen / "splits" / "split_indices.npz"
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split = load_split_manifest(split_npz_path, expected_pool_size=int(manifest["pool_size"]))
    assert_split_integrity(split, expected_pool_size=int(manifest["pool_size"]))
    if manifest.get("dataset_sha256") != sha256_file(Path(str(data_config["csv_path"]))):
        raise ValueError("Current dataset does not match frozen baseline SHA-256")

    schema = _read_yaml((root / str(data_config["schema_path"])).resolve())
    features = [str(value) for value in schema["feature_columns"]]
    categorical = [str(value) for value in schema["categorical_columns"]]
    numeric = [value for value in features if value not in categorical]
    label_column = str(schema["label_column"])
    benign_label = str(schema["benign_label"])
    csv_path = Path(str(data_config["csv_path"])).resolve()
    selected = load_selected_rows(
        csv_path,
        columns=features + [label_column],
        selected_indices=split.all_indices,
        chunk_size=int(data_config["read_chunk_size"]),
    )
    preprocessing_path = frozen / "models" / "preprocessing.pkl"
    with preprocessing_path.open("rb") as handle:
        preprocessing = pickle.load(handle)
    numeric_pipeline = preprocessing["numeric_pipeline"]
    categorical_encoder = preprocessing["categorical_encoder"]

    partitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        partitions[name] = _window_features(
            selected.loc[indices], indices,
            numeric_columns=numeric, categorical_columns=categorical,
            label_column=label_column, benign_label=benign_label,
            numeric_pipeline=numeric_pipeline, categorical_encoder=categorical_encoder,
            window_size=int(data_config["window_size"]),
        )[:2]
    del selected
    X_train, y_train = partitions["train"]
    X_validation, y_validation = partitions["validation"]
    X_test, y_test = partitions["test"]
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    model_names = ["logistic_regression", "random_forest"]
    rows: list[dict[str, Any]] = []
    logs = output / "logs"
    metrics_dir = output / "metrics"
    logs.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    for model_name in model_names:
        for seed_value in run_config["model_seeds"]:
            seed = int(seed_value)
            _set_seed(seed)
            model = _build_model(model_name, config, seed)
            if model_name == "logistic_regression":
                fit_train, fit_validation, fit_test = X_train_scaled, X_validation_scaled, X_test_scaled
            else:
                fit_train, fit_validation, fit_test = X_train, X_validation, X_test
            started = time.monotonic()
            model.fit(fit_train, y_train)
            training_seconds = time.monotonic() - started
            validation_probabilities = model.predict_proba(fit_validation)[:, 1]
            threshold = _validation_threshold(y_validation, validation_probabilities)
            inference_started = time.perf_counter()
            test_probabilities = model.predict_proba(fit_test)[:, 1]
            inference_seconds = time.perf_counter() - inference_started
            result = _metrics(y_test, test_probabilities, threshold)
            result.update({
                "model": model_name,
                "seed": seed,
                "parameter_count": int(getattr(model, "n_estimators", 0)) if model_name == "random_forest" else int(X_train.shape[1]),
                "training_seconds": float(training_seconds),
                "inference_seconds": float(inference_seconds),
                "inference_throughput_samples_per_second": float(len(y_test) / inference_seconds),
                "test_sample_count": int(len(y_test)),
                "validation_threshold": float(threshold),
            })
            rows.append(result)
            (metrics_dir / f"{model_name}_seed{seed}.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            (logs / f"{model_name}_seed{seed}.log").write_text(
                f"model={model_name}\nseed={seed}\ntraining_seconds={training_seconds:.6f}\n"
                f"validation_threshold={threshold:.9f}\ntest_sample_count={len(y_test)}\n",
                encoding="utf-8",
            )
            print(f"result model={model_name} seed={seed} F1={result['f1']:.4f} FAR={result['far']:.4f}")

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "classical_by_seed.csv", index=False)
    aggregate_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        frame = by_seed[by_seed["model"] == model_name]
        aggregate_rows.append({
            "model": "Logistic Regression" if model_name == "logistic_regression" else "Random Forest",
            **{metric: float(frame[metric].mean()) for metric in METRICS},
            **{f"{metric}_std": float(frame[metric].std(ddof=1)) for metric in METRICS},
        })
    focal_aggregate = root / "artifacts" / "experiments" / "focal_loss" / "unsw_focal_loss_v1" / "metrics" / "bce_vs_focal_aggregate.csv"
    focal_frame = pd.read_csv(focal_aggregate)
    for loss_name, label in (("BCE", "BCE Transformer"), ("Focal", "Focal Transformer")):
        row = focal_frame[(focal_frame["loss"] == loss_name) & (focal_frame["stat"] == "mean")].iloc[0]
        std = focal_frame[(focal_frame["loss"] == loss_name) & (focal_frame["stat"] == "std")].iloc[0]
        aggregate_rows.append({"model": label, **{metric: float(row[metric]) for metric in METRICS},
                               **{f"{metric}_std": float(std[metric]) for metric in METRICS}})
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate = aggregate[["model", *METRICS, *(f"{metric}_std" for metric in METRICS)]]
    aggregate.to_csv(output / "classical_aggregate.csv", index=False)
    _plot_comparison(aggregate, output / "model_comparison_metrics")
    _write_report(output / "classical_baselines_report.md", source_rows=by_seed, summary=aggregate)
    (output / "protocol.json").write_text(json.dumps({
        "frozen_baseline": str(frozen),
        "split_manifest": str(split_manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "window_size": int(data_config["window_size"]),
        "train_rows": int(len(split.train)), "validation_rows": int(len(split.validation)), "test_rows": int(len(split.test)),
        "train_windows": int(len(y_train)), "validation_windows": int(len(y_validation)), "test_windows": int(len(y_test)),
        "threshold_selection": "validation_only_max_f1", "test_evaluation": "one_pass_per_seed",
    }, indent=2) + "\n", encoding="utf-8")
    return {"output": output, "aggregate": aggregate, "by_seed": by_seed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classical comparison baselines")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
