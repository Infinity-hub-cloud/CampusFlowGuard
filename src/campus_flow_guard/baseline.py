"""Run the resource-bounded trusted CampusFlowGuard baseline.

来源说明：本文件是参考 FlowTransformer 上游源代码后的自主改写。
Upstream: https://github.com/liamdm/FlowTransformer.git
Upstream copyright: FlowTransformer 2023 by liamdm / liam@riftcs.com
License: GNU AGPL-3.0; see the repository LICENSE and NOTICE.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=r"urllib3 .*doesn't match a supported version!")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

from campus_flow_guard.data_protocol import (
    SplitIndices,
    assert_split_integrity,
    build_partition_window_indices,
    create_stratified_split,
    load_selected_rows,
    load_split_manifest,
    save_split_manifest,
    sha256_file,
)


@dataclass(frozen=True)
class WindowedPartition:
    numeric: np.ndarray
    categorical: list[np.ndarray]
    labels: np.ndarray
    source_targets: np.ndarray

    @property
    def inputs(self) -> tuple[np.ndarray, ...]:
        return (self.numeric, *self.categorical)


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return loaded


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (RuntimeError, AttributeError):
        pass


def _make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"campus_flow_guard.baseline.{path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _load_or_create_split(
    *,
    labels: np.ndarray,
    config: dict[str, Any],
    csv_path: Path,
    schema_path: Path,
    split_npz_path: Path,
    split_json_path: Path,
    data_fingerprint_path: Path,
) -> tuple[SplitIndices, dict[str, Any]]:
    run_config = _section(config, "run")
    data_config = _section(config, "data")
    dataset_sha256 = sha256_file(csv_path)
    schema_sha256 = sha256_file(schema_path)
    metadata = {
        "dataset_path": str(csv_path.resolve()),
        "dataset_size_bytes": int(csv_path.stat().st_size),
        "dataset_sha256": dataset_sha256,
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": schema_sha256,
        "dataset_row_count": int(len(labels)),
        "pool_size": int(data_config["pool_size"]),
        "split_seed": int(run_config["split_seed"]),
        "fractions": {
            "train": float(data_config["train_fraction"]),
            "validation": float(data_config["validation_fraction"]),
            "test": float(data_config["test_fraction"]),
        },
        "source_and_license_status": "待人工确认",
        "time_field_present": False,
        "time_extrapolation_claimed": False,
    }
    data_fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    data_fingerprint_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if split_npz_path.exists() != split_json_path.exists():
        raise FileNotFoundError("Split NPZ and JSON manifest must either both exist or both be absent")
    if split_npz_path.exists():
        existing = json.loads(split_json_path.read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "schema_sha256", "pool_size", "split_seed"):
            if existing.get(key) != metadata[key]:
                raise ValueError(f"Existing split manifest does not match current {key}")
        return load_split_manifest(
            split_npz_path,
            expected_pool_size=int(data_config["pool_size"]),
        ), metadata

    split = create_stratified_split(
        labels,
        pool_size=int(data_config["pool_size"]),
        train_fraction=float(data_config["train_fraction"]),
        validation_fraction=float(data_config["validation_fraction"]),
        test_fraction=float(data_config["test_fraction"]),
        seed=int(run_config["split_seed"]),
    )
    save_split_manifest(
        split,
        npz_path=split_npz_path,
        json_path=split_json_path,
        metadata=metadata,
        labels=labels,
    )
    return split, metadata


def _build_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
    return np.swapaxes(windows, 1, 2).copy()


def _prepare_partition(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    benign_label: str,
    numeric_pipeline: Pipeline,
    categorical_encoder: OrdinalEncoder,
    window_size: int,
) -> WindowedPartition:
    ordered = frame.sort_index()
    source_windows, source_targets = build_partition_window_indices(
        ordered.index.to_numpy(dtype=np.int64), window_size
    )
    partition_source_set = set(ordered.index.to_numpy(dtype=np.int64).tolist())
    if any(not set(window.tolist()) <= partition_source_set for window in source_windows):
        raise RuntimeError("A generated window crossed its partition boundary")

    numeric = numeric_pipeline.transform(ordered[numeric_columns]).astype(np.float32)
    categorical = (
        categorical_encoder.transform(ordered[categorical_columns].astype(str)).astype(np.int32) + 1
    )
    labels = ordered[label_column].ne(benign_label).to_numpy(dtype=np.int32)
    return WindowedPartition(
        numeric=_build_windows(numeric, window_size),
        categorical=[
            _build_windows(categorical[:, index : index + 1], window_size).squeeze(-1)
            for index in range(categorical.shape[1])
        ],
        labels=labels[window_size - 1 :],
        source_targets=source_targets,
    )


def _build_record_projection_model(
    *,
    window_size: int,
    numeric_feature_count: int,
    vocabulary_sizes: list[int],
    model_config: dict[str, Any],
    learning_rate: float,
) -> keras.Model:
    internal_size = int(model_config["internal_size"])
    num_heads = int(model_config["num_heads"])
    if internal_size % num_heads:
        raise ValueError("model.internal_size must be divisible by model.num_heads")

    numeric_input = keras.Input(
        shape=(window_size, numeric_feature_count), dtype="float32", name="numeric_features"
    )
    inputs: list[Any] = [numeric_input]
    record_features: list[Any] = [numeric_input]
    for index, vocabulary_size in enumerate(vocabulary_sizes):
        categorical_input = keras.Input(
            shape=(window_size,), dtype="int32", name=f"categorical_{index}"
        )
        inputs.append(categorical_input)
        record_features.append(
            layers.Embedding(
                input_dim=int(vocabulary_size),
                output_dim=int(model_config["categorical_embedding_dim"]),
                name=f"categorical_embedding_{index}",
            )(categorical_input)
        )

    x = layers.Concatenate(name="record_features")(record_features)
    x = layers.Dense(internal_size, activation="relu", name="record_projection")(x)
    for index in range(int(model_config["transformer_layers"])):
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=internal_size // num_heads,
            dropout=float(model_config["dropout"]),
            name=f"transformer_attention_{index}",
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"attention_norm_{index}")(x + attention)
        feed_forward = layers.Dense(
            internal_size * 2, activation="relu", name=f"ffn_expand_{index}"
        )(x)
        feed_forward = layers.Dropout(
            float(model_config["dropout"]), name=f"ffn_dropout_{index}"
        )(feed_forward)
        feed_forward = layers.Dense(internal_size, name=f"ffn_project_{index}")(feed_forward)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_norm_{index}")(
            x + feed_forward
        )

    x = layers.Lambda(lambda tensor: tensor[:, -1, :], name="last_token")(x)
    x = layers.Dense(int(model_config["mlp_units"]), activation="relu", name="classification_mlp")(x)
    x = layers.Dropout(float(model_config["dropout"]), name="classification_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="malicious_probability")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="trusted_record_projection_transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


class BalancedSequence(keras.utils.Sequence):
    """Deterministic train-only balanced target sampling."""

    def __init__(
        self,
        inputs: tuple[np.ndarray, ...],
        labels: np.ndarray,
        *,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self.epoch = 0
        self.positive_indices = np.flatnonzero(labels == 1)
        self.negative_indices = np.flatnonzero(labels == 0)
        if not len(self.positive_indices) or not len(self.negative_indices):
            raise ValueError("Balanced sampling requires both classes in train")

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, batch_index: int) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        rng = np.random.default_rng(self.seed + self.epoch * self.steps + batch_index)
        positive_count = self.batch_size // 2
        indices = np.concatenate(
            (
                rng.choice(self.positive_indices, positive_count, replace=True),
                rng.choice(self.negative_indices, self.batch_size - positive_count, replace=True),
            )
        )
        rng.shuffle(indices)
        return tuple(array[indices] for array in self.inputs), self.labels[indices]

    def on_epoch_end(self) -> None:
        self.epoch += 1


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("class_weight requires both classes in train")
    total = float(counts.sum())
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


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


def _validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def _fit_model(
    model: keras.Model,
    *,
    method: str,
    train: WindowedPartition,
    validation: WindowedPartition,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
    callbacks: list[keras.callbacks.Callback],
) -> keras.callbacks.History:
    common = {
        "validation_data": (validation.inputs, validation.labels),
        "epochs": epochs,
        "verbose": 0,
        "callbacks": callbacks,
    }
    if method == "balanced_sampling":
        sequence = BalancedSequence(
            train.inputs,
            train.labels,
            batch_size=batch_size,
            steps=steps_per_epoch,
            seed=seed,
        )
        return model.fit(sequence, **common)
    if method not in {"none", "class_weight"}:
        raise ValueError(f"Unknown training method: {method}")
    return model.fit(
        train.inputs,
        train.labels,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        shuffle=True,
        class_weight=_class_weights(train.labels) if method == "class_weight" else None,
        **common,
    )


def _plot_confusion(metrics: dict[str, Any], path: Path, title: str) -> None:
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], ["Benign", "Attack"])
    axis.set_yticks([0, 1], ["Benign", "Attack"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_curves(labels: np.ndarray, probabilities: np.ndarray, roc_path: Path, pr_path: Path, title: str) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC={roc_auc_score(labels, probabilities):.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.plot(recall, precision, label=f"AP={average_precision_score(labels, probabilities):.4f}")
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(pr_path, dpi=160)
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    comparison_results: list[dict[str, Any]],
    selected_method: str,
    seed_results: list[dict[str, Any]],
    summary: pd.DataFrame,
) -> None:
    metric_columns = ["precision", "recall", "f1", "far", "balanced_accuracy", "roc_auc", "pr_auc", "training_seconds", "inference_throughput_samples_per_second"]
    means = summary.loc[summary["seed"] == "mean"].iloc[0]
    standard_deviations = summary.loc[summary["seed"] == "std"].iloc[0]
    lines = [
        "# CampusFlowGuard 资源受限可信 baseline 报告",
        "",
        "## 结论边界",
        "",
        "本报告来自真实 `NF-UNSW-NB15-v2.csv` 的固定 120,000 行分层实验池，不是全量 FlowTransformer 论文复现。数据没有时间字段，不声称时间外推。数据来源、许可证和发布方哈希均为**待人工确认**。",
        "",
        "## 数据协议",
        "",
        f"- 数据 SHA-256：`{metadata['dataset_sha256']}`。",
        f"- split seed：`{metadata['split_seed']}`；train/validation/test = 80%/10%/10%。",
        "- 先划分行索引，再分别在三个集合内按原始行号排序构造 8 行窗口，目标为最后一行。",
        "- 数值和类别预处理器只在 train 上拟合；validation 用于方案、最佳 epoch 和阈值选择；test 只用于最终评估。",
        "",
        "## 三种训练方式小试（validation，固定阈值 0.5）",
        "",
        "| 方法 | TP | FP | TN | FN | Precision | Recall | F1 | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in comparison_results:
        lines.append(
            f"| {result['method']} | {result['TP']} | {result['FP']} | {result['TN']} | {result['FN']} | "
            f"{result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {result['pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"选择方案：`{selected_method}`。选择过程未读取 test 指标。类别平衡属于 baseline 训练处理，不作为创新。",
            "",
            "## 主 baseline 三 seed 测试结果",
            "",
            "| Seed | TP | FP | TN | FN | Precision | Recall | F1 | FAR | Bal Acc | ROC-AUC | PR-AUC |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in seed_results:
        lines.append(
            f"| {result['seed']} | {result['TP']} | {result['FP']} | {result['TN']} | {result['FN']} | "
            f"{result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {result['far']:.4f} | "
            f"{result['balanced_accuracy']:.4f} | {result['roc_auc']:.4f} | {result['pr_auc']:.4f} |"
        )
    lines.extend(["", "## 均值与标准差", ""])
    for column in metric_columns:
        lines.append(f"- `{column}`：{means[column]:.6f} ± {standard_deviations[column]:.6f}")
    lines.extend(
        [
            "",
            "## 模型与输出",
            "",
            "主模型固定为 Record Projection、2-layer Transformer、2 heads、internal size 128、Last Token、window size 8。每个 seed 保存 validation PR-AUC 最佳权重、训练日志、阈值、混淆矩阵、ROC 和 PR 图。完整数值见 `artifacts/metrics/baseline_summary.csv`。",
            "",
            "## 尚未解决",
            "",
            "上游来源和许可证待人工确认；当前仅 CPU 运行；120,000 行资源受限实验池不代表全量结果；随机分层窗口中的相邻行可能在原 CSV 中有间隔。进入创新实验时必须保持同一 split manifest 和评估协议。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _read_yaml(config_path)
    project_root = config_path.parents[2]
    run_config = _section(config, "run")
    data_config = _section(config, "data")
    model_config = _section(config, "model")
    comparison_config = _section(config, "comparison")
    training_config = _section(config, "training")
    output_config = _section(config, "outputs")

    artifacts = (project_root / str(output_config["artifacts_dir"])).resolve()
    logs = (project_root / str(output_config["logs_dir"])).resolve()
    directories = {
        "metrics": artifacts / "metrics",
        "models": artifacts / "models",
        "configs": artifacts / "configs",
        "splits": artifacts / "splits",
        "metadata": artifacts / "metadata",
        "plots": artifacts / "plots",
        "reports": artifacts / "reports",
        "logs": logs,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    run_name = str(run_config["name"])
    summary_path = directories["metrics"] / "baseline_summary.csv"
    report_path = directories["reports"] / "trusted_baseline_report_zh.md"
    comparison_path = directories["metrics"] / f"{run_name}_balance_comparison.json"
    config_copy_path = directories["configs"] / f"{run_name}.yaml"
    split_npz_path = directories["splits"] / f"{run_name}_split_indices.npz"
    split_json_path = directories["splits"] / f"{run_name}_split_manifest.json"
    fingerprint_path = directories["metadata"] / f"{run_name}_data_fingerprint.json"
    for output_path in (summary_path, report_path, comparison_path):
        if output_path.exists():
            raise FileExistsError(f"Trusted baseline output already exists: {output_path}")
    shutil.copy2(config_path, config_copy_path)

    csv_path = Path(str(data_config["csv_path"])).expanduser().resolve()
    schema_path = (project_root / str(data_config["schema_path"])).resolve()
    schema = _read_yaml(schema_path)
    feature_columns = [str(column) for column in schema["feature_columns"]]
    categorical_columns = [str(column) for column in schema["categorical_columns"]]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    label_column = str(schema["label_column"])
    benign_label = str(schema["benign_label"])

    print("阶段结果: 读取真实标签并准备 strict split")
    labels = (
        pd.read_csv(csv_path, usecols=[label_column])[label_column]
        .ne(benign_label)
        .to_numpy(dtype=np.int8)
    )
    split, metadata = _load_or_create_split(
        labels=labels,
        config=config,
        csv_path=csv_path,
        schema_path=schema_path,
        split_npz_path=split_npz_path,
        split_json_path=split_json_path,
        data_fingerprint_path=fingerprint_path,
    )
    assert_split_integrity(split, expected_pool_size=int(data_config["pool_size"]))
    selected_frame = load_selected_rows(
        csv_path,
        columns=feature_columns + [label_column],
        selected_indices=split.all_indices,
        chunk_size=int(data_config["read_chunk_size"]),
    )
    for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        observed = selected_frame.loc[indices, label_column].ne(benign_label).to_numpy(dtype=np.int8)
        if not np.array_equal(observed, labels[indices]):
            raise RuntimeError(f"Loaded {name} labels differ from split-time labels")

    train_frame = selected_frame.loc[split.train]
    validation_frame = selected_frame.loc[split.validation]
    test_frame = selected_frame.loc[split.test]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    numeric_pipeline.fit(train_frame[numeric_columns])
    categorical_encoder.fit(train_frame[categorical_columns].astype(str))
    preprocessing_path = directories["models"] / f"{run_name}_preprocessing.pkl"
    with preprocessing_path.open("wb") as handle:
        pickle.dump(
            {
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "numeric_pipeline": numeric_pipeline,
                "categorical_encoder": categorical_encoder,
            },
            handle,
        )

    window_size = int(model_config["window_size"])
    train = _prepare_partition(
        train_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_pipeline=numeric_pipeline,
        categorical_encoder=categorical_encoder,
        window_size=window_size,
    )
    validation = _prepare_partition(
        validation_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_pipeline=numeric_pipeline,
        categorical_encoder=categorical_encoder,
        window_size=window_size,
    )
    test = _prepare_partition(
        test_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_pipeline=numeric_pipeline,
        categorical_encoder=categorical_encoder,
        window_size=window_size,
    )
    vocabulary_sizes = [len(categories) + 1 for categories in categorical_encoder.categories_]
    del selected_frame, train_frame, validation_frame, test_frame, labels

    print("阶段结果: 对比 none、class_weight、balanced_sampling（仅 validation）")
    comparison_results: list[dict[str, Any]] = []
    for method in [str(value) for value in comparison_config["methods"]]:
        keras.backend.clear_session()
        seed = int(run_config["selection_seed"])
        _set_seed(seed)
        model = _build_record_projection_model(
            window_size=window_size,
            numeric_feature_count=len(numeric_columns),
            vocabulary_sizes=vocabulary_sizes,
            model_config=model_config,
            learning_rate=float(training_config["learning_rate"]),
        )
        comparison_log = directories["logs"] / f"{run_name}_compare_{method}.csv"
        started = time.monotonic()
        _fit_model(
            model,
            method=method,
            train=train,
            validation=validation,
            epochs=int(comparison_config["epochs"]),
            batch_size=int(comparison_config["batch_size"]),
            steps_per_epoch=int(comparison_config["steps_per_epoch"]),
            seed=seed,
            callbacks=[keras.callbacks.CSVLogger(comparison_log)],
        )
        validation_probabilities = model.predict(validation.inputs, verbose=0).reshape(-1)
        result = {
            "method": method,
            "training_seconds": float(time.monotonic() - started),
            **_metrics(
                validation.labels,
                validation_probabilities,
                float(comparison_config["threshold"]),
            ),
        }
        comparison_results.append(result)
        print(f"阶段结果: {method} validation TP={result['TP']} F1={result['f1']:.4f} PR-AUC={result['pr_auc']:.4f}")

    viable = [result for result in comparison_results if result["TP"] > 0]
    if not viable:
        comparison_path.write_text(
            json.dumps({"results": comparison_results, "selected_method": None}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("No balancing method produced non-zero validation TP at threshold 0.5")
    selected = max(viable, key=lambda result: (result["f1"], result["pr_auc"]))
    selected_method = str(selected["method"])
    comparison_path.write_text(
        json.dumps(
            {
                "selection_scope": "validation_only_fixed_threshold_0.5",
                "results": comparison_results,
                "selected_method": selected_method,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"阶段结果: 选择训练方式 {selected_method}")

    seed_results: list[dict[str, Any]] = []
    for seed_value in run_config["model_seeds"]:
        seed = int(seed_value)
        keras.backend.clear_session()
        _set_seed(seed)
        model = _build_record_projection_model(
            window_size=window_size,
            numeric_feature_count=len(numeric_columns),
            vocabulary_sizes=vocabulary_sizes,
            model_config=model_config,
            learning_rate=float(training_config["learning_rate"]),
        )
        weights_path = directories["models"] / f"{run_name}_seed{seed}.weights.h5"
        summary_path_for_seed = directories["models"] / f"{run_name}_seed{seed}_summary.txt"
        log_path = directories["logs"] / f"{run_name}_seed{seed}.csv"
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        summary_path_for_seed.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.CSVLogger(log_path),
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=int(training_config["early_stopping_patience"]),
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                weights_path,
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
        training_started = time.monotonic()
        history = _fit_model(
            model,
            method=selected_method,
            train=train,
            validation=validation,
            epochs=int(training_config["max_epochs"]),
            batch_size=int(training_config["batch_size"]),
            steps_per_epoch=int(training_config["steps_per_epoch"]),
            seed=seed,
            callbacks=callbacks,
        )
        training_seconds = time.monotonic() - training_started
        model.load_weights(weights_path)
        validation_probabilities = model.predict(validation.inputs, verbose=0).reshape(-1)
        threshold = _validation_threshold(validation.labels, validation_probabilities)
        validation_metrics = _metrics(validation.labels, validation_probabilities, threshold)

        inference_started = time.perf_counter()
        test_probabilities = model.predict(test.inputs, verbose=0).reshape(-1)
        inference_seconds = time.perf_counter() - inference_started
        test_metrics = _metrics(test.labels, test_probabilities, threshold)
        test_metrics.update(
            {
                "seed": seed,
                "selected_method": selected_method,
                "parameter_count": int(model.count_params()),
                "training_seconds": float(training_seconds),
                "inference_seconds": float(inference_seconds),
                "inference_throughput_samples_per_second": float(len(test.labels) / inference_seconds),
                "epochs_completed": int(len(history.history["loss"])),
                "best_epoch": int(np.argmax(history.history["val_pr_auc"]) + 1),
                "validation_f1_at_selected_threshold": float(validation_metrics["f1"]),
                "test_sample_count": int(len(test.labels)),
            }
        )
        seed_metrics_path = directories["metrics"] / f"{run_name}_seed{seed}.json"
        seed_metrics_path.write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _plot_confusion(
            test_metrics,
            directories["plots"] / f"{run_name}_seed{seed}_confusion.png",
            f"Test confusion matrix, seed {seed}",
        )
        _plot_curves(
            test.labels,
            test_probabilities,
            directories["plots"] / f"{run_name}_seed{seed}_roc.png",
            directories["plots"] / f"{run_name}_seed{seed}_pr.png",
            f"Test curves, seed {seed}",
        )
        seed_results.append(test_metrics)
        print(
            f"阶段结果: seed={seed} TP={test_metrics['TP']} F1={test_metrics['f1']:.4f} "
            f"ROC-AUC={test_metrics['roc_auc']:.4f} PR-AUC={test_metrics['pr_auc']:.4f}"
        )

    seed_frame = pd.DataFrame(seed_results)
    numeric_summary_columns = [
        "TP", "FP", "TN", "FN", "precision", "recall", "f1", "far",
        "balanced_accuracy", "roc_auc", "pr_auc", "parameter_count",
        "training_seconds", "inference_throughput_samples_per_second",
    ]
    mean_row = {"seed": "mean", **seed_frame[numeric_summary_columns].mean().to_dict()}
    std_row = {"seed": "std", **seed_frame[numeric_summary_columns].std(ddof=1).to_dict()}
    summary = pd.concat([seed_frame, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    _write_report(
        report_path,
        metadata=metadata,
        comparison_results=comparison_results,
        selected_method=selected_method,
        seed_results=seed_results,
        summary=summary,
    )
    print(f"阶段结果: 汇总已保存 {summary_path}")
    return {
        "selected_method": selected_method,
        "comparison_results": comparison_results,
        "seed_results": seed_results,
        "summary_path": summary_path,
        "report_path": report_path,
        "split_manifest_path": split_json_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict CampusFlowGuard trusted baseline")
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
