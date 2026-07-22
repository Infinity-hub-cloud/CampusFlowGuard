"""Minimal, traceable FlowTransformer-style smoke training entry point.

来源说明：本文件是参考 FlowTransformer 上游源代码后的自主改写，用于验证
CampusFlowGuard 数据处理、窗口、模型、训练和评估闭环，不声称原始架构或
FlowTransformer 实现思想为本项目原创。

Upstream: https://github.com/liamdm/FlowTransformer.git
Upstream copyright: FlowTransformer 2023 by liamdm / liam@riftcs.com
License: GNU AGPL-3.0; see the repository LICENSE and NOTICE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
# The environment diagnosis records this requests/urllib3 compatibility warning.
# Suppress it here so successful smoke-run terminal output remains machine-readable.
warnings.filterwarnings("ignore", message=r"urllib3 .* doesn't match a supported version!")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class PreparedSplit:
    """Encoded, windowed examples for one data partition."""

    numeric_windows: np.ndarray
    categorical_windows: list[np.ndarray]
    labels: np.ndarray


def _read_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (RuntimeError, AttributeError):
        # Some TensorFlow builds do not expose deterministic kernels for every op.
        pass


def _load_schema(path: Path) -> dict[str, Any]:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"Schema root must be a mapping: {path}")
    return schema


def _read_smoke_rows(
    csv_path: Path,
    columns: list[str],
    start_row: int,
    max_rows: int,
) -> pd.DataFrame:
    if start_row < 0 or max_rows <= 0:
        raise ValueError("data.start_row must be non-negative and data.max_rows must be positive")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Configured CSV does not exist: {csv_path}")

    # Keep the CSV header and skip only preceding data rows.
    return pd.read_csv(
        csv_path,
        usecols=columns,
        skiprows=range(1, start_row + 1),
        nrows=max_rows,
    )


def _split_contiguous(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1")

    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
    partitions = (frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:])
    if min(len(partition) for partition in partitions) == 0:
        raise ValueError("Smoke subset is too small for the configured contiguous split")
    return partitions


def _build_windows(values: np.ndarray, labels: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build target-ending windows within one partition only."""

    if window_size < 2:
        raise ValueError("window.size must be at least 2")
    if len(values) < window_size:
        raise ValueError("A partition must contain at least window.size rows")
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size, axis=0)
    # sliding_window_view returns (examples, features, sequence); Keras expects sequence first.
    windows = np.swapaxes(windows, 1, 2).copy()
    return windows, labels[window_size - 1 :].astype(np.int32, copy=False)


def _prepare_split(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    benign_label: str,
    numeric_transformer: Pipeline,
    categorical_transformer: OrdinalEncoder,
    window_size: int,
) -> PreparedSplit:
    numeric = numeric_transformer.transform(frame[numeric_columns]).astype(np.float32)
    categorical = (
        categorical_transformer.transform(frame[categorical_columns].astype(str)).astype(np.int32) + 1
    )
    labels = frame[label_column].ne(benign_label).to_numpy(dtype=np.int32)
    numeric_windows, window_labels = _build_windows(numeric, labels, window_size)
    categorical_windows, categorical_labels = _build_windows(categorical, labels, window_size)
    if not np.array_equal(window_labels, categorical_labels):
        raise RuntimeError("Numeric and categorical window labels diverged")
    return PreparedSplit(
        numeric_windows=numeric_windows,
        categorical_windows=[categorical_windows[:, :, index] for index in range(categorical.shape[1])],
        labels=window_labels,
    )


def _build_model(
    *,
    window_size: int,
    numeric_feature_count: int,
    categorical_vocab_sizes: list[int],
    model_config: dict[str, Any],
    learning_rate: float,
) -> keras.Model:
    model_dim = int(model_config["model_dim"])
    num_heads = int(model_config["num_heads"])
    if model_dim % num_heads != 0:
        raise ValueError("model.model_dim must be divisible by model.num_heads")

    numeric_input = keras.Input(
        shape=(window_size, numeric_feature_count), dtype="float32", name="numeric_features"
    )
    inputs: list[keras.KerasTensor] = [numeric_input]
    encoded_features: list[keras.KerasTensor] = [numeric_input]
    embedding_dim = int(model_config["categorical_embedding_dim"])
    for index, vocabulary_size in enumerate(categorical_vocab_sizes):
        categorical_input = keras.Input(
            shape=(window_size,), dtype="int32", name=f"categorical_{index}"
        )
        inputs.append(categorical_input)
        encoded_features.append(
            layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=embedding_dim,
                name=f"categorical_embedding_{index}",
            )(categorical_input)
        )

    x = layers.Concatenate(name="feature_concatenation")(encoded_features)
    x = layers.Dense(model_dim, name="feature_projection")(x)
    for layer_index in range(int(model_config["transformer_layers"])):
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=float(model_config["dropout"]),
            name=f"transformer_attention_{layer_index}",
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"attention_norm_{layer_index}")(
            x + attention
        )
        feed_forward = layers.Dense(model_dim * 2, activation="relu", name=f"ffn_expand_{layer_index}")(
            x
        )
        feed_forward = layers.Dropout(float(model_config["dropout"]), name=f"ffn_dropout_{layer_index}")(
            feed_forward
        )
        feed_forward = layers.Dense(model_dim, name=f"ffn_project_{layer_index}")(feed_forward)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_norm_{layer_index}")(x + feed_forward)

    x = layers.GlobalAveragePooling1D(name="sequence_pooling")(x)
    x = layers.Dense(int(model_config["mlp_units"]), activation="relu", name="classification_mlp")(x)
    x = layers.Dropout(float(model_config["dropout"]), name="classification_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="malicious_probability")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="campus_flow_guard_smoke_transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class _HistoryLogger(keras.callbacks.Callback):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        record = {"epoch": epoch + 1, **{key: float(value) for key, value in (logs or {}).items()}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _inputs(split: PreparedSplit) -> list[np.ndarray]:
    return [split.numeric_windows, *split.categorical_windows]


def _make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("campus_flow_guard.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def run(config_path: Path) -> dict[str, Path]:
    config_path = config_path.expanduser().resolve()
    config = _read_yaml(config_path)
    run_config = _require_mapping(config, "run")
    data_config = _require_mapping(config, "data")
    model_config = _require_mapping(config, "model")
    training_config = _require_mapping(config, "training")
    output_config = _require_mapping(config, "outputs")

    project_root = config_path.parents[2]
    run_name = str(run_config["name"])
    seed = int(run_config["seed"])
    artifacts_dir = (project_root / str(output_config["artifacts_dir"])).resolve()
    logs_dir = (project_root / str(output_config["logs_dir"])).resolve()
    metrics_dir = artifacts_dir / "metrics"
    models_dir = artifacts_dir / "models"
    configs_dir = artifacts_dir / "configs"
    for directory in (metrics_dir, models_dir, configs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "metrics": metrics_dir / f"{run_name}.json",
        "model": models_dir / f"{run_name}.keras",
        "model_summary": models_dir / f"{run_name}_summary.txt",
        "config": configs_dir / f"{run_name}.yaml",
        "log": logs_dir / f"{run_name}.log",
        "history": logs_dir / f"{run_name}_history.jsonl",
    }
    existing_outputs = [path for path in outputs.values() if path.exists()]
    if existing_outputs:
        formatted = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Smoke run outputs already exist; refusing to overwrite: {formatted}")

    shutil.copy2(config_path, outputs["config"])
    logger = _make_logger(outputs["log"])
    started_at = datetime.now(timezone.utc)
    print(f"开始时间: {started_at.astimezone().isoformat(timespec='seconds')}")
    logger.info("Smoke run started with seed=%s config=%s", seed, config_path)
    _set_seed(seed)

    schema_path = (project_root / str(data_config["schema_path"])).resolve()
    schema = _load_schema(schema_path)
    feature_columns = [str(column) for column in schema["feature_columns"]]
    categorical_columns = [str(column) for column in schema["categorical_columns"]]
    label_column = str(schema["label_column"])
    benign_label = str(schema["benign_label"])
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    csv_path = Path(str(data_config["csv_path"])).expanduser()
    frame = _read_smoke_rows(
        csv_path,
        feature_columns + [label_column],
        int(data_config["start_row"]),
        int(data_config["max_rows"]),
    )
    if len(frame) != int(data_config["max_rows"]):
        raise ValueError(f"Requested {data_config['max_rows']} rows but read {len(frame)}")
    required_columns = set(feature_columns + [label_column])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing schema columns: {sorted(missing_columns)}")

    train_frame, validation_frame, test_frame = _split_contiguous(
        frame,
        float(data_config["train_fraction"]),
        float(data_config["validation_fraction"]),
        float(data_config["test_fraction"]),
    )
    for name, partition in (("train", train_frame), ("validation", validation_frame), ("test", test_frame)):
        positives = int(partition[label_column].ne(benign_label).sum())
        if positives == 0 or positives == len(partition):
            raise ValueError(f"{name} partition lacks both classes; select another fixed smoke range")
        logger.info("%s rows=%s malicious=%s benign=%s", name, len(partition), positives, len(partition) - positives)

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    numeric_transformer.fit(train_frame[numeric_columns])
    categorical_transformer.fit(train_frame[categorical_columns].astype(str))
    window_size = int(_require_mapping(config, "window")["size"])
    prepared_train = _prepare_split(
        train_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_transformer=numeric_transformer,
        categorical_transformer=categorical_transformer,
        window_size=window_size,
    )
    prepared_validation = _prepare_split(
        validation_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_transformer=numeric_transformer,
        categorical_transformer=categorical_transformer,
        window_size=window_size,
    )
    prepared_test = _prepare_split(
        test_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        label_column=label_column,
        benign_label=benign_label,
        numeric_transformer=numeric_transformer,
        categorical_transformer=categorical_transformer,
        window_size=window_size,
    )
    vocabulary_sizes = [len(categories) + 1 for categories in categorical_transformer.categories_]
    model = _build_model(
        window_size=window_size,
        numeric_feature_count=len(numeric_columns),
        categorical_vocab_sizes=vocabulary_sizes,
        model_config=model_config,
        learning_rate=float(training_config["learning_rate"]),
    )
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    outputs["model_summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    forward_output = model([array[:1] for array in _inputs(prepared_train)], training=False)
    logger.info("Single-batch forward shape=%s", tuple(forward_output.shape))
    training_started_at = time.monotonic()
    history = model.fit(
        _inputs(prepared_train),
        prepared_train.labels,
        validation_data=(_inputs(prepared_validation), prepared_validation.labels),
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        steps_per_epoch=int(training_config["steps_per_epoch"]),
        shuffle=True,
        verbose=0,
        callbacks=[_HistoryLogger(outputs["history"])],
    )
    training_seconds = time.monotonic() - training_started_at
    evaluation = model.evaluate(_inputs(prepared_test), prepared_test.labels, verbose=0, return_dict=True)
    probabilities = model.predict(_inputs(prepared_test), verbose=0).reshape(-1)
    threshold = float(training_config["threshold"])
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(prepared_test.labels, predictions, labels=[0, 1])
    final_metrics = {
        "loss": float(evaluation["loss"]),
        "accuracy": float(evaluation["accuracy"]),
        "precision": float(precision_score(prepared_test.labels, predictions, zero_division=0)),
        "recall": float(recall_score(prepared_test.labels, predictions, zero_division=0)),
        "f1": float(f1_score(prepared_test.labels, predictions, zero_division=0)),
        "confusion_matrix_labels": ["Benign", "Attack"],
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": threshold,
    }
    model.save(outputs["model"])
    ended_at = datetime.now(timezone.utc)
    metrics_document = {
        "run_type": "smoke_only_not_a_baseline_experiment",
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "training_seconds": round(training_seconds, 6),
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "data": {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": _sha256(csv_path),
            "start_row": int(data_config["start_row"]),
            "rows_loaded": len(frame),
            "feature_columns": feature_columns,
            "categorical_columns": categorical_columns,
            "label_column": label_column,
            "benign_label": benign_label,
            "split_method": "fixed_contiguous_rows_with_partition_local_windows",
            "partition_rows": {
                "train": len(train_frame),
                "validation": len(validation_frame),
                "test": len(test_frame),
            },
            "window_examples": {
                "train": len(prepared_train.labels),
                "validation": len(prepared_validation.labels),
                "test": len(prepared_test.labels),
            },
        },
        "model": {
            "architecture": "compact_flow_transformer_style_smoke_model",
            "parameter_count": int(model.count_params()),
        },
        "training_history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "test_metrics": final_metrics,
    }
    outputs["metrics"].write_text(
        json.dumps(metrics_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Smoke run completed. Metrics=%s", json.dumps(final_metrics, ensure_ascii=False))
    print(f"结束时间: {ended_at.astimezone().isoformat(timespec='seconds')}")
    print(
        "最终指标: "
        + json.dumps(
            {key: final_metrics[key] for key in ("loss", "accuracy", "precision", "recall", "f1")},
            ensure_ascii=False,
        )
    )
    print("输出文件: " + ", ".join(str(path) for path in outputs.values()))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CampusFlowGuard smoke training pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Path to a smoke YAML configuration")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"训练失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
