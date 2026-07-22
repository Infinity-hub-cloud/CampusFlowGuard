"""Compare frozen BCE baseline results with a loss-only Focal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import warnings
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
from tensorflow import keras

from campus_flow_guard.data_protocol import SplitIndices, load_selected_rows, sha256_file
from campus_flow_guard.threshold_optimization import _validate_frozen_input

tf.get_logger().setLevel("ERROR")


METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "far",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
]


def compile_with_focal_loss(
    model: keras.Model,
    *,
    learning_rate: float,
    gamma: float,
    apply_class_balancing: bool,
    alpha: float,
    from_logits: bool,
) -> keras.Model:
    """Replace only the compiled loss while preserving architecture and metrics."""

    architecture_before = model.get_config()
    parameter_count_before = model.count_params()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=gamma,
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            from_logits=from_logits,
        ),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    if model.get_config() != architecture_before or model.count_params() != parameter_count_before:
        raise RuntimeError("Compiling Focal Loss changed the model architecture")
    return model


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def _plot_comparison(frame: pd.DataFrame, path: Path) -> None:
    means = frame.groupby("loss", sort=False)[METRIC_COLUMNS].mean()
    axis = means.plot(kind="bar", figsize=(9.0, 5.0))
    axis.set(title="Frozen BCE baseline vs Focal Loss", ylabel="Three-seed mean")
    axis.tick_params(axis="x", labelrotation=0)
    axis.legend(fontsize=8, ncol=2)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)


def _write_report(path: Path, seed_frame: pd.DataFrame, aggregate: pd.DataFrame, loss_config: dict[str, Any]) -> None:
    lines = [
        "# BCE Baseline vs Focal Loss",
        "",
        "本实验复用冻结 split、预处理、模型结构、balanced sampling、early stopping、validation 阈值选择和三个模型 seed。唯一训练变化是将 Binary Cross-Entropy 替换为 Binary Focal Cross-Entropy。test 未参与阈值或超参数选择。",
        "",
        f"Focal 固定参数：`gamma={loss_config['gamma']}`、`apply_class_balancing={str(loss_config['apply_class_balancing']).lower()}`、`alpha={loss_config['alpha']}`、`from_logits={str(loss_config['from_logits']).lower()}`。没有执行 Focal 超参数搜索。",
        "",
        "## 各 seed test 结果",
        "",
        "| Loss | Seed | Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in seed_frame.iterrows():
        lines.append(
            f"| {row['loss']} | {int(row['seed'])} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['far']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 三 seed 均值 ± 标准差",
            "",
            "| Loss | Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for loss_name in ("BCE", "Focal"):
        mean = aggregate[(aggregate["loss"] == loss_name) & (aggregate["stat"] == "mean")].iloc[0]
        std = aggregate[(aggregate["loss"] == loss_name) & (aggregate["stat"] == "std")].iloc[0]
        cells = [f"{mean[column]:.4f} ± {std[column]:.4f}" for column in METRIC_COLUMNS]
        lines.append(f"| {loss_name} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Path]:
    from campus_flow_guard.baseline import (
        _build_record_projection_model,
        _fit_model,
        _metrics,
        _prepare_partition,
        _set_seed,
        _validation_threshold,
    )

    config_path = config_path.expanduser().resolve()
    project_root = config_path.parents[2]
    experiment_config = _read_yaml(config_path)
    freeze_root = (project_root / str(experiment_config["run"]["frozen_baseline_dir"])).resolve()
    output_root = (project_root / str(experiment_config["outputs"]["directory"])).resolve()
    if freeze_root == output_root or freeze_root in output_root.parents:
        raise ValueError("Experiment outputs must not be inside the frozen baseline archive")
    final_json_path = output_root / "focal_experiment.json"
    if final_json_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed Focal experiment: {final_json_path}")
    for directory in (output_root, output_root / "metrics", output_root / "models", output_root / "logs", output_root / "plots"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "experiment_config.yaml")

    manifest_path = freeze_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frozen_config = _read_yaml(_validate_frozen_input(freeze_root, "config/executed_config.yaml", manifest))
    with np.load(_validate_frozen_input(freeze_root, "splits/split_indices.npz", manifest), allow_pickle=False) as loaded:
        split = SplitIndices(
            train=loaded["train"].astype(np.int64),
            validation=loaded["validation"].astype(np.int64),
            test=loaded["test"].astype(np.int64),
        )
    with _validate_frozen_input(freeze_root, "models/preprocessing.pkl", manifest).open("rb") as handle:
        preprocessing = pickle.load(handle)
    model_seeds = [int(value) for value in manifest["model_seeds"]]
    if model_seeds != [int(value) for value in frozen_config["run"]["model_seeds"]]:
        raise ValueError("Frozen manifest and executed config disagree on model seeds")

    fingerprint = json.loads(
        _validate_frozen_input(freeze_root, "metadata/data_fingerprint.json", manifest).read_text(encoding="utf-8")
    )
    schema_path = (project_root / str(frozen_config["data"]["schema_path"])).resolve()
    if sha256_file(schema_path) != fingerprint["schema_sha256"]:
        raise ValueError("Schema differs from the frozen baseline fingerprint")
    schema = _read_yaml(schema_path)
    csv_path = Path(str(frozen_config["data"]["csv_path"])).resolve()
    if sha256_file(csv_path) != fingerprint["dataset_sha256"]:
        raise ValueError("Dataset differs from the frozen baseline fingerprint")

    selected_frame = load_selected_rows(
        csv_path,
        columns=[*preprocessing["numeric_columns"], *preprocessing["categorical_columns"], str(schema["label_column"])],
        selected_indices=np.sort(split.all_indices),
        chunk_size=int(frozen_config["data"]["read_chunk_size"]),
    )
    prepare_common = {
        "numeric_columns": preprocessing["numeric_columns"],
        "categorical_columns": preprocessing["categorical_columns"],
        "label_column": str(schema["label_column"]),
        "benign_label": str(schema["benign_label"]),
        "numeric_pipeline": preprocessing["numeric_pipeline"],
        "categorical_encoder": preprocessing["categorical_encoder"],
        "window_size": int(frozen_config["model"]["window_size"]),
    }
    train = _prepare_partition(selected_frame.loc[split.train], **prepare_common)
    validation = _prepare_partition(selected_frame.loc[split.validation], **prepare_common)
    test = _prepare_partition(selected_frame.loc[split.test], **prepare_common)
    del selected_frame

    baseline_records: list[dict[str, Any]] = []
    for seed in model_seeds:
        frozen_metric = json.loads(
            _validate_frozen_input(freeze_root, f"metrics/seed{seed}.json", manifest).read_text(encoding="utf-8")
        )
        baseline_records.append({"loss": "BCE", "seed": seed, **{column: frozen_metric[column] for column in METRIC_COLUMNS}})

    loss_config = experiment_config["loss"]
    focal_records: list[dict[str, Any]] = []
    architecture_hash: str | None = None
    expected_parameter_count = int(
        json.loads(_validate_frozen_input(freeze_root, f"metrics/seed{model_seeds[0]}.json", manifest).read_text(encoding="utf-8"))["parameter_count"]
    )
    for seed in model_seeds:
        keras.backend.clear_session()
        _set_seed(seed)
        model = _build_record_projection_model(
            window_size=int(frozen_config["model"]["window_size"]),
            numeric_feature_count=len(preprocessing["numeric_columns"]),
            vocabulary_sizes=[len(categories) + 1 for categories in preprocessing["categorical_encoder"].categories_],
            model_config=frozen_config["model"],
            learning_rate=float(frozen_config["training"]["learning_rate"]),
        )
        current_architecture_hash = hashlib.sha256(
            json.dumps(model.get_config(), sort_keys=True).encode("utf-8")
        ).hexdigest()
        architecture_hash = architecture_hash or current_architecture_hash
        if current_architecture_hash != architecture_hash or model.count_params() != expected_parameter_count:
            raise RuntimeError("Focal experiment model differs from frozen baseline architecture")
        compile_with_focal_loss(
            model,
            learning_rate=float(frozen_config["training"]["learning_rate"]),
            gamma=float(loss_config["gamma"]),
            apply_class_balancing=bool(loss_config["apply_class_balancing"]),
            alpha=float(loss_config["alpha"]),
            from_logits=bool(loss_config["from_logits"]),
        )
        weights_path = output_root / "models" / f"seed{seed}.weights.h5"
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.CSVLogger(output_root / "logs" / f"seed{seed}.csv"),
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=int(frozen_config["training"]["early_stopping_patience"]),
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
        started = time.monotonic()
        history = _fit_model(
            model,
            method="balanced_sampling",
            train=train,
            validation=validation,
            epochs=int(frozen_config["training"]["max_epochs"]),
            batch_size=int(frozen_config["training"]["batch_size"]),
            steps_per_epoch=int(frozen_config["training"]["steps_per_epoch"]),
            seed=seed,
            callbacks=callbacks,
        )
        training_seconds = time.monotonic() - started
        model.load_weights(weights_path)
        validation_probabilities = model.predict(validation.inputs, verbose=0).reshape(-1)
        threshold = _validation_threshold(validation.labels, validation_probabilities)
        test_probabilities = model.predict(test.inputs, verbose=0).reshape(-1)
        metrics = _metrics(test.labels, test_probabilities, threshold)
        record = {
            "loss": "Focal",
            "seed": seed,
            **{column: metrics[column] for column in METRIC_COLUMNS},
            "threshold": float(threshold),
            "TP": int(metrics["TP"]),
            "FP": int(metrics["FP"]),
            "TN": int(metrics["TN"]),
            "FN": int(metrics["FN"]),
            "parameter_count": int(model.count_params()),
            "training_seconds": float(training_seconds),
            "epochs_completed": int(len(history.history["loss"])),
            "best_epoch": int(np.argmax(history.history["val_pr_auc"]) + 1),
        }
        focal_records.append(record)
        (output_root / "metrics" / f"seed{seed}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"阶段结果: Focal seed={seed} F1={record['f1']:.4f} ROC-AUC={record['roc_auc']:.4f} PR-AUC={record['pr_auc']:.4f}")

    seed_frame = pd.DataFrame([*baseline_records, *focal_records])
    comparison_path = output_root / "metrics" / "bce_vs_focal_by_seed.csv"
    seed_frame.to_csv(comparison_path, index=False)
    aggregate_rows: list[dict[str, Any]] = []
    for loss_name, group in seed_frame.groupby("loss", sort=False):
        aggregate_rows.append({"loss": loss_name, "stat": "mean", **group[METRIC_COLUMNS].mean().to_dict()})
        aggregate_rows.append({"loss": loss_name, "stat": "std", **group[METRIC_COLUMNS].std(ddof=1).to_dict()})
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = output_root / "metrics" / "bce_vs_focal_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    _plot_comparison(seed_frame, output_root / "plots" / "bce_vs_focal_metrics.png")
    report_path = output_root / "bce_vs_focal_report.md"
    _write_report(report_path, seed_frame, aggregate, loss_config)
    final_json_path.write_text(
        json.dumps(
            {
                "run_name": experiment_config["run"]["name"],
                "frozen_baseline_id": manifest["baseline_id"],
                "frozen_manifest_sha256": sha256_file(manifest_path),
                "split_indices_sha256": sha256_file(freeze_root / "splits" / "split_indices.npz"),
                "model_seeds": model_seeds,
                "model_architecture_sha256": architecture_hash,
                "parameter_count": expected_parameter_count,
                "only_training_change": "binary_crossentropy_to_binary_focal_crossentropy",
                "test_used_for_selection": False,
                "loss_config": loss_config,
                "focal_results": focal_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "final_json": final_json_path,
        "comparison": comparison_path,
        "aggregate": aggregate_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a loss-only Focal experiment against frozen BCE")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        outputs = run(args.config)
        print("阶段结果: BCE vs Focal 对比已保存 " + str(outputs["report"]))
    except Exception as exc:
        print(f"阶段失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
