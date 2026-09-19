"""Create publication-style threshold analysis plots from frozen validation scans.

The input CSV files are read-only frozen experiment outputs.  Savitzky-Golay
filtering is applied only to the lines drawn in the figures; raw values are
used for F1-optimum markers, annotations, and validation checks.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "threshold_optimization" / "unsw_trusted_baseline_v1"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "derived_visualizations" / "threshold_analysis_20260725"
)
REQUIRED_COLUMNS = ("threshold", "far", "recall", "f1")
EXPECTED_SEEDS = (20260715, 20260716, 20260717)
COLORS = ("#1F77B4", "#D62728", "#2CA02C")
SMOOTH_WINDOW_LENGTH = 51
SMOOTH_POLYNOMIAL_ORDER = 3


@dataclass(frozen=True)
class ThresholdScan:
    """Validated raw threshold metrics for one fixed validation seed."""

    seed: int
    frame: pd.DataFrame


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _seed_from_path(path: Path) -> int:
    matched = re.fullmatch(r"validation_threshold_search_seed(\d+)\.csv", path.name)
    if matched is None:
        raise ValueError(f"Unexpected threshold scan filename: {path.name}")
    return int(matched.group(1))


def _load_scans(input_directory: Path) -> list[ThresholdScan]:
    paths = sorted(input_directory.glob("validation_threshold_search_seed*.csv"))
    if not paths:
        raise FileNotFoundError(f"No validation threshold scans found in: {input_directory}")

    scans: list[ThresholdScan] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy().sort_values("threshold", kind="stable")
        if frame.empty or frame["threshold"].duplicated().any():
            raise ValueError(f"{path.name} must contain unique, non-empty threshold values")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains non-finite metric values")
        if (frame["threshold"] < 0).any() or (frame["threshold"] > 1).any():
            raise ValueError(f"{path.name} contains a threshold outside [0, 1]")
        for metric in ("far", "recall", "f1"):
            if (frame[metric] < 0).any() or (frame[metric] > 1).any():
                raise ValueError(f"{path.name} contains {metric} outside [0, 1]")
        scans.append(ThresholdScan(seed=_seed_from_path(path), frame=frame))

    found_seeds = tuple(scan.seed for scan in scans)
    if found_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {found_seeds}")
    return scans


def _display_savgol(values: np.ndarray) -> np.ndarray:
    """Apply light Savitzky-Golay smoothing only to the rendered line."""

    if len(values) < 5:
        return values.copy()
    window_length = min(SMOOTH_WINDOW_LENGTH, len(values))
    if window_length % 2 == 0:
        window_length -= 1
    polynomial_order = min(SMOOTH_POLYNOMIAL_ORDER, window_length - 2)
    smoothed = savgol_filter(values, window_length=window_length, polyorder=polynomial_order)
    return np.clip(smoothed, 0.0, 1.0)


def _visible_display_curve(
    thresholds: np.ndarray,
    smoothed_values: np.ndarray,
    ylim: tuple[float, float],
) -> np.ndarray:
    """Mask off-range edge-tail points to avoid clipped display artifacts."""

    lower, upper = ylim
    visible = (
        (thresholds > 0.0)
        & (thresholds < 1.0)
        & (smoothed_values >= lower)
        & (smoothed_values <= upper)
    )
    return np.where(visible, smoothed_values, np.nan)


def _metric_ylim(scans: list[ThresholdScan], metric: str) -> tuple[float, float]:
    values = np.concatenate([scan.frame[metric].to_numpy(dtype=float) for scan in scans])
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    padding = max(span * 0.06, 0.02)
    return max(0.0, minimum - padding), min(1.0, maximum + padding)


def _operating_ylim(scans: list[ThresholdScan], metric: str) -> tuple[float, float]:
    """Return a data-driven y-range that separates the three seed curves.

    Exact threshold endpoints can force FAR, recall, and F1 to 0 or 1 and
    flatten the validation operating region.  The focus range uses only
    non-endpoint thresholds and preserves the complete threshold x-axis; raw
    values outside this y-range are not altered and are disclosed in-figure.
    """

    values = np.concatenate(
        [
            scan.frame.loc[
                scan.frame["threshold"].between(0.001, 0.999), metric
            ].to_numpy(dtype=float)
            for scan in scans
        ]
    )
    if metric == "far":
        upper = max(float(np.quantile(values, 0.995)) * 1.10, 0.0075)
        return 0.0, min(1.0, upper)
    if metric == "recall":
        lower = max(0.0, float(np.quantile(values, 0.01)) - 0.01)
        return lower, 1.003
    if metric == "f1":
        lower = max(0.0, float(np.quantile(values, 0.01)) - 0.01)
        upper = min(1.0, float(np.quantile(values, 0.995)) + 0.005)
        return lower, upper
    raise ValueError(f"Unsupported operating-range metric: {metric}")


def _add_operating_range_note(ax: plt.Axes, scans: list[ThresholdScan], metric: str) -> None:
    lower, upper = ax.get_ylim()
    values = np.concatenate([scan.frame[metric].to_numpy(dtype=float) for scan in scans])
    hidden_count = int(((values < lower) | (values > upper)).sum())
    if hidden_count:
        ax.text(
            0.015,
            0.035,
            f"Operating-range y-axis; {hidden_count} edge values lie outside the display.",
            transform=ax.transAxes,
            color="#596878",
            fontsize=8.3,
            va="bottom",
        )


def _prepare_axes(ax: plt.Axes, *, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#C9D2DC", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#526273")
    ax.spines["bottom"].set_color("#526273")
    ax.tick_params(colors="#344454")


def _save_figure(figure: plt.Figure, output_directory: Path, stem: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_metric(
    scans: list[ThresholdScan],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_directory: Path,
    stem: str,
    percent_axis: bool = False,
    far_constraint: bool = False,
) -> None:
    figure, ax = plt.subplots(figsize=(8.9, 5.8))
    handles = []
    ylim = _operating_ylim(scans, metric)
    for scan, color in zip(scans, COLORS, strict=True):
        thresholds = scan.frame["threshold"].to_numpy(dtype=float)
        raw_values = scan.frame[metric].to_numpy(dtype=float)
        line = ax.plot(
            thresholds,
            _visible_display_curve(thresholds, _display_savgol(raw_values), ylim),
            color=color,
            linewidth=2.25,
            label=f"Seed {scan.seed}",
        )[0]
        handles.append(line)

    if far_constraint:
        handles.append(
            ax.axhline(
                0.005,
                color="#505A66",
                linestyle="--",
                linewidth=1.3,
                label="FAR constraint = 0.5%",
            )
        )

    _prepare_axes(ax, ylabel=ylabel, ylim=ylim)
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
        ax.yaxis.set_major_locator(MultipleLocator(0.0025))
    elif metric == "recall":
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
    elif metric == "f1":
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
    if far_constraint:
        ax.text(
            0.015,
            0.95,
            "Threshold = 0 gives FAR = 100% for all seeds.",
            transform=ax.transAxes,
            color="#596878",
            fontsize=8.5,
            va="top",
        )
    _add_operating_range_note(ax, scans, metric)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        handlelength=2.4,
    )
    figure.subplots_adjust(top=0.82, left=0.12, right=0.97, bottom=0.13)
    _save_figure(figure, output_directory, stem)


def _plot_f1(scans: list[ThresholdScan], output_directory: Path) -> None:
    figure, ax = plt.subplots(figsize=(11.0, 5.9))
    handles = []
    maximums: list[tuple[ThresholdScan, pd.Series, str]] = []
    ylim = _operating_ylim(scans, "f1")
    for scan, color in zip(scans, COLORS, strict=True):
        thresholds = scan.frame["threshold"].to_numpy(dtype=float)
        raw_f1 = scan.frame["f1"].to_numpy(dtype=float)
        line = ax.plot(
            thresholds,
            _visible_display_curve(thresholds, _display_savgol(raw_f1), ylim),
            color=color,
            linewidth=2.25,
            label=f"Seed {scan.seed}",
        )[0]
        handles.append(line)
        maximum = scan.frame.loc[scan.frame["f1"].idxmax()]
        maximums.append((scan, maximum, color))
        ax.scatter(
            float(maximum["threshold"]),
            float(maximum["f1"]),
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )

    _prepare_axes(ax, ylabel="F1-score", ylim=ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    _add_operating_range_note(ax, scans, "f1")
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.41, 0.995),
        ncol=len(handles),
        frameon=False,
        handlelength=2.4,
    )

    annotation_positions = (0.84, 0.51, 0.18)
    for (scan, maximum, color), vertical_position in zip(
        maximums, annotation_positions, strict=True
    ):
        annotation_text = (
            f"Seed {scan.seed}\n"
            f"optimal threshold = {float(maximum['threshold']):.3f}\n"
            f"F1 = {float(maximum['f1']):.4f}"
        )
        ax.annotate(
            annotation_text,
            xy=(float(maximum["threshold"]), float(maximum["f1"])),
            xycoords="data",
            xytext=(1.03, vertical_position),
            textcoords="axes fraction",
            color="#243241",
            fontsize=9.5,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.34", "fc": "white", "ec": color, "lw": 0.9},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1, "shrinkA": 5},
            annotation_clip=False,
        )

    figure.subplots_adjust(top=0.82, left=0.10, right=0.69, bottom=0.13)
    _save_figure(figure, output_directory, "F1_threshold_curve")


def run(input_directory: Path, output_directory: Path) -> list[Path]:
    """Read frozen scans and write six derived figure files without altering inputs."""

    _configure_style()
    scans = _load_scans(input_directory.expanduser().resolve())
    output_directory = output_directory.expanduser().resolve()
    _plot_metric(
        scans,
        metric="far",
        title="Validation false alarm rate across decision thresholds",
        ylabel="False alarm rate",
        output_directory=output_directory,
        stem="FAR_threshold_curve",
        percent_axis=True,
        far_constraint=True,
    )
    _plot_metric(
        scans,
        metric="recall",
        title="Validation recall across decision thresholds",
        ylabel="Recall",
        output_directory=output_directory,
        stem="Recall_threshold_curve",
    )
    _plot_f1(scans, output_directory)
    return [
        output_directory / f"{stem}.{suffix}"
        for stem in ("FAR_threshold_curve", "Recall_threshold_curve", "F1_threshold_curve")
        for suffix in ("png", "pdf")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create derived paper-style figures from frozen validation threshold scans."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    try:
        output_paths = run(args.input_dir, args.output_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Threshold figure generation failed: {type(exc).__name__}: {exc}")
        return 1
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
