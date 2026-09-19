"""Render high-resolution confusion matrices from frozen baseline counts.

This script reads only the frozen baseline summary and writes derived figures
outside the frozen archive. No model inference or experiment is performed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "artifacts" / "metrics" / "baseline_summary.csv"
OUTPUT = PROJECT_ROOT / "artifacts" / "derived_visualizations" / "baseline_confusion_matrices_20260727"


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("seed", "").isdigit()]


def render(seed: str, row: dict[str, str]) -> None:
    tp = int(float(row["TP"]))
    fp = int(float(row["FP"]))
    tn = int(float(row["TN"]))
    fn = int(float(row["FN"]))
    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    total = int(matrix.sum())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=300)
    semantic_cells = np.array([[0, 1], [1, 0]], dtype=int)
    cool_warm = ListedColormap(["#205B7A", "#B84A3A"])
    ax.imshow(semantic_cells, cmap=cool_warm, interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([0, 1], labels=["Benign", "Attack"])
    ax.set_yticks([0, 1], labels=["Benign", "Attack"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Baseline confusion matrix | seed {seed}", pad=58)
    ax.set_aspect("equal")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{value:,}\n({value / total:.2%})",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor="#205B7A", label="Correct prediction"),
            Patch(facecolor="#B84A3A", label="Misclassification"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"baseline_confusion_seed{seed}.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT / f"baseline_confusion_seed{seed}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    expected = {"20260715", "20260716", "20260717"}
    found = {row["seed"] for row in rows}
    missing = expected - found
    if missing:
        raise SystemExit(f"Missing frozen baseline seeds: {sorted(missing)}")
    for row in rows:
        if row["seed"] in expected:
            render(row["seed"], row)
    print(f"Wrote {len(expected)} seed confusion matrices to {OUTPUT}")


if __name__ == "__main__":
    main()
