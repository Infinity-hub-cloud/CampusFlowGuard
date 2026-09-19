"""Render the deployed risk-level mapping as a report-ready figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference" / "local_focal.yaml"
RISK_CONFIG = PROJECT_ROOT / "configs" / "risk" / "risk_mapping.yaml"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "report_assets" / "risk_mapping"


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def main() -> None:
    inference = _read_yaml(INFERENCE_CONFIG)
    mapping = _read_yaml(RISK_CONFIG)
    threshold = float(inference["decision"]["threshold"])
    fraction = float(mapping["boundaries"]["high_cutoff_fraction"])
    high_cutoff = threshold + fraction * (1.0 - threshold)
    if not 0.0 < threshold < high_cutoff < 1.0:
        raise ValueError("Risk boundaries must satisfy 0 < threshold < high cutoff < 1")

    colors = {"Low": "#4C956C", "Medium": "#E9B949", "High": "#C8553D"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
    })
    fig, axis = plt.subplots(figsize=(10.8, 4.2), dpi=300)
    bar_y = 0.15
    bar_height = 0.34
    segments = [
        (0.0, threshold, "Low"),
        (threshold, high_cutoff - threshold, "Medium"),
        (high_cutoff, 1.0 - high_cutoff, "High"),
    ]
    for start, width, label in segments:
        axis.add_patch(
            Rectangle(
                (start, bar_y),
                width,
                bar_height,
                facecolor=colors[label],
                edgecolor="white",
                linewidth=2.0,
            )
        )

    axis.text(threshold / 2.0, bar_y + bar_height / 2.0, "Low\np < threshold",
              ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    axis.text((threshold + high_cutoff) / 2.0, bar_y + bar_height / 2.0,
              "Medium\nthreshold <= p < high cutoff", ha="center", va="center",
              color="#382B00", fontweight="bold", fontsize=9)
    axis.annotate(
        "High",
        xy=((high_cutoff + 1.0) / 2.0, bar_y + bar_height / 2.0),
        xytext=(0.965, 0.62),
        ha="center",
        va="bottom",
        color=colors["High"],
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": colors["High"], "lw": 1.2},
    )

    for boundary, color in ((threshold, "#355070"), (high_cutoff, "#8B2F2F")):
        axis.axvline(boundary, ymin=0.28, ymax=0.69, color=color, linestyle="--", linewidth=1.5)
    axis.annotate(
        f"Decision threshold = {threshold:.4f}\n(validation F1 maximum)",
        xy=(threshold, bar_y + bar_height),
        xytext=(0.52, 0.93),
        ha="center",
        va="center",
        color="#355070",
        arrowprops={"arrowstyle": "->", "color": "#355070", "lw": 1.2},
    )
    axis.annotate(
        f"High cutoff = {high_cutoff:.4f}",
        xy=(high_cutoff, bar_y + bar_height),
        xytext=(0.84, 1.15),
        ha="center",
        va="center",
        color="#8B2F2F",
        arrowprops={"arrowstyle": "->", "color": "#8B2F2F", "lw": 1.2},
    )

    axis.text(
        0.0,
        -0.28,
        r"$\tau_H = \tau + 0.75(1-\tau)$",
        ha="left",
        va="center",
        fontsize=12,
        color="#222222",
    )
    axis.text(
        1.0,
        -0.28,
        "Validation-selected threshold; test excluded",
        ha="right",
        va="center",
        fontsize=9,
        color="#555555",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.38, 1.32)
    axis.set_xlabel("Attack probability p")
    axis.set_yticks([])
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.55)
    axis.set_axisbelow(True)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "risk_level_mapping"
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"threshold={threshold:.12f}")
    print(f"high_cutoff={high_cutoff:.12f}")
    print(f"output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
