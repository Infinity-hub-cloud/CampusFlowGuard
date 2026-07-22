from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.inference import run as run_inference


def run_demo(config_path: Path, input_path: Path, output_path: Path) -> dict:
    document = run_inference(config_path, input_path)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal CampusFlowGuard demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference" / "local_focal.yaml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "demo" / "mixed_demo.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "demo" / "mixed_demo_report.json",
    )
    args = parser.parse_args()
    try:
        document = run_demo(args.config, args.input, args.output)
    except Exception as exc:
        print(f"快速演示失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = document["detection_summary"]
    print(f"总流量数: {summary['input_row_count']}")
    print(f"攻击数: {summary['detected_attack_count']}")
    print(
        "风险等级统计: "
        + ", ".join(
            f"{level}={count}" for level, count in summary["risk_level_counts"].items()
        )
    )
    print(f"JSON报告: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
