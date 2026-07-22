"""Map attack probabilities and locked validation thresholds to risk levels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RiskMapping:
    version: int
    low_label: str
    medium_label: str
    high_label: str
    high_cutoff_fraction: float


@dataclass(frozen=True)
class RiskAssessment:
    attack_probability: float
    decision_threshold: float
    high_cutoff: float
    risk_level: str
    mapping_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit_interval(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return converted


def load_risk_mapping(path: Path) -> RiskMapping:
    loaded = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Risk mapping root must be a mapping: {path}")
    labels = loaded.get("labels")
    boundaries = loaded.get("boundaries")
    if not isinstance(labels, dict) or not isinstance(boundaries, dict):
        raise ValueError("Risk mapping requires labels and boundaries mappings")
    if boundaries.get("high_cutoff_strategy") != "fraction_of_remaining_probability_range":
        raise ValueError("Unsupported high cutoff strategy")
    fraction = float(boundaries["high_cutoff_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("high_cutoff_fraction must be strictly between 0 and 1")
    return RiskMapping(
        version=int(loaded["version"]),
        low_label=str(labels["low"]),
        medium_label=str(labels["medium"]),
        high_label=str(labels["high"]),
        high_cutoff_fraction=fraction,
    )


def extract_decision_threshold(threshold_result: float | Mapping[str, Any]) -> float:
    if isinstance(threshold_result, Mapping):
        if "threshold" not in threshold_result:
            raise ValueError("Threshold result mapping must contain 'threshold'")
        return _unit_interval(float(threshold_result["threshold"]), "threshold")
    return _unit_interval(float(threshold_result), "threshold")


def assess_risk(
    attack_probability: float,
    threshold_result: float | Mapping[str, Any],
    mapping: RiskMapping,
) -> RiskAssessment:
    probability = _unit_interval(attack_probability, "attack_probability")
    decision_threshold = extract_decision_threshold(threshold_result)
    high_cutoff = decision_threshold + (
        (1.0 - decision_threshold) * mapping.high_cutoff_fraction
    )
    if probability < decision_threshold:
        level = mapping.low_label
    elif probability < high_cutoff:
        level = mapping.medium_label
    else:
        level = mapping.high_label
    return RiskAssessment(
        attack_probability=probability,
        decision_threshold=decision_threshold,
        high_cutoff=high_cutoff,
        risk_level=level,
        mapping_version=mapping.version,
    )


def load_validation_threshold(
    threshold_result_path: Path,
    *,
    seed: int,
    policy: str,
) -> dict[str, Any]:
    document = json.loads(
        threshold_result_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    if document.get("selection_scope") != "validation_only":
        raise ValueError("Threshold result was not selected exclusively on validation")
    if document.get("test_used_for_threshold_selection") is not False:
        raise ValueError("Threshold result indicates that test participated in selection")
    for seed_result in document.get("seed_results", []):
        if int(seed_result.get("seed")) == seed:
            selections = seed_result.get("validation_selection", {})
            if policy not in selections:
                raise KeyError(f"Threshold policy is absent for seed {seed}: {policy}")
            result = selections[policy]
            extract_decision_threshold(result)
            return result
    raise KeyError(f"Threshold seed is absent: {seed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map attack probabilities to risk levels")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--threshold-result", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--policy")
    parser.add_argument("--probability", required=True, type=float, action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ValueError(f"Risk mapping root must be a mapping: {config_path}")
        mapping = load_risk_mapping(config_path)
        policy = args.policy or str(raw_config["threshold_source"]["default_policy"])
        threshold_result = load_validation_threshold(
            args.threshold_result,
            seed=args.seed,
            policy=policy,
        )
        document = {
            "threshold_source": {
                "path": str(args.threshold_result.expanduser().resolve()),
                "selection_scope": "validation_only",
                "seed": args.seed,
                "policy": policy,
                "threshold": float(threshold_result["threshold"]),
            },
            "mapping": {
                "version": mapping.version,
                "high_cutoff_strategy": "fraction_of_remaining_probability_range",
                "high_cutoff_fraction": mapping.high_cutoff_fraction,
                "boundary_rules": {
                    "low": "probability < decision_threshold",
                    "medium": "decision_threshold <= probability < high_cutoff",
                    "high": "probability >= high_cutoff",
                },
            },
            "assessments": [
                assess_risk(probability, threshold_result, mapping).to_dict()
                for probability in args.probability
            ],
        }
        rendered = json.dumps(document, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"风险映射失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
