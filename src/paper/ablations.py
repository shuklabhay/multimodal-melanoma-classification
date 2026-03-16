from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AblationSpec:
    """Normalized ablation specification loaded from YAML."""

    raw: dict[str, Any]


@dataclass(frozen=True)
class AblationPlanPaths:
    plan_json: Path


def _default_spec() -> dict[str, Any]:
    return {
        "name": "tiered_core_ablation_v1",
        "constraints": {
            "max_concurrent_gpu_instances": 10,
            "runtime": "modal",
            "environment": "uv",
        },
        "backbones": ["mobilenet_v4_conv_large", "convnext_tiny", "resnet50"],
        "seeds": [0, 1, 2],
        "retrain": {
            "baseline_seed_completion": ["image", "fusion"],
            "train_time_misalignment": True,
            "metadata_only_narrative": True,
        },
        "eval_only": {
            "shuffle_at_test": True,
            "empty_text": True,
            "constant_text": True,
            "missing_text_curve": [0.1, 0.25, 0.5, 0.75, 0.9],
            "adversarial_swap_pairs": [
                ["MEL", "NV"],
                ["MEL", "BKL"],
                ["AK", "SCC"],
            ],
        },
        "deferred": {
            "fusion_architecture_sweep": ["gated", "cross_attention", "film"],
            "text_encoder_swap": ["clinicalbert", "smaller_baseline"],
        },
    }


def write_default_ablation_spec(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(_default_spec(), f, sort_keys=False)
    return out_path


def load_ablation_spec(path: str | Path) -> AblationSpec:
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid ablation spec at {p}")
    return AblationSpec(raw=raw)


def build_ablation_run_plan(
    spec: AblationSpec,
    out_json: str | Path = "outputs/paper/ablation_run_plan.json",
) -> AblationPlanPaths:
    raw = spec.raw
    backbones = raw.get("backbones", [])
    seeds = raw.get("seeds", [0, 1, 2])

    rows: list[dict[str, Any]] = []

    retrain = raw.get("retrain", {})
    for backbone in backbones:
        for seed in seeds:
            for model_type in retrain.get("baseline_seed_completion", []):
                rows.append(
                    {
                        "job_type": "train",
                        "family": "baseline_seed_completion",
                        "backbone": backbone,
                        "seed": seed,
                        "model_type": model_type,
                        "perturbation": "none",
                    }
                )

            if retrain.get("train_time_misalignment", False):
                rows.append(
                    {
                        "job_type": "train",
                        "family": "train_time_misalignment",
                        "backbone": backbone,
                        "seed": seed,
                        "model_type": "fusion",
                        "perturbation": "train_shuffle",
                    }
                )

            if retrain.get("metadata_only_narrative", False):
                rows.append(
                    {
                        "job_type": "train",
                        "family": "metadata_only_narrative",
                        "backbone": backbone,
                        "seed": seed,
                        "model_type": "fusion",
                        "perturbation": "metadata_only",
                    }
                )

    eval_only = raw.get("eval_only", {})
    missing_curve = eval_only.get("missing_text_curve", [])
    eval_modes = [
        ("shuffle_at_test", "shuffle"),
        ("empty_text", "zero"),
        ("constant_text", "constant"),
        ("adversarial_swap", "adversarial_swap"),
    ]

    for backbone in backbones:
        for seed in seeds:
            rows.append(
                {
                    "job_type": "eval",
                    "family": "control",
                    "backbone": backbone,
                    "seed": seed,
                    "model_type": "fusion",
                    "perturbation": "none",
                }
            )

            for flag, perturb in eval_modes:
                enabled = (
                    eval_only.get(flag, False)
                    if flag != "adversarial_swap"
                    else bool(eval_only.get("adversarial_swap_pairs"))
                )
                if not enabled:
                    continue
                rows.append(
                    {
                        "job_type": "eval",
                        "family": "eval_only",
                        "backbone": backbone,
                        "seed": seed,
                        "model_type": "fusion",
                        "perturbation": perturb,
                    }
                )

            for rate in missing_curve:
                rows.append(
                    {
                        "job_type": "eval",
                        "family": "missing_text_curve",
                        "backbone": backbone,
                        "seed": seed,
                        "model_type": "fusion",
                        "perturbation": "missing",
                        "missing_rate": float(rate),
                    }
                )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    return AblationPlanPaths(plan_json=out_path)
