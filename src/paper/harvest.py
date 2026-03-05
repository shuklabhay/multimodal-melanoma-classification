from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv


BASELINE_DATE = "2026-02-25"
GPT5_REFERENCE_DATE = "2025-12-07"
NARRATIVE_DATE_PREFIX = "2026-02-0"


@dataclass(frozen=True)
class HarvestPaths:
    base_dir: Path
    baseline_json: Path
    narrative_json: Path
    gpt5_json: Path
    per_class_json: Path


def _run_to_iso_date(created_at: str | None) -> str | None:
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _collect_run_record(run: wandb.apis.public.runs.Run) -> dict[str, Any]:
    config = _to_jsonable(dict(run.config or {}))
    summary = _to_jsonable(dict(run.summary or {}))

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "created_date": _run_to_iso_date(run.created_at),
        "config_name": config.get("config_name"),
        "model_type": config.get("type"),
        "image_enc": config.get("image_enc"),
        "test_accuracy": _safe_float(summary.get("test_accuracy")),
        "test_balanced_accuracy": _safe_float(summary.get("test_balanced_accuracy")),
        "test_auroc": _safe_float(summary.get("test_auroc")),
        "test_ece": _safe_float(summary.get("test_ece")),
        "test_brier": _safe_float(summary.get("test_brier")),
        "test_loss": _safe_float(summary.get("test_loss")),
        "best_epoch": summary.get("best_epoch"),
        "best_val_balanced_accuracy": _safe_float(summary.get("best_val_balanced_accuracy")),
        "runtime_s": summary.get("_runtime"),
        "step": summary.get("_step"),
        "summary": summary,
        "config": config,
    }


def _download_confusion_tables(
    runs: list[wandb.apis.public.runs.Run], out_dir: Path
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    for run in runs:
        summary = dict(run.summary or {})
        table_ref = summary.get("test_confusion_matrix_table")
        if not hasattr(table_ref, "get"):
            continue

        path = table_ref.get("path")
        if not path:
            continue

        try:
            local = run.file(path).download(root=str(out_dir), replace=True)
            mapping[run.id] = local.name
        except Exception:
            continue

    return mapping


def _extract_per_class_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = record["summary"]
    for key, value in summary.items():
        if not str(key).startswith("test_cw/bal_acc_"):
            continue
        class_key = str(key).replace("test_cw/bal_acc_", "")
        rows.append(
            {
                "run_id": record["run_id"],
                "run_name": record["run_name"],
                "model_type": record["model_type"],
                "image_enc": record["image_enc"],
                "class_key": class_key,
                "bal_acc": _safe_float(value),
            }
        )
    return rows


def _paths(base_dir: str | Path) -> HarvestPaths:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    return HarvestPaths(
        base_dir=root,
        baseline_json=root / "runs_baseline.json",
        narrative_json=root / "runs_narrative_quality.json",
        gpt5_json=root / "runs_gpt5_reference.json",
        per_class_json=root / "per_class_bal_acc.json",
    )


def harvest_wandb_runs(
    output_dir: str | Path = "outputs/paper",
    download_confusions: bool = True,
) -> HarvestPaths:
    load_dotenv(".env")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    if not entity or not project:
        raise ValueError("WANDB_ENTITY and WANDB_PROJECT must be set")

    api = wandb.Api(timeout=120)
    path = f"{entity}/{project}"
    runs = list(api.runs(path, per_page=1000))

    out = _paths(output_dir)

    baseline_runs: list[wandb.apis.public.runs.Run] = []
    narrative_runs: list[wandb.apis.public.runs.Run] = []
    gpt5_runs: list[wandb.apis.public.runs.Run] = []

    for run in runs:
        created_date = _run_to_iso_date(run.created_at)
        config = dict(run.config or {})
        name = (run.name or "").lower()

        if (
            created_date == BASELINE_DATE
            and run.state == "finished"
            and config.get("type") in {"image", "fusion", "text"}
        ):
            baseline_runs.append(run)
            continue

        if (
            run.state == "finished"
            and name.startswith("narratives_")
            and (run.created_at or "").startswith(NARRATIVE_DATE_PREFIX)
        ):
            narrative_runs.append(run)
            continue

        if (
            run.state == "finished"
            and "gpt5_chat" in name
            and created_date == GPT5_REFERENCE_DATE
        ):
            gpt5_runs.append(run)
            continue

    baseline_records = [_collect_run_record(r) for r in baseline_runs]
    narrative_records = [_collect_run_record(r) for r in narrative_runs]
    gpt5_records = [_collect_run_record(r) for r in gpt5_runs]

    baseline_records = sorted(
        baseline_records,
        key=lambda r: (r["created_at"] or "", r["run_name"] or ""),
    )
    narrative_records = sorted(narrative_records, key=lambda r: r["run_name"] or "")
    gpt5_records = sorted(gpt5_records, key=lambda r: r["run_name"] or "")

    if len(baseline_records) != 13:
        raise RuntimeError(
            f"Expected 13 baseline runs on {BASELINE_DATE}, found {len(baseline_records)}"
        )

    confusion_map: dict[str, str] = {}
    if download_confusions:
        confusion_map = _download_confusion_tables(
            baseline_runs,
            out.base_dir / "confusions",
        )

    baseline_payload = []
    for rec in baseline_records:
        baseline_payload.append(
            {
                **{
                    k: v
                    for k, v in rec.items()
                    if k not in {"summary", "config"}
                },
                "confusion_json": confusion_map.get(rec["run_id"]),
                "config": rec["config"],
                "summary": rec["summary"],
            }
        )

    out.baseline_json.write_text(json.dumps(baseline_payload, indent=2, default=str))
    out.narrative_json.write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in {"summary", "config"}}
                for r in narrative_records
            ],
            indent=2,
            default=str,
        )
    )
    out.gpt5_json.write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in {"summary", "config"}}
                for r in gpt5_records
            ],
            indent=2,
            default=str,
        )
    )

    per_class_rows: list[dict[str, Any]] = []
    for rec in baseline_records:
        per_class_rows.extend(_extract_per_class_rows(rec))
    out.per_class_json.write_text(json.dumps(per_class_rows, indent=2, default=str))

    return out
