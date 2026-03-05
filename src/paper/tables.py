from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_MAP = {
    0: "MEL",
    1: "NV",
    2: "BCC",
    3: "AK",
    4: "BKL",
    5: "DF",
    6: "VASC",
    7: "SCC",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_tables(
    input_dir: str | Path = "outputs/paper",
    manifests_dir: str | Path = "assets/manifests",
) -> dict[str, Path]:
    base = Path(input_dir)
    manifests = Path(manifests_dir)

    baseline = _load_json(base / "runs_baseline.json")
    per_class = _load_json(base / "per_class_bal_acc.json")

    text_run = [r for r in baseline if r.get("model_type") == "text"]
    image_runs = [r for r in baseline if r.get("model_type") == "image"]
    fusion_runs = [r for r in baseline if r.get("model_type") == "fusion"]

    by_enc_img = {r["image_enc"]: r for r in image_runs}
    by_enc_fus = {r["image_enc"]: r for r in fusion_runs}

    enc_rows = []
    for enc in sorted(set(by_enc_img.keys()) & set(by_enc_fus.keys())):
        i = by_enc_img[enc]
        f = by_enc_fus[enc]
        enc_rows.append(
            {
                "encoder": enc,
                "img_accuracy": float(i["test_accuracy"]),
                "img_bal_acc": float(i["test_balanced_accuracy"]),
                "img_auroc": float(i["test_auroc"]),
                "img_ece": float(i["test_ece"]),
                "img_brier": float(i["test_brier"]),
                "fus_accuracy": float(f["test_accuracy"]),
                "fus_bal_acc": float(f["test_balanced_accuracy"]),
                "fus_auroc": float(f["test_auroc"]),
                "fus_ece": float(f["test_ece"]),
                "fus_brier": float(f["test_brier"]),
                "delta_accuracy": float(f["test_accuracy"]) - float(i["test_accuracy"]),
                "delta_bal_acc": float(f["test_balanced_accuracy"]) - float(i["test_balanced_accuracy"]),
                "delta_auroc": float(f["test_auroc"]) - float(i["test_auroc"]),
            }
        )

    summary = {
        "text_accuracy": float(text_run[0]["test_accuracy"]),
        "text_bal_acc": float(text_run[0]["test_balanced_accuracy"]),
        "text_auroc": float(text_run[0]["test_auroc"]),
        "image_acc_mean": _mean([float(r["test_accuracy"]) for r in image_runs]),
        "image_bal_acc_mean": _mean([float(r["test_balanced_accuracy"]) for r in image_runs]),
        "image_auroc_mean": _mean([float(r["test_auroc"]) for r in image_runs]),
        "image_ece_mean": _mean([float(r["test_ece"]) for r in image_runs]),
        "image_brier_mean": _mean([float(r["test_brier"]) for r in image_runs]),
        "fusion_acc_mean": _mean([float(r["test_accuracy"]) for r in fusion_runs]),
        "fusion_bal_acc_mean": _mean([float(r["test_balanced_accuracy"]) for r in fusion_runs]),
        "fusion_auroc_mean": _mean([float(r["test_auroc"]) for r in fusion_runs]),
        "fusion_ece_mean": _mean([float(r["test_ece"]) for r in fusion_runs]),
        "fusion_brier_mean": _mean([float(r["test_brier"]) for r in fusion_runs]),
        "delta_bal_acc_mean": _mean([r["delta_bal_acc"] for r in enc_rows]),
        "delta_bal_acc_min": min(r["delta_bal_acc"] for r in enc_rows),
        "delta_bal_acc_max": max(r["delta_bal_acc"] for r in enc_rows),
        "delta_auroc_mean": _mean([r["delta_auroc"] for r in enc_rows]),
        "delta_auroc_min": min(r["delta_auroc"] for r in enc_rows),
        "delta_auroc_max": max(r["delta_auroc"] for r in enc_rows),
    }

    per_class_grouped: dict[str, dict[str, list[float]]] = {}
    for row in per_class:
        key = row["class_key"]
        per_class_grouped.setdefault(key, {"image": [], "fusion": []})
        mt = row["model_type"]
        if mt in {"image", "fusion"} and row.get("bal_acc") is not None:
            per_class_grouped[key][mt].append(float(row["bal_acc"]))

    per_class_delta = []
    for class_key in sorted(per_class_grouped.keys()):
        img_vals = per_class_grouped[class_key]["image"]
        fus_vals = per_class_grouped[class_key]["fusion"]
        if not img_vals or not fus_vals:
            continue
        img_mean = _mean(img_vals)
        fus_mean = _mean(fus_vals)
        per_class_delta.append(
            {
                "class_key": class_key,
                "img_mean": img_mean,
                "fus_mean": fus_mean,
                "delta": fus_mean - img_mean,
            }
        )

    class_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        rows = _load_jsonl(manifests / f"isic2019_split_{split}.jsonl")
        counts = Counter(r["label"] for r in rows)
        for label, short in CLASS_MAP.items():
            class_counts[split][short] = counts[label]

    values_payload = {
        "summary": summary,
        "encoder_rows": enc_rows,
        "per_class_delta": per_class_delta,
        "class_counts": class_counts,
    }

    values_path = base / "paper_values.json"
    values_path.write_text(json.dumps(values_payload, indent=2))

    return {
        "paper_values_json": values_path,
    }
