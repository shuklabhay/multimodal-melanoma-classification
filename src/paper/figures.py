from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text())
    rows = payload.get("data", [])

    labels = sorted({r[0] for r in rows})
    idx = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)

    for actual, pred, count in rows:
        matrix[idx[actual], idx[pred]] = float(count)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums

    return labels, matrix


def _plot_main_metric_bars(summary: dict, out_path: Path) -> None:
    metrics = ["Accuracy", "Balanced Accuracy", "AUROC"]
    img_vals = [
        summary["image_acc_mean"],
        summary["image_bal_acc_mean"],
        summary["image_auroc_mean"],
    ]
    fus_vals = [
        summary["fusion_acc_mean"],
        summary["fusion_bal_acc_mean"],
        summary["fusion_auroc_mean"],
    ]
    txt_vals = [
        summary["text_accuracy"],
        summary["text_bal_acc"],
        summary["text_auroc"],
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    # Okabe-Ito-inspired palette for clearer contrast and colorblind safety.
    ax.bar(x - width, txt_vals, width=width, label="Text-only", color="#999999")
    ax.bar(x, img_vals, width=width, label="Image-only", color="#0072B2")
    ax.bar(x + width, fus_vals, width=width, label="Fusion", color="#E69F00")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Main Performance Metrics (Mean Across Encoders)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_calibration_bars(summary: dict, out_path: Path) -> None:
    metrics = ["ECE", "Brier"]
    img_vals = [summary["image_ece_mean"], summary["image_brier_mean"]]
    fus_vals = [summary["fusion_ece_mean"], summary["fusion_brier_mean"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        x - width / 2,
        img_vals,
        width=width,
        label="Image-only",
        color="#1f77b4",
    )
    ax.bar(
        x + width / 2,
        fus_vals,
        width=width,
        label="Fusion",
        color="#2ca02c",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Lower is better")
    ax.set_title("Calibration Metrics (Mean Across Encoders)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_per_class_delta(per_class_delta: list[dict], out_path: Path) -> None:
    order = ["0MEL", "1NV", "2BCC", "3AK", "4BKL", "5DF", "6VASC", "7SCC"]
    short_name = {
        "0MEL": "MEL",
        "1NV": "NV",
        "2BCC": "BCC",
        "3AK": "AK",
        "4BKL": "BKL",
        "5DF": "DF",
        "6VASC": "VASC",
        "7SCC": "SCC",
    }
    by_key = {row["class_key"]: row for row in per_class_delta}

    classes = [k for k in order if k in by_key]
    y = np.arange(len(classes))
    img_vals = [float(by_key[k]["img_mean"]) for k in classes]
    fus_vals = [float(by_key[k]["fus_mean"]) for k in classes]
    deltas_pp = [(f - i) * 100.0 for i, f in zip(img_vals, fus_vals)]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for yi, x0, x1 in zip(y, img_vals, fus_vals):
        ax.hlines(yi, x0, x1, color="#999999", linewidth=2.0, alpha=0.9, zorder=1)
    ax.scatter(img_vals, y, label="Image-only", color="#0072B2", s=58, zorder=3)
    ax.scatter(fus_vals, y, label="Fusion", color="#E69F00", s=58, zorder=3)

    for yi, x1, dpp in zip(y, fus_vals, deltas_pp):
        sign = "+" if dpp >= 0 else "-"
        ax.text(
            x1 + 0.015,
            yi,
            f"{sign}{abs(dpp):.1f} pp",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([short_name[k] for k in classes])
    ax.set_xlim(0.30, 0.82)
    ax.set_xlabel("Balanced Accuracy")
    ax.set_title("Per-class Balanced Accuracy: Image-only vs Fusion")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_confusion_panel(
    img_conf_path: Path,
    fus_conf_path: Path,
    out_path: Path,
) -> None:
    labels_i, mat_i = _load_confusion_matrix(img_conf_path)
    labels_f, mat_f = _load_confusion_matrix(fus_conf_path)
    if labels_i != labels_f:
        raise ValueError("Confusion matrices do not have matching class labels")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(mat_i, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_title("Image-only")
    im1 = axes[1].imshow(mat_f, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_title("Fusion")
    im2 = axes[2].imshow(mat_f - mat_i, cmap="bwr", vmin=-0.4, vmax=0.4)
    axes[2].set_title("Fusion - Image")

    for ax in axes:
        ax.set_xticks(range(len(labels_i)))
        ax.set_yticks(range(len(labels_i)))
        ax.set_xticklabels(labels_i, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels_i, fontsize=8)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Row-normalized Confusion Matrices (MobileNetV4)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def build_figures(
    input_dir: str | Path = "outputs/paper",
    output_dir: str | Path = "assets/figures/paper",
) -> dict[str, Path]:
    base = Path(input_dir)
    out = Path(output_dir)
    _ensure_dir(out)

    values = json.loads((base / "paper_values.json").read_text())
    baseline = json.loads((base / "runs_baseline.json").read_text())

    fig_main = out / "figure_main_metrics.png"
    fig_cal = out / "figure_calibration_metrics.png"
    fig_delta = out / "figure_per_class_delta.png"

    _plot_main_metric_bars(values["summary"], fig_main)
    _plot_calibration_bars(values["summary"], fig_cal)
    _plot_per_class_delta(values["per_class_delta"], fig_delta)

    m4_img = [
        r
        for r in baseline
        if r.get("image_enc") == "mobilenet_v4_conv_large"
        and r.get("model_type") == "image"
    ]
    m4_fus = [
        r
        for r in baseline
        if r.get("image_enc") == "mobilenet_v4_conv_large"
        and r.get("model_type") == "fusion"
    ]

    fig_conf = out / "figure_confusion_mobilenetv4.png"
    if len(m4_img) == 1 and len(m4_fus) == 1:
        img_ref = m4_img[0].get("confusion_json")
        fus_ref = m4_fus[0].get("confusion_json")
        if img_ref and fus_ref:
            _plot_confusion_panel(Path(img_ref), Path(fus_ref), fig_conf)

    return {
        "main_metrics": fig_main,
        "calibration": fig_cal,
        "per_class_delta": fig_delta,
        "confusion_panel": fig_conf,
    }
