from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from src.models.model_builder import build_model, load_config
from src.utils.data_utils import load_manifest
from src.utils.eval_utils import (
    ISIC19_LABEL_CONFIG,
    compute_metrics,
    load_checkpoint,
)


_SHORT_TO_LABEL = {cfg["short"]: label for label, cfg in ISIC19_LABEL_CONFIG.items()}
_DEFAULT_SWAP_PAIRS = [("MEL", "NV"), ("MEL", "BKL"), ("AK", "SCC")]


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    """Center crop 256->224 then normalize with ImageNet stats."""
    if images.ndim != 4:
        raise ValueError(f"Expected images rank 4, got shape={images.shape}")

    h, w = images.shape[1], images.shape[2]
    if h >= 224 and w >= 224:
        top = (h - 224) // 2
        left = (w - 224) // 2
        images = images[:, top : top + 224, left : left + 224, :]

    x = images.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (x - mean) / std


def _center_crop_224(images: np.ndarray) -> np.ndarray:
    """Center crop 256->224 for display."""
    h, w = images.shape[1], images.shape[2]
    top = max(0, (h - 224) // 2)
    left = max(0, (w - 224) // 2)
    return images[:, top : top + 224, left : left + 224, :]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _parse_swap_pairs(swap_pairs: list[str] | None) -> list[tuple[int, int]]:
    if not swap_pairs:
        return [(_SHORT_TO_LABEL[a], _SHORT_TO_LABEL[b]) for a, b in _DEFAULT_SWAP_PAIRS]

    pairs: list[tuple[int, int]] = []
    for raw in swap_pairs:
        parts = [p.strip().upper() for p in raw.split(":")]
        if len(parts) != 2:
            raise ValueError(f"Invalid swap pair format '{raw}', expected A:B")
        a, b = parts
        if a not in _SHORT_TO_LABEL or b not in _SHORT_TO_LABEL:
            raise ValueError(f"Unknown class in swap pair '{raw}'")
        pairs.append((_SHORT_TO_LABEL[a], _SHORT_TO_LABEL[b]))
    return pairs


def _apply_embedding_perturbation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    perturbation: str,
    seed: int,
    missing_rate: float,
    swap_pairs: list[str] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    emb = embeddings.copy()
    rng = np.random.default_rng(seed)
    stats: dict[str, Any] = {"perturbation": perturbation}

    if perturbation == "none":
        return emb, stats

    if perturbation == "shuffle":
        perm = rng.permutation(len(emb))
        emb = emb[perm]
        stats["perm_seed"] = seed
        return emb, stats

    if perturbation == "zero":
        emb[:] = 0.0
        return emb, stats

    if perturbation == "constant":
        emb[:] = emb[0]
        stats["constant_source_index"] = 0
        return emb, stats

    if perturbation == "missing":
        mask = rng.random(len(emb)) < missing_rate
        emb[mask] = 0.0
        stats["missing_rate"] = missing_rate
        stats["missing_count"] = int(mask.sum())
        return emb, stats

    if perturbation == "adversarial_swap":
        total_swapped = 0
        for a, b in _parse_swap_pairs(swap_pairs):
            idx_a = np.where(labels == a)[0]
            idx_b = np.where(labels == b)[0]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            rng.shuffle(idx_a)
            rng.shuffle(idx_b)
            n = min(len(idx_a), len(idx_b))
            take_a = idx_a[:n]
            take_b = idx_b[:n]

            tmp = emb[take_a].copy()
            emb[take_a] = emb[take_b]
            emb[take_b] = tmp
            total_swapped += 2 * n

        stats["swap_seed"] = seed
        stats["swap_count"] = total_swapped
        return emb, stats

    raise ValueError(
        f"Unsupported perturbation '{perturbation}'. Use one of: "
        "none|shuffle|zero|constant|missing|adversarial_swap"
    )


def evaluate_checkpoint_bundle(
    run_name: str,
    config_name: str,
    model_type: str | None = None,
    bundle_path: str = "/data/bundled/test.npz",
    batch_size: int = 32,
    perturbation: str = "none",
    seed: int = 0,
    missing_rate: float = 0.5,
    swap_pairs: list[str] | None = None,
    output_json: str | None = None,
    output_npz: str | None = None,
) -> dict[str, Any]:
    config = load_config(f"/root/config/{config_name}.yaml")
    if model_type is not None:
        config["type"] = model_type

    model = build_model(config=config, num_classes=8, seed=seed)
    load_checkpoint(model, run_name)

    bundle = np.load(bundle_path, allow_pickle=True)
    images_raw = bundle["images"]
    embeddings_raw = bundle["embeddings"].astype(np.float32)
    labels = bundle["labels"].astype(np.int32)

    images = _preprocess_images(images_raw)
    embeddings, perturb_stats = _apply_embedding_perturbation(
        embeddings_raw,
        labels,
        perturbation=perturbation,
        seed=seed,
        missing_rate=missing_rate,
        swap_pairs=swap_pairs,
    )

    losses: list[float] = []
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []

    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        img_b = jnp.array(images[start:end])
        emb_b = jnp.array(embeddings[start:end])
        y_b = jnp.array(labels[start:end])

        logits_b = model(images=img_b, embeddings=emb_b, train=False)
        loss_b = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_b, y_b))

        losses.append(float(loss_b))
        logits_chunks.append(np.array(logits_b))
        labels_chunks.append(np.array(y_b))

    logits = jnp.array(np.concatenate(logits_chunks, axis=0))
    labels_arr = jnp.array(np.concatenate(labels_chunks, axis=0))
    metrics = compute_metrics(logits, labels_arr, losses, num_classes=8)
    preds = np.array(jnp.argmax(logits, axis=-1))

    result: dict[str, Any] = {
        "run_name": run_name,
        "config_name": config_name,
        "model_type": config.get("type"),
        "bundle_path": bundle_path,
        "batch_size": batch_size,
        "perturbation": perturbation,
        "seed": seed,
        "missing_rate": missing_rate,
        "metrics": metrics,
        "perturbation_stats": perturb_stats,
        "num_samples": int(len(labels)),
    }

    if output_json:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(result, indent=2, default=float))

    if output_npz:
        out_npz_path = Path(output_npz)
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz_path,
            logits=np.array(logits),
            labels=np.array(labels_arr),
            predictions=preds,
            embeddings=embeddings,
        )

    return result


def _integrated_gradients_image(
    model: Any,
    image: np.ndarray,
    embedding: np.ndarray,
    target_class: int,
    steps: int = 24,
) -> np.ndarray:
    """Integrated gradients attribution map for image input."""
    image_j = jnp.array(image, dtype=jnp.float32)
    emb_j = jnp.array(embedding, dtype=jnp.float32)
    baseline = jnp.zeros_like(image_j)

    def target_logit(img: jnp.ndarray) -> jnp.ndarray:
        logits = model(images=img[None, ...], embeddings=emb_j[None, ...], train=False)
        return logits[0, target_class]

    grad_fn = jax.grad(target_logit)
    total_grad = jnp.zeros_like(image_j)

    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        x = baseline + alpha * (image_j - baseline)
        total_grad = total_grad + grad_fn(x)

    ig = (image_j - baseline) * total_grad / float(steps)
    heat = np.mean(np.abs(np.array(ig)), axis=-1)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    return heat


def _integrated_gradients_embedding(
    model: Any,
    image: np.ndarray,
    embedding: np.ndarray,
    target_class: int,
    steps: int = 24,
) -> np.ndarray:
    """Integrated gradients attribution vector for text embedding input."""
    img_j = jnp.array(image, dtype=jnp.float32)
    emb_j = jnp.array(embedding, dtype=jnp.float32)
    baseline = jnp.zeros_like(emb_j)

    def target_logit(emb: jnp.ndarray) -> jnp.ndarray:
        logits = model(images=img_j[None, ...], embeddings=emb[None, ...], train=False)
        return logits[0, target_class]

    grad_fn = jax.grad(target_logit)
    total_grad = jnp.zeros_like(emb_j)

    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        e = baseline + alpha * (emb_j - baseline)
        total_grad = total_grad + grad_fn(e)

    ig = (emb_j - baseline) * total_grad / float(steps)
    return np.array(ig)


def _select_error_cases(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> list[dict[str, Any]]:
    """Pick two representative error cases with melanoma emphasis."""
    pairs = [
        ("mel_to_nv", 0, 1),
        ("mel_to_bkl", 0, 4),
        ("nv_to_mel", 1, 0),
        ("bkl_to_mel", 4, 0),
    ]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    for name, true_cls, pred_cls in pairs:
        idx = np.where((labels == true_cls) & (preds == pred_cls))[0]
        if len(idx) == 0:
            continue
        best = idx[np.argmax(probs[idx, pred_cls])]
        b = int(best)
        if b in used:
            continue
        used.add(b)
        selected.append({"case_name": name, "index": b})
        if len(selected) >= 2:
            return selected

    err_idx = np.where(labels != preds)[0]
    if len(err_idx) > 0:
        conf = probs[err_idx, preds[err_idx]]
        order = err_idx[np.argsort(conf)[::-1]]
        for i in order:
            ii = int(i)
            if ii in used:
                continue
            used.add(ii)
            selected.append({"case_name": "high_conf_error", "index": ii})
            if len(selected) >= 2:
                break

    return selected


def generate_explainability_case_studies(
    run_name: str,
    config_name: str,
    model_type: str = "fusion",
    bundle_path: str = "/data/bundled/test.npz",
    manifest_path: str = "/root/assets/manifests/split_test.jsonl",
    output_dir: str = "/data/buffer",
    figure_name: str = "figure_explainability_cases.png",
    json_name: str = "explainability_cases.json",
    steps: int = 24,
) -> dict[str, Any]:
    """Generate error-analysis case studies with attribution maps."""
    config = load_config(f"/root/config/{config_name}.yaml")
    config["type"] = model_type
    model = build_model(config=config, num_classes=8, seed=0)
    load_checkpoint(model, run_name)

    bundle = np.load(bundle_path, allow_pickle=True)
    images_raw = bundle["images"]
    embeddings = bundle["embeddings"].astype(np.float32)
    labels = bundle["labels"].astype(np.int32)

    images_proc = _preprocess_images(images_raw)
    images_disp = _center_crop_224(images_raw).astype(np.float32) / 255.0

    logits_chunks: list[np.ndarray] = []
    batch_size = 64
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        logits = model(
            images=jnp.array(images_proc[start:end]),
            embeddings=jnp.array(embeddings[start:end]),
            train=False,
        )
        logits_chunks.append(np.array(logits))
    logits = np.concatenate(logits_chunks, axis=0)
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=-1)

    manifest = load_manifest(manifest_path)
    if len(manifest) != len(labels):
        raise ValueError(f"Manifest size mismatch: {len(manifest)} vs bundle {len(labels)}")

    label_short = {idx: cfg["short"] for idx, cfg in ISIC19_LABEL_CONFIG.items()}

    cases_meta = _select_error_cases(labels, preds, probs)
    if not cases_meta:
        raise RuntimeError("No error cases found for explainability output")

    rows = len(cases_meta)
    fig, axs = plt.subplots(rows, 3, figsize=(14, 5 * rows))
    if rows == 1:
        axs = np.array([axs])

    case_rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(cases_meta):
        i = item["index"]
        img = images_proc[i]
        emb = embeddings[i]
        pred_cls = int(preds[i])
        true_cls = int(labels[i])

        heat = _integrated_gradients_image(model, img, emb, pred_cls, steps=steps)
        ig_emb = _integrated_gradients_embedding(model, img, emb, pred_cls, steps=steps)

        logits_zero = model(
            images=jnp.array(img[None, ...]),
            embeddings=jnp.zeros((1, emb.shape[0]), dtype=jnp.float32),
            train=False,
        )
        probs_zero = _softmax(np.array(logits_zero))[0]
        probs_full = probs[i]

        disp = images_disp[i]
        cmap = plt.get_cmap("jet")
        heat_rgb = cmap(heat)[..., :3]
        overlay = np.clip(0.55 * disp + 0.45 * heat_rgb, 0.0, 1.0)

        axs[row_idx, 0].imshow(disp)
        axs[row_idx, 0].set_title(
            f"Case {row_idx+1}: {item['case_name']}\n"
            f"True={label_short[true_cls]} Pred={label_short[pred_cls]}"
        )
        axs[row_idx, 0].axis("off")

        axs[row_idx, 1].imshow(overlay)
        axs[row_idx, 1].set_title("Image attribution (IG overlay)")
        axs[row_idx, 1].axis("off")

        text_path = Path(manifest[i]["text_path"])
        narrative = ""
        if text_path.exists():
            narrative = text_path.read_text(errors="ignore").strip()
        narrative_short = (narrative[:420] + "...") if len(narrative) > 420 else narrative

        p_pred_full = float(probs_full[pred_cls])
        p_pred_zero = float(probs_zero[pred_cls])
        p_true_full = float(probs_full[true_cls])
        p_true_zero = float(probs_zero[true_cls])
        text_impact = p_pred_full - p_pred_zero
        emb_ig_l1 = float(np.sum(np.abs(ig_emb)))

        panel = (
            f"Index: {i}\n"
            f"Pred prob (full text): {p_pred_full:.3f}\n"
            f"Pred prob (zero text): {p_pred_zero:.3f}\n"
            f"True prob (full text): {p_true_full:.3f}\n"
            f"True prob (zero text): {p_true_zero:.3f}\n"
            f"Text impact on pred prob: {text_impact:+.3f}\n"
            f"Embedding IG L1: {emb_ig_l1:.2f}\n\n"
            f"Narrative:\n{narrative_short}"
        )
        axs[row_idx, 2].text(0.0, 1.0, panel, va="top", ha="left", fontsize=9, wrap=True)
        axs[row_idx, 2].axis("off")

        case_rows.append(
            {
                "index": i,
                "case_name": item["case_name"],
                "true_label": label_short[true_cls],
                "pred_label": label_short[pred_cls],
                "pred_prob_full": p_pred_full,
                "pred_prob_zero": p_pred_zero,
                "true_prob_full": p_true_full,
                "true_prob_zero": p_true_zero,
                "text_impact_pred_prob": text_impact,
                "embedding_ig_l1": emb_ig_l1,
                "text_path": manifest[i]["text_path"],
                "image_path": manifest[i]["image_path"],
            }
        )

    fig.suptitle(
        f"Explainability Case Studies ({run_name})\n"
        "Left: original image, middle: image IG overlay, right: narrative + counterfactual text effect",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / figure_name
    json_path = out_dir / json_name
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    result = {
        "run_name": run_name,
        "config_name": config_name,
        "model_type": model_type,
        "figure_path": str(fig_path),
        "json_path": str(json_path),
        "cases": case_rows,
    }
    json_path.write_text(json.dumps(result, indent=2))
    return result
