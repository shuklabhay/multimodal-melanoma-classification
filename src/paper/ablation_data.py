from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.models.backbones import SentenceTransformerEncoder


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _safe_age(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return str(int(float(value)))
    except Exception:
        s = str(value).strip()
        return s if s else "unknown"


def _image_stem_from_manifest_row(row: dict[str, Any]) -> str:
    return Path(_safe_str(row.get("image_path"), default="")).stem


def _image_stem_from_narrative_row(row: dict[str, Any]) -> str:
    return Path(_safe_str(row.get("text_path"), default="")).stem


def _metadata_text(row: dict[str, Any]) -> str:
    age = _safe_age(row.get("age_approx"))
    sex = _safe_str(row.get("sex"))
    site = _safe_str(row.get("anatom_site_general"))
    return (
        f"I am approximately {age} years old. "
        f"Sex recorded as {sex}. "
        f"Lesion site recorded as {site}."
    )


def _symptoms_text(row: dict[str, Any]) -> str:
    symptoms = row.get("symptoms") or []
    if isinstance(symptoms, list) and symptoms:
        joined = "; ".join(str(s).strip() for s in symptoms if str(s).strip())
        if joined:
            return f"Symptoms reported: {joined}."
    return "Symptoms reported: none."


def _temporal_text(row: dict[str, Any]) -> str:
    tq = _safe_str(row.get("temporal_quantifier"))
    mem = _safe_str(row.get("memory_category"))
    struct = _safe_str(row.get("structure"))
    return (
        f"Onset timing: {tq}. "
        f"Memory precision category: {mem}. "
        f"Temporal narrative structure: {struct}."
    )


def _risk_text(row: dict[str, Any]) -> str:
    age = _safe_age(row.get("age_approx"))
    sex = _safe_str(row.get("sex"))
    site = _safe_str(row.get("anatom_site_general"))
    return (
        "Risk-context proxy from available metadata. "
        f"Age approximately {age}. Sex {sex}. Anatomical site {site}."
    )


def _join_list_field(value: Any) -> str:
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        if items:
            return "; ".join(items)
    s = str(value).strip() if value is not None else ""
    return s if s else "none"


def _template_full_text(row: dict[str, Any]) -> str:
    """Deterministic template preserving narrative semantics without free-form style."""
    age = _safe_age(row.get("age_approx"))
    sex = _safe_str(row.get("sex"))
    site = _safe_str(row.get("anatom_site_general"))
    mode = _safe_str(row.get("narrative_mode"))
    symptoms = _join_list_field(row.get("symptoms"))
    temporal = _safe_str(row.get("temporal_quantifier"))
    memory = _safe_str(row.get("memory_category"))
    structure = _safe_str(row.get("structure"))
    action = _safe_str(row.get("action_cues"), default="none")
    context = _safe_str(row.get("context_slot"), default="none")
    return (
        f"Patient metadata: age approximately {age}; sex {sex}; anatomical site {site}. "
        f"Narrative mode: {mode}. "
        f"Symptoms: {symptoms}. "
        f"Onset timing: {temporal}. "
        f"Memory precision category: {memory}. "
        f"Narrative structure: {structure}. "
        f"Action cues: {action}. "
        f"Context slot: {context}."
    )


def _structured_meta_age_stats(rows: list[dict[str, Any]]) -> tuple[float, float]:
    ages = [
        float(row["age_approx"]) for row in rows if row.get("age_approx") is not None
    ]
    if not ages:
        return 0.0, 1.0
    age_mean = float(np.mean(ages))
    age_std = float(np.std(ages))
    return age_mean, age_std if age_std > 0.0 else 1.0


def _structured_meta_categories(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    sex_values = sorted({_safe_str(row.get("sex")) for row in rows})
    site_values = sorted({_safe_str(row.get("anatom_site_general")) for row in rows})
    return sex_values, site_values


def _structured_meta_vector(
    row: dict[str, Any],
    age_mean: float,
    age_std: float,
    sex_values: list[str],
    site_values: list[str],
) -> np.ndarray:
    sex_value = _safe_str(row.get("sex"))
    site_value = _safe_str(row.get("anatom_site_general"))
    age_value = row.get("age_approx")
    age_missing = 1.0 if age_value is None else 0.0
    age_scaled = 0.0 if age_value is None else (float(age_value) - age_mean) / age_std
    sex_one_hot = [1.0 if sex_value == value else 0.0 for value in sex_values]
    site_one_hot = [1.0 if site_value == value else 0.0 for value in site_values]
    return np.asarray(
        [age_scaled, age_missing, *sex_one_hot, *site_one_hot], dtype=np.float32
    )


def _ordered_narrative_rows_for_split(
    manifests_dir: Path,
    split: str,
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    narrative_manifest_path = manifests_dir / f"isic2019_narratives_split_{split}.jsonl"
    if not narrative_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing narrative manifest for split={split}: {narrative_manifest_path}"
        )
    narrative_rows = _load_jsonl(narrative_manifest_path)
    narr_by_stem = {_image_stem_from_narrative_row(r): r for r in narrative_rows}

    ordered_narr_rows: list[dict[str, Any]] = []
    for row in split_rows:
        stem = _image_stem_from_manifest_row(row)
        narr = narr_by_stem.get(stem)
        if narr is None:
            raise KeyError(
                f"No narrative row found for split={split} sample stem={stem}"
            )
        ordered_narr_rows.append(narr)
    return ordered_narr_rows


def _encode_texts(
    encoder: SentenceTransformerEncoder,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    emb = encoder.encode_batch(texts, batch_size=batch_size, show_progress=False)
    return emb.astype(np.float32)


def _write_bundle(
    path: Path, images: np.ndarray, embeddings: np.ndarray, labels: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images, embeddings=embeddings, labels=labels)


def prepare_targeted_ablation_bundles(
    base_bundle_dir: str = "/data/bundled",
    manifests_dir: str = "/root/assets/manifests",
    output_dir: str = "/data/bundled_ablation",
    seeds: tuple[int, ...] = (0, 1, 2),
    batch_size: int = 384,
) -> dict[str, Any]:
    """Create bundle assets for targeted paper ablations.

    Outputs:
    - metadata-only bundles: {output_dir}/metadata_only/{train,val,test}.npz
    - template-full bundles: {output_dir}/template_full/{train,val,test}.npz
    - train-shuffle bundles: {output_dir}/train_shuffle/seed{S}/train.npz
    - field-test bundles: {output_dir}/field/{symptoms,temporal,risk}/test.npz
    """
    base = Path(base_bundle_dir)
    manifests = Path(manifests_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_manifests = {
        "train": _load_jsonl(manifests / "split_train.jsonl"),
        "val": _load_jsonl(manifests / "split_val.jsonl"),
        "test": _load_jsonl(manifests / "split_test.jsonl"),
    }
    split_bundles = {
        "train": np.load(base / "train.npz", allow_pickle=False),
        "val": np.load(base / "val.npz", allow_pickle=False),
        "test": np.load(base / "test.npz", allow_pickle=False),
    }

    for split in ("train", "val", "test"):
        n_manifest = len(split_manifests[split])
        n_bundle = int(split_bundles[split]["labels"].shape[0])
        if n_manifest != n_bundle:
            raise ValueError(
                f"Split size mismatch for {split}: manifest={n_manifest}, bundle={n_bundle}"
            )

    encoder = SentenceTransformerEncoder()

    # Metadata-only bundles for train/val/test.
    metadata_dir = out / "metadata_only"
    for split in ("train", "val", "test"):
        rows = split_manifests[split]
        texts = [_metadata_text(r) for r in rows]
        emb = _encode_texts(encoder, texts, batch_size=batch_size)
        bundle = split_bundles[split]
        _write_bundle(
            metadata_dir / f"{split}.npz",
            images=bundle["images"],
            embeddings=emb,
            labels=bundle["labels"],
        )

    # Style-control bundles: deterministic template text with full narrative fields.
    template_full_dir = out / "template_full"
    for split in ("train", "val", "test"):
        rows = split_manifests[split]
        ordered_narr_rows = _ordered_narrative_rows_for_split(manifests, split, rows)
        texts = [_template_full_text(r) for r in ordered_narr_rows]
        emb = _encode_texts(encoder, texts, batch_size=batch_size)
        bundle = split_bundles[split]
        _write_bundle(
            template_full_dir / f"{split}.npz",
            images=bundle["images"],
            embeddings=emb,
            labels=bundle["labels"],
        )

    # Train-time shuffle bundles (seed-specific permutation over train embeddings).
    train_shuffle_dir = out / "train_shuffle"
    train_bundle = split_bundles["train"]
    train_embeddings = train_bundle["embeddings"].astype(np.float32)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_embeddings.shape[0])
        shuffled = train_embeddings[perm]
        _write_bundle(
            train_shuffle_dir / f"seed{seed}" / "train.npz",
            images=train_bundle["images"],
            embeddings=shuffled,
            labels=train_bundle["labels"],
        )

    # Field-ablation test bundles (symptoms / temporal / risk).
    ordered_narr_rows = _ordered_narrative_rows_for_split(
        manifests_dir=manifests,
        split="test",
        split_rows=split_manifests["test"],
    )

    field_to_text_fn = {
        "symptoms": _symptoms_text,
        "temporal": _temporal_text,
        "risk": _risk_text,
    }
    test_bundle = split_bundles["test"]
    field_dir = out / "field"
    for field, fn in field_to_text_fn.items():
        texts = [fn(r) for r in ordered_narr_rows]
        emb = _encode_texts(encoder, texts, batch_size=batch_size)
        _write_bundle(
            field_dir / field / "test.npz",
            images=test_bundle["images"],
            embeddings=emb,
            labels=test_bundle["labels"],
        )

    return {
        "metadata_only_dir": str(metadata_dir),
        "template_full_dir": str(template_full_dir),
        "train_shuffle_dir": str(train_shuffle_dir),
        "field_dir": str(field_dir),
        "seeds": list(seeds),
        "counts": {
            split: len(split_manifests[split]) for split in ("train", "val", "test")
        },
    }


def prepare_structured_metadata_bundles(
    base_bundle_dir: str = "/data/bundled",
    manifests_dir: str = "/root/assets/manifests",
    output_dir: str = "/data/bundled_structured_meta",
) -> dict[str, Any]:
    """Create train/val/test bundles with direct structured metadata features."""
    base = Path(base_bundle_dir)
    manifests = Path(manifests_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_manifests = {
        "train": _load_jsonl(manifests / "split_train.jsonl"),
        "val": _load_jsonl(manifests / "split_val.jsonl"),
        "test": _load_jsonl(manifests / "split_test.jsonl"),
    }
    split_bundles = {
        "train": np.load(base / "train.npz", allow_pickle=False),
        "val": np.load(base / "val.npz", allow_pickle=False),
        "test": np.load(base / "test.npz", allow_pickle=False),
    }

    age_mean, age_std = _structured_meta_age_stats(split_manifests["train"])
    sex_values, site_values = _structured_meta_categories(split_manifests["train"])

    for split in ("train", "val", "test"):
        rows = split_manifests[split]
        bundle = split_bundles[split]
        if len(rows) != int(bundle["labels"].shape[0]):
            raise ValueError(
                f"Split size mismatch for {split}: manifest={len(rows)}, bundle={int(bundle['labels'].shape[0])}"
            )
        embeddings = np.stack(
            [
                _structured_meta_vector(
                    row,
                    age_mean=age_mean,
                    age_std=age_std,
                    sex_values=sex_values,
                    site_values=site_values,
                )
                for row in rows
            ],
            axis=0,
        ).astype(np.float32)
        _write_bundle(
            out / f"{split}.npz",
            images=bundle["images"],
            embeddings=embeddings,
            labels=bundle["labels"],
        )

    return {
        "output_dir": str(out),
        "feature_dim": 2 + len(sex_values) + len(site_values),
        "age_mean": age_mean,
        "age_std": age_std,
        "sex_values": sex_values,
        "site_values": site_values,
        "counts": {
            split: len(split_manifests[split]) for split in ("train", "val", "test")
        },
    }
