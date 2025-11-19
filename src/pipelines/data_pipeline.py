import random
from pathlib import Path
from typing import Any

import imagehash
import polars as pl
from PIL import Image
from tqdm import tqdm


def filter_by_filename(df: pl.DataFrame) -> pl.DataFrame:
    """Removes images with '_unknown' or '_downsampled' in their filenames.

    Args:
        df: DataFrame with image metadata including 'isic_id' column.

    Returns:
        Filtered DataFrame with problematic filenames removed.
    """
    return df.filter(
        ~pl.col("isic_id").str.contains("_unknown")
        & ~pl.col("isic_id").str.contains("_downsampled")
    )


def filter_by_label(df: pl.DataFrame) -> pl.DataFrame:
    """Drops entries with 'UNK' or 'UNKNOWN' diagnostic labels.

    Args:
        df: DataFrame with 'diagnosis_3' column.

    Returns:
        Filtered DataFrame with unknown labels removed.
    """
    return df.filter(
        ~pl.col("diagnosis_3").str.to_uppercase().is_in(["UNK", "UNKNOWN"])
    )


def filter_duplicates_by_hash(
    df: pl.DataFrame, images_dir: Path, threshold: int = 10
) -> pl.DataFrame:
    """Removes near-duplicate images using perceptual hashing (dhash).

    Args:
        df: DataFrame with image metadata including 'image_name' column.
        images_dir: Path to directory containing images.
        threshold: Maximum Hamming distance to consider images as duplicates.

    Returns:
        DataFrame with duplicates removed (keeping first occurrence).
    """
    hashes: dict[str, Any] = {}
    keep_indices: list[int] = []

    for idx, row in enumerate(
        tqdm(df.iter_rows(named=True), total=len(df), desc="Deduplicating images")
    ):
        isic_id = row["isic_id"]
        image_path = images_dir / f"{isic_id}.jpg"

        if not image_path.exists():
            continue

        img = Image.open(image_path)
        img_hash = imagehash.dhash(img)

        is_duplicate = False
        for existing_hash in hashes.values():
            if img_hash - existing_hash <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            hashes[isic_id] = img_hash
            keep_indices.append(idx)

    return df[keep_indices]


def filter_one_per_patient(df: pl.DataFrame) -> pl.DataFrame:
    """Ensures only one image per patient, selecting randomly if multiple exist.

    Args:
        df: DataFrame with 'patient_id' column.

    Returns:
        Filtered DataFrame with one image per patient.
    """
    df_with_row_num = df.with_row_index("_row_idx")

    grouped = df_with_row_num.group_by("patient_id").agg(
        [pl.col("_row_idx").alias("row_indices")]
    )

    selected_rows = []
    for group in tqdm(
        grouped.iter_rows(named=True), total=len(grouped), desc="Filtering patients"
    ):
        indices = group["row_indices"]
        selected_idx = random.choice(indices) if isinstance(indices, list) else indices
        selected_rows.append(selected_idx)

    return df.filter(pl.int_range(0, df.height).is_in(selected_rows))


def process_isic2019_train_data(
    data_dir: Path, train_ratio: float = 0.9
) -> dict[str, pl.DataFrame]:
    """Main pipeline to process ISIC2019 training data with filtering and splitting.

    Args:
        data_dir: Path to the ISIC2019 training data directory (e.g., /data/raw/train/isic2019).
        train_ratio: Ratio of data to use for training (default 0.9 for 90/10 split).

    Returns:
        Dictionary with 'train' and 'val' keys containing filtered DataFrames.
    """
    metadata_path = data_dir / "metadata.csv"
    images_dir = data_dir / "images"

    df = pl.read_csv(metadata_path)
    print(f"Initial dataset size: {len(df)}")

    df = filter_by_filename(df)
    print(f"After filename filtering: {len(df)}")

    df = filter_by_label(df)
    print(f"After label filtering: {len(df)}")

    df = filter_duplicates_by_hash(df, images_dir)
    print(f"After duplicate removal: {len(df)}")

    df = filter_one_per_patient(df)
    print(f"After per-patient filtering: {len(df)}")

    df = df.sample(fraction=1.0, shuffle=True, seed=42)

    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    print(f"Train split: {len(train_df)}")
    print(f"Val split: {len(val_df)}")

    return {"train": train_df, "val": val_df}


def get_label_from_diagnosis(diagnosis: str) -> int:
    """Maps diagnosis_3 text to integer label using substring matching.

    Args:
        diagnosis: Text from diagnosis_3 column.

    Returns:
        Integer label 0-7, or -1 if no match found.
    """
    if not diagnosis:
        return -1

    d = diagnosis.lower()
    label_mappings = {
        "Melanoma": 0,
        "Melanocytic nevus": 1,
        "nevus": 1,
        "Basal cell carcinoma": 2,
        "Actinic keratosis": 3,
        "actinic keratoses": 3,
        "Benign keratosis": 4,
        "seborrheic keratosis": 4,
        "solar lentigo": 4,
        "lichen planus-like keratosis": 4,
        "Dermatofibroma": 5,
        "Vascular lesion": 6,
        "Squamous cell carcinoma": 7,
    }

    for key, label in label_mappings.items():
        if key.lower() in d:
            return label

    return -1


def df_to_manifest_dict(df: pl.DataFrame, images_dir: Path) -> list[dict]:
    """Converts Polars DataFrame to list of manifest dictionaries.

    Args:
        df: DataFrame with ISIC2019 metadata.
        images_dir: Path to images directory for constructing full paths.

    Returns:
        List of dictionaries with manifest schema.
    """
    manifests = []

    for row in tqdm(
        df.iter_rows(named=True), total=len(df), desc="Generating manifests"
    ):
        isic_id = row.get("isic_id", "")
        diagnosis = row.get("diagnosis_3", "")

        manifest_entry = {
            "image_path": str(images_dir / f"{isic_id}.jpg"),
            "label": get_label_from_diagnosis(diagnosis),
            "patient_id": row.get("patient_id", ""),
            "age_approx": row.get("age_approx"),
            "sex": row.get("sex", "").lower() if row.get("sex") else None,
            "anatom_site_general": row.get("anatom_site_general"),
        }

        manifests.append(manifest_entry)

    return manifests
