import random
import subprocess
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm


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


def filter_one_per_patient(df: pl.DataFrame) -> pl.DataFrame:
    """Ensures only one image per patient, selecting randomly if multiple exist.

    Args:
        df: DataFrame with 'patient_id' column.

    Returns:
        Filtered DataFrame with one image per patient.
    """
    df_filled = df.with_columns(
        pl.col("patient_id").fill_null(pl.format("unknown_{}", pl.col("isic_id")))
    )
    df_with_row_num = df_filled.with_row_index("_row_idx")

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


def download_isic2019() -> None:
    """Downloads ISIC 2019 datasets from ISIC Archive using isic-cli."""
    print("Downloading ISIC 2019 datasets from ISIC Archive using isic-cli...")
    base_dir = Path("/data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    collections = {
        "training": {"id": 65, "dir": base_dir / "train" / "isic2019"},
        "test": {"id": 72, "dir": base_dir / "test" / "isic2019"},
    }

    for name, config in collections.items():
        print(f"\n  Downloading {name} set (collection {config['id']})...")
        config["dir"].mkdir(parents=True, exist_ok=True)

        print(f"    Downloading metadata...")
        metadata_output = config["dir"] / "metadata.csv"
        subprocess.run(
            [
                "isic",
                "metadata",
                "download",
                "--collections",
                str(config["id"]),
                "--outfile",
                str(metadata_output),
            ],
            check=True,
        )

        print(f"    Downloading images...")
        images_dir = config["dir"] / "images"
        subprocess.run(
            [
                "isic",
                "image",
                "download",
                "--collections",
                str(config["id"]),
                str(images_dir),
            ],
            check=True,
        )

        print(f"    Completed {name} set download")

    print("\nDownload complete!")


def verify_download() -> None:
    """Verifies downloaded data integrity by checking image and metadata counts."""
    print("\nVerifying data...")
    base_dir = Path("/data/raw")

    collections = {
        "training": {"dir": base_dir / "train" / "isic2019", "expected": 25331},
        "test": {"dir": base_dir / "test" / "isic2019", "expected": 8238},
    }

    for name, config in collections.items():
        print(f"\n  Verifying {name} set...")
        images_dir = config["dir"] / "images"
        metadata_path = config["dir"] / "metadata.csv"

        if images_dir.exists():
            image_files = list(images_dir.rglob("*.jpg"))
            image_count = len(image_files)
            print(f"    Images found: {image_count}")
        else:
            image_count = 0
            print(f"    ⚠ Images directory not found")

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_rows = sum(1 for _ in f) - 1
            print(f"    Metadata rows: {metadata_rows}")

            df = pl.read_csv(metadata_path)
            invalid_labels = df.filter(
                pl.col("diagnosis_3").map_elements(
                    get_label_from_diagnosis, return_dtype=pl.Int64
                )
                == -1
            )
            invalid_count = len(invalid_labels)

            if invalid_count > 0:
                print(
                    f"    ⚠ WARNING: {invalid_count} entries with unmapped labels (-1)"
                )
                print(f"      All diagnoses failing mapping:")
                for diagnosis in (
                    invalid_labels.select("diagnosis_3").unique().to_series()
                ):
                    print(f"        - {diagnosis}")
            else:
                print(f"    ✓ All labels mapped successfully")
        else:
            metadata_rows = 0
            print(f"    ⚠ Metadata CSV not found")

        expected = config["expected"]
        if image_count == metadata_rows == expected:
            print(f"    ✓ All counts aligned at {expected}")
        else:
            print(f"    ⚠ Count mismatch:")
            print(f"      Expected: {expected}")
            print(f"      Images: {image_count}")
            print(f"      Metadata: {metadata_rows}")


def process_isic2019(
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

    df = pl.read_csv(metadata_path)
    print(f"Initial dataset size: {len(df)}")

    df = filter_by_filename(df)
    print(f"After filename filtering: {len(df)}")

    df = filter_by_label(df)
    print(f"After label filtering: {len(df)}")

    df = filter_one_per_patient(df)
    print(f"After per-patient filtering: {len(df)}")

    df = df.sample(fraction=1.0, shuffle=True, seed=42)

    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    print(f"Train split: {len(train_df)}")
    print(f"Val split: {len(val_df)}")

    return {"train": train_df, "val": val_df}


def create_manifests(train_ratio: float = 0.9) -> None:
    """Creates train/val manifests and generates balance summaries.

    Args:
        train_ratio: Ratio of data to use for training (default 0.9).
    """
    print("Starting data processing pipeline...")
    data_dir = Path("/data/raw/train/isic2019")
    images_dir = data_dir / "images"
    splits = process_isic2019(data_dir, train_ratio=train_ratio)

    print("\nConverting to manifests...")
    buffer_dir = Path("/data/buffer")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {}
    for split, df in splits.items():
        manifest_path = buffer_dir / f"split_{split}.jsonl"
        print(f"Writing {split} manifest to {manifest_path}...")

        manifest_list = df_to_manifest_dict(df, images_dir)

        with jsonlines.open(manifest_path, mode="w") as writer:
            writer.write_all(manifest_list)

        print(f"Saved {len(manifest_list)} entries to {manifest_path}")
        manifest_data[split] = manifest_list

    print("\nGenerating balance summaries...")
    _generate_balance_summaries(splits, buffer_dir)
    print(f"\nAll outputs saved to {buffer_dir}")


def verify_manifests() -> None:
    """Verifies created manifests for data quality and label validity."""
    print("\nVerifying manifests...")
    buffer_dir = Path("/data/buffer")

    if not buffer_dir.exists():
        print(
            "  ⚠ Buffer directory not found. Run 'process' first to create manifests."
        )
        return

    manifest_files = list(buffer_dir.glob("split_*.jsonl"))

    if not manifest_files:
        print("  ⚠ No manifest files found in buffer directory.")
        return

    for manifest_file in sorted(manifest_files):
        split_name = manifest_file.stem.replace("split_", "")
        print(f"\n  Verifying {split_name} manifest...")

        entries = []
        with jsonlines.open(manifest_file) as reader:
            entries = list(reader)

        print(f"    Total entries: {len(entries)}")

        labels = [entry.get("label", -1) for entry in entries]
        invalid_count = sum(1 for label in labels if label == -1)

        if invalid_count > 0:
            print(f"    ⚠ ERROR: {invalid_count} entries with invalid labels (-1)")
        else:
            print(f"    ✓ All labels valid (0-7)")

        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"    Label distribution:")
        for label in sorted(label_counts.keys()):
            print(f"      Label {label}: {label_counts[label]} entries")


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
        label = get_label_from_diagnosis(diagnosis)

        if label == -1:
            continue

        manifest_entry = {
            "image_path": str(images_dir / f"{isic_id}.jpg"),
            "label": label,
            "patient_id": row.get("patient_id", ""),
            "age_approx": row.get("age_approx"),
            "sex": row.get("sex", "").lower() if row.get("sex") else None,
            "anatom_site_general": row.get("anatom_site_general"),
        }

        manifests.append(manifest_entry)

    return manifests


def _generate_balance_summaries(
    splits: dict[str, pl.DataFrame], output_dir: Path
) -> None:
    """Generates and saves balance summary visualizations.

    Args:
        splits: Dictionary with 'train' and 'val' DataFrames.
        output_dir: Directory to save summary images.
    """
    label_names = {
        0: "Melanoma",
        1: "Melanocytic nevus",
        2: "Basal cell carcinoma",
        3: "Actinic keratosis",
        4: "Benign keratosis",
        5: "Dermatofibroma",
        6: "Vascular lesion",
        7: "Squamous cell carcinoma",
    }

    for split_name, df in splits.items():
        df_with_labels = df.with_columns(
            pl.col("diagnosis_3")
            .map_elements(get_label_from_diagnosis, return_dtype=pl.Int64)
            .alias("label")
        )

        _plot_label_distribution(df_with_labels, label_names, split_name, output_dir)
        _plot_site_distribution(df_with_labels, label_names, split_name, output_dir)
        _plot_gender_distribution(df_with_labels, label_names, split_name, output_dir)


def _plot_label_distribution(
    df: pl.DataFrame, label_names: dict, split_name: str, output_dir: Path
) -> None:
    """Plots class-wise label distribution."""
    label_counts = df.group_by("label").agg(pl.len().alias("count")).sort("label")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [
        label_names.get(row["label"], f"Label {row['label']}")
        for row in label_counts.iter_rows(named=True)
    ]
    counts = [row["count"] for row in label_counts.iter_rows(named=True)]

    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Label Distribution - {split_name.capitalize()} Split")
    plt.tight_layout()
    plt.savefig(output_dir / f"balance_label_{split_name}.png", dpi=150)
    plt.close()


def _plot_site_distribution(
    df: pl.DataFrame, label_names: dict, split_name: str, output_dir: Path
) -> None:
    """Plots anatomical site distribution by class and split."""
    site_label_counts = (
        df.group_by(["anatom_site_general", "label"])
        .agg(pl.len().alias("count"))
        .sort(["anatom_site_general", "label"])
    )

    sites = (
        site_label_counts.select("anatom_site_general").unique().to_series().to_list()
    )
    label_ids = sorted(df.select("label").unique().to_series().to_list())

    data_matrix = []
    for site in sites:
        row = []
        for label_id in label_ids:
            if site is None:
                count = (
                    site_label_counts.filter(
                        pl.col("anatom_site_general").is_null()
                        & (pl.col("label") == label_id)
                    )
                    .select("count")
                    .to_series()
                )
            else:
                count = (
                    site_label_counts.filter(
                        (pl.col("anatom_site_general") == site)
                        & (pl.col("label") == label_id)
                    )
                    .select("count")
                    .to_series()
                )
            row.append(count[0] if len(count) > 0 else 0)
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(label_ids)))
    ax.set_xticklabels(
        [label_names.get(lid, f"L{lid}") for lid in label_ids], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([s if s else "Unknown" for s in sites])
    ax.set_xlabel("Class")
    ax.set_ylabel("Anatomical Site")
    ax.set_title(f"Site Distribution by Class - {split_name.capitalize()} Split")

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"balance_site_{split_name}.png", dpi=150)
    plt.close()


def _plot_gender_distribution(
    df: pl.DataFrame, label_names: dict, split_name: str, output_dir: Path
) -> None:
    """Plots gender distribution by class and split."""
    gender_label_counts = (
        df.group_by(["sex", "label"])
        .agg(pl.len().alias("count"))
        .sort(["sex", "label"])
    )

    label_ids = sorted(df.select("label").unique().to_series().to_list())
    genders = ["male", "female"]

    male_counts = []
    female_counts = []

    for label_id in label_ids:
        male_count = (
            gender_label_counts.filter(
                (pl.col("sex") == "male") & (pl.col("label") == label_id)
            )
            .select("count")
            .to_series()
        )
        male_counts.append(male_count[0] if len(male_count) > 0 else 0)

        female_count = (
            gender_label_counts.filter(
                (pl.col("sex") == "female") & (pl.col("label") == label_id)
            )
            .select("count")
            .to_series()
        )
        female_counts.append(female_count[0] if len(female_count) > 0 else 0)

    x = range(len(label_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], male_counts, width, label="Male")
    ax.bar([i + width / 2 for i in x], female_counts, width, label="Female")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_names.get(lid, f"L{lid}") for lid in label_ids], rotation=45, ha="right"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Gender Distribution by Class - {split_name.capitalize()} Split")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"balance_gender_{split_name}.png", dpi=150)
    plt.close()
