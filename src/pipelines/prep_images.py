import random
import subprocess
from pathlib import Path
from typing import cast

import jsonlines
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from src.pipelines.label_config import LABEL_CONFIG


def get_label_from_diagnosis(diagnosis_2: str, diagnosis_3: str) -> int:
    """Maps diagnosis text to integer label using substring matching.

    Args:
        diagnosis_2: Text from diagnosis_2 column.
        diagnosis_3: Text from diagnosis_3 column.

    Returns:
        Integer label 0-7, or -1 if no match found.
    """
    for diagnosis in [diagnosis_2, diagnosis_3]:
        if not diagnosis:
            continue

        d = diagnosis.lower()
        for label_id, config in LABEL_CONFIG.items():
            for diagnosis_str in config["diagnoses"]:
                if diagnosis_str.lower() in d:
                    return label_id

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
    """Drops entries with diagnoses that cannot be mapped to valid labels (0-7).

    Args:
        df: DataFrame with 'diagnosis_2' and 'diagnosis_3' columns.

    Returns:
        Filtered DataFrame with unmappable diagnoses removed.
    """
    df_with_labels = df.with_columns(
        pl.struct(["diagnosis_2", "diagnosis_3"])
        .map_elements(
            lambda x: get_label_from_diagnosis(x["diagnosis_2"], x["diagnosis_3"]),
            return_dtype=pl.Int64,
        )
        .alias("_temp_label")
    )
    return df_with_labels.filter(pl.col("_temp_label") != -1).drop("_temp_label")


def split_by_patient(
    df: pl.DataFrame, train_ratio: float = 0.9
) -> dict[str, pl.DataFrame]:
    """Splits data by patient ID to ensure no patient appears in both train and val.

    Args:
        df: DataFrame with 'patient_id' column.
        train_ratio: Ratio of patients to use for training (default 0.9).

    Returns:
        Dictionary with 'train' and 'val' DataFrames containing all images for assigned patients.
    """
    df_filled = df.with_columns(
        pl.col("patient_id").fill_null(pl.format("unknown_{}", pl.col("isic_id")))
    )

    unique_patients = (
        df_filled.select("patient_id").unique().sort("patient_id").to_series().to_list()
    )
    random.seed(42)
    random.shuffle(unique_patients)

    split_idx = int(len(unique_patients) * train_ratio)
    train_patients = set(unique_patients[:split_idx])
    val_patients = set(unique_patients[split_idx:])

    train_df = df_filled.filter(pl.col("patient_id").is_in(train_patients))
    val_df = df_filled.filter(pl.col("patient_id").is_in(val_patients))

    return {"train": train_df, "val": val_df}


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
                pl.struct(["diagnosis_2", "diagnosis_3"]).map_elements(
                    lambda x: get_label_from_diagnosis(
                        x["diagnosis_2"], x["diagnosis_3"]
                    ),
                    return_dtype=pl.Int64,
                )
                == -1
            )
            invalid_count = len(invalid_labels)

            if invalid_count > 0:
                print(
                    f"    ⚠ WARNING: {invalid_count} entries with unmapped labels (-1)"
                )
                print(f"      All diagnoses failing mapping:")
                for row in (
                    invalid_labels.select(["diagnosis_2", "diagnosis_3"])
                    .unique()
                    .iter_rows(named=True)
                ):
                    print(
                        f"        - diagnosis_2: {row['diagnosis_2']}, diagnosis_3: {row['diagnosis_3']}"
                    )
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


def create_train_val_splits(
    data_dir: Path, train_ratio: float = 0.9
) -> dict[str, pl.DataFrame]:
    """Main pipeline to process ISIC2019 training data with filtering and patient-level splitting.

    Args:
        data_dir: Path to the ISIC2019 training data directory (e.g., /data/raw/train/isic2019).
        train_ratio: Ratio of patients to use for training (default 0.9 for 90/10 split).

    Returns:
        Dictionary with 'train' and 'val' keys containing all images for assigned patients.
    """
    metadata_path = data_dir / "metadata.csv"

    df = pl.read_csv(metadata_path)
    print(f"Initial dataset size: {len(df)}")

    df = filter_by_filename(df)
    print(f"After filename filtering: {len(df)}")

    df = filter_by_label(df)
    print(f"After label filtering: {len(df)}")

    df = df.sort("isic_id")
    splits = split_by_patient(df, train_ratio=train_ratio)

    unique_patients = len(df.select("patient_id").unique())
    print(f"Total unique patients: {unique_patients}")
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")

    return splits


def create_test_split(data_dir: Path) -> dict[str, pl.DataFrame]:
    """Processes ISIC2019 test data.

    Args:
        data_dir: Path to the ISIC2019 test data directory.

    Returns:
        Dictionary with 'test' key containing filtered DataFrame.
    """
    metadata_path = data_dir / "metadata.csv"

    df = pl.read_csv(metadata_path)
    print(f"Initial test dataset size: {len(df)}")

    df = filter_by_filename(df)
    print(f"After filename filtering: {len(df)}")

    df = filter_by_label(df)
    print(f"After label filtering: {len(df)}")
    print(f"Final test split: {len(df)}")

    return {"test": df}


def create_manifests(train_ratio: float = 0.9) -> None:
    """Creates split manifests and generates balance summaries.

    Args:
        train_ratio: Ratio of data to use for training (default 0.9).
    """
    print("Starting data processing pipeline...")
    train_data_dir = Path("/data/raw/train/isic2019")
    test_data_dir = Path("/data/raw/test/isic2019")
    images_dir_train = train_data_dir / "images"
    images_dir_test = test_data_dir / "images"

    train_val_splits = create_train_val_splits(train_data_dir, train_ratio=train_ratio)
    test_split = create_test_split(test_data_dir)
    splits = {**train_val_splits, **test_split}

    print("\nConverting to manifests...")
    buffer_dir = Path("/data/buffer")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    splits_with_labels = {}
    for split, df in splits.items():
        df_with_labels = df.with_columns(
            pl.struct(["diagnosis_2", "diagnosis_3"])
            .map_elements(
                lambda x: get_label_from_diagnosis(x["diagnosis_2"], x["diagnosis_3"]),
                return_dtype=pl.Int64,
            )
            .alias("label")
        )
        splits_with_labels[split] = df_with_labels

    for split, df in splits_with_labels.items():
        manifest_path = buffer_dir / f"isic_split_{split}.jsonl"
        print(f"Writing {split} manifest to {manifest_path}...")

        current_images_dir = images_dir_test if split == "test" else images_dir_train
        manifest_list = df_to_manifest_dict(df, current_images_dir)
        random.seed(40)
        random.shuffle(manifest_list)

        with jsonlines.open(manifest_path, mode="w") as writer:
            writer.write_all(manifest_list)

        print(f"Saved {len(manifest_list)} entries to {manifest_path}")

    print("\nGenerating balance summaries...")
    _generate_balance_summaries(splits_with_labels, buffer_dir)
    print(f"\nAll outputs saved to {buffer_dir}")


def verify_manifests() -> None:
    """Verifies created manifests for data quality and label validity."""
    print("\nVerifying manifests...")
    manifests_dir = Path("assets/manifests")

    if not manifests_dir.exists():
        print(
            "  ⚠ Manifests directory not found. Run 'process' first to create manifests."
        )
        return

    manifest_files = list(manifests_dir.glob("split_*.jsonl"))

    if not manifest_files:
        print("  ⚠ No manifest files found in manifests directory.")
        return

    all_manifests = {}
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

        patient_ids = [
            entry.get("patient_id") for entry in entries if entry.get("patient_id")
        ]
        unique_patients = len(set(patient_ids))

        if unique_patients == len(patient_ids):
            print(f"    ✓ All patient IDs unique ({unique_patients} patients)")
        else:
            duplicate_count = len(patient_ids) - unique_patients
            print(f"    ⚠ WARNING: {duplicate_count} patients have multiple images")

        all_manifests[split_name] = entries

    if "train" in all_manifests and "val" in all_manifests:
        print(f"\n  Checking for patient overlap between splits...")
        train_patients = set(
            e.get("patient_id") for e in all_manifests["train"] if e.get("patient_id")
        )
        val_patients = set(
            e.get("patient_id") for e in all_manifests["val"] if e.get("patient_id")
        )

        overlap = train_patients & val_patients
        if overlap:
            print(
                f"    ⚠ ERROR: {len(overlap)} patients appear in both train and val splits"
            )
        else:
            print(f"    ✓ No patient overlap between splits")


def df_to_manifest_dict(df: pl.DataFrame, images_dir: Path) -> list[dict]:
    """Converts Polars DataFrame to list of manifest dictionaries.

    Args:
        df: DataFrame with ISIC2019 metadata (already filtered, with label column).
        images_dir: Path to images directory for constructing full paths.

    Returns:
        List of dictionaries with manifest schema.
    """
    manifests = []

    for row in tqdm(
        df.iter_rows(named=True), total=len(df), desc="Generating manifests"
    ):
        manifest_entry = {
            "image_path": str(images_dir / f"{row.get('isic_id', '')}.jpg"),
            "label": row.get("label"),
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
        splits: Dictionary with 'train' and 'val' DataFrames (already filtered, with labels).
        output_dir: Directory to save summary images.
    """
    label_names = {
        label_id: config["name"] for label_id, config in LABEL_CONFIG.items()
    }
    _plot_label_distribution(splits, label_names, output_dir)
    _plot_site_distribution(splits, label_names, output_dir)
    _plot_gender_distribution(splits, label_names, output_dir)
    _plot_age_distribution(splits, label_names, output_dir)


def _plot_label_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path
) -> None:
    """Plots train vs val vs test label distribution on one plot with different colors."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    all_label_ids = sorted(label_names.keys())
    x = range(len(all_label_ids))
    width = 0.25

    for i, split_name in enumerate(split_order):
        if split_name not in splits:
            continue

        df = splits[split_name]
        total_samples = len(df)
        label_counts = df.group_by("label").agg(pl.len().alias("count")).sort("label")

        counts_dict = {
            row["label"]: row["count"] for row in label_counts.iter_rows(named=True)
        }
        counts = [counts_dict.get(label_id, 0) for label_id in all_label_ids]
        percentages = [(count / total_samples) * 100 for count in counts]

        offset = width * (i - 1)
        ax.bar(
            [xi + offset for xi in x],
            percentages,
            width,
            label=f"{split_name.capitalize()} (n={total_samples})",
            color=split_colors[split_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_names[lid] for lid in all_label_ids],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_label.png", dpi=150)
    plt.close()


def _plot_site_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path
) -> None:
    """Plots train vs val vs test site distribution side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    all_sites_set = set()
    for df in splits.values():
        sites_in_split = df.select("anatom_site_general").unique().to_series().to_list()
        all_sites_set.update(sites_in_split)

    all_sites = sorted([s for s in all_sites_set if s is not None]) + (
        [None] if None in all_sites_set else []
    )

    split_order = ["train", "val", "test"]
    for ax, split_name in zip(axes, split_order):
        if split_name not in splits:
            continue
        df = splits[split_name]
        total_samples = len(df)

        site_label_counts = (
            df.group_by(["anatom_site_general", "label"])
            .agg(pl.len().alias("count"))
            .sort(["anatom_site_general", "label"])
        )

        label_ids = sorted(df.select("label").unique().to_series().to_list())

        data_matrix = []
        for site in all_sites:
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
                count_val = count[0] if len(count) > 0 else 0
                percentage = (count_val / total_samples) * 100
                row.append(percentage)
            data_matrix.append(row)

        im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(label_ids)))
        ax.set_xticklabels(
            [label_names.get(lid, f"L{lid}") for lid in label_ids],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(all_sites)))
        ax.set_yticklabels([s if s else "Unknown" for s in all_sites])
        ax.set_xlabel("Class")
        ax.set_ylabel("Anatomical Site")
        ax.set_title(f"{split_name.capitalize()} Split (n={total_samples})")
        plt.colorbar(im, ax=ax, label="Percentage (%)")

    fig.suptitle("Site Distribution by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "balance_site.png", dpi=150)
    plt.close()


def _plot_gender_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path
) -> None:
    """Plots train vs val vs test gender distribution on one chart with grouped bars."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    all_label_ids = sorted(label_names.keys())
    x = range(len(all_label_ids))
    width = 0.12

    for i, split_name in enumerate(split_order):
        if split_name not in splits:
            continue

        df = splits[split_name]
        total_samples = len(df)
        gender_label_counts = (
            df.group_by(["sex", "label"])
            .agg(pl.len().alias("count"))
            .sort(["sex", "label"])
        )

        male_counts = []
        female_counts = []

        for label_id in all_label_ids:
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

        male_percentages = [(count / total_samples) * 100 for count in male_counts]
        female_percentages = [(count / total_samples) * 100 for count in female_counts]

        total_male = sum(male_counts)
        total_female = sum(female_counts)

        offset_male = width * (i * 2 - 2)
        offset_female = width * (i * 2 - 1)

        ax.bar(
            [xi + offset_male for xi in x],
            male_percentages,
            width,
            label=f"{split_name.capitalize()} Male (n={total_male})",
            color=split_colors[split_name],
            alpha=0.8,
        )
        ax.bar(
            [xi + offset_female for xi in x],
            female_percentages,
            width,
            label=f"{split_name.capitalize()} Female (n={total_female})",
            color=split_colors[split_name],
            alpha=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_names[lid] for lid in all_label_ids],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Gender Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_gender.png", dpi=150)
    plt.close()


def _plot_age_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path
) -> None:
    """Plots train vs val vs test age distribution on one chart with violin plots."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    all_label_ids = sorted(label_names.keys())
    width = 0.25

    positions_list = []
    data_list = []
    colors_list = []

    for i, split_name in enumerate(split_order):
        if split_name not in splits:
            continue

        df = splits[split_name]

        for j, label_id in enumerate(all_label_ids):
            label_data = (
                df.filter(pl.col("label") == label_id).select("age_approx").to_series()
            )
            label_data = label_data.filter(label_data.is_not_null())

            if len(label_data) > 0:
                position = j * (len(split_order) * width + 0.2) + i * width
                positions_list.append(position)
                data_list.append(label_data.to_list())
                colors_list.append(split_colors[split_name])

    parts = ax.violinplot(
        data_list,
        positions=positions_list,
        widths=width * 0.9,
        showmeans=True,
        showmedians=True,
    )

    for pc, color in zip(cast(list, parts["bodies"]), colors_list):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    x_positions = [
        j * (len(split_order) * width + 0.2) + width for j in range(len(all_label_ids))
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [label_names[lid] for lid in all_label_ids],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Age (years)")
    ax.set_title("Age Distribution by Class", fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=split_colors[split_name], alpha=0.7, label=split_name.capitalize()
        )
        for split_name in split_order
        if split_name in splits
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_age.png", dpi=150)
    plt.close()
