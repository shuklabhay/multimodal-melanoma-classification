import random
import subprocess
from pathlib import Path
from typing import cast

import jsonlines
import matplotlib.pyplot as plt
import polars as pl
from imagededup.methods import PHash
from tqdm import tqdm

from src.pipelines.dataset_config import SPLIT_COLORS, DatasetConfig


def _can_map_label(row: dict, config: DatasetConfig) -> bool:
    combined = " ".join(
        [
            str(row.get(col, "")).lower()
            for col in config.diagnosis_columns
            if row.get(col)
        ]
    )
    for label_cfg in config.label_config.values():
        for pattern in label_cfg["diagnoses"]:
            if pattern.lower() in combined:
                return True
    return False


def filter_by_label(df: pl.DataFrame, config: DatasetConfig) -> pl.DataFrame:
    df_with_flag = df.with_columns(
        pl.struct(config.diagnosis_columns)
        .map_elements(
            lambda x: _can_map_label(x, config),
            return_dtype=pl.Boolean,
        )
        .alias("_can_map")
    )
    return df_with_flag.filter(pl.col("_can_map")).drop("_can_map")


def deduplicate_images(
    df: pl.DataFrame,
    config: DatasetConfig,
    images_dir: Path,
    max_distance_threshold: int = 2,
) -> pl.DataFrame:
    image_ids = df.select(config.image_id_column).to_series().to_list()
    existing_images = set()
    for image_id in image_ids:
        img_path = images_dir / f"{image_id}.jpg"
        if img_path.exists():
            existing_images.add(image_id)

    if not existing_images:
        return df

    hasher = PHash()
    encodings = hasher.encode_images(image_dir=str(images_dir), num_enc_workers=8)
    filtered_encodings = {
        k: v for k, v in encodings.items() if k.replace(".jpg", "") in existing_images
    }

    if not filtered_encodings:
        return df

    duplicates = hasher.find_duplicates(
        encoding_map=filtered_encodings,
        max_distance_threshold=max_distance_threshold,
    )

    to_remove = set()
    for img, dups in duplicates.items():
        for dup in dups:
            if img < dup:
                to_remove.add(dup.replace(".jpg", ""))
            else:
                to_remove.add(img.replace(".jpg", ""))

    if to_remove:
        return df.filter(~pl.col(config.image_id_column).is_in(list(to_remove)))

    return df


def download_from_isic_archive(collection_id: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading metadata for collection {collection_id}...")
    subprocess.run(
        [
            "isic",
            "metadata",
            "download",
            "--collections",
            str(collection_id),
            "--outfile",
            str(output_dir / "metadata.csv"),
        ],
        check=True,
    )

    print(f"  Downloading images for collection {collection_id}...")
    subprocess.run(
        [
            "isic",
            "image",
            "download",
            "--collections",
            str(collection_id),
            str(output_dir / "images"),
        ],
        check=True,
    )


def get_label_from_row(row: dict, config: DatasetConfig) -> int:
    combined = " ".join(
        [
            str(row.get(col, "")).lower()
            for col in config.diagnosis_columns
            if row.get(col)
        ]
    )

    for label_id, label_cfg in config.label_config.items():
        for pattern in label_cfg["diagnoses"]:
            if pattern.lower() in combined:
                return label_id

    diagnosis_values = {col: row.get(col) for col in config.diagnosis_columns}
    raise ValueError(f"Could not map diagnosis to label: {diagnosis_values}")


def download_dataset(config: DatasetConfig) -> None:
    print(f"Downloading {config.name} datasets...")
    download_from_isic_archive(
        config.train_collection_id, Path(f"/data/raw/train/{config.name}")
    )
    download_from_isic_archive(
        config.test_collection_id, Path(f"/data/raw/test/{config.name}")
    )
    print(f"\n✓ {config.name} download complete!")


def verify_download(config: DatasetConfig) -> None:
    print(f"\nVerifying {config.name} data...")
    base_dir = Path("/data/raw")

    collections = {
        "training": {
            "dir": base_dir / "train" / config.name,
            "expected": config.expected_train_count,
        },
        "test": {
            "dir": base_dir / "test" / config.name,
            "expected": config.expected_test_count,
        },
    }

    for name, coll_config in collections.items():
        print(f"\n  Verifying {name} set...")
        images_dir = coll_config["dir"] / "images"
        metadata_path = coll_config["dir"] / "metadata.csv"

        if images_dir.exists():
            image_files = list(images_dir.rglob("*.jpg"))
            image_count = len(image_files)
            print(f"    Images found: {image_count}")
        else:
            image_count = 0
            print("    ⚠ Images directory not found")

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_rows = sum(1 for _ in f) - 1
            print(f"    Metadata rows: {metadata_rows}")

            df = pl.read_csv(metadata_path)
            unmapped = []
            for row in df.iter_rows(named=True):
                try:
                    get_label_from_row(row, config)
                except ValueError:
                    diagnosis_values = {
                        col: row.get(col) for col in config.diagnosis_columns
                    }
                    unmapped.append(diagnosis_values)

            if unmapped:
                print(f"    ⚠ WARNING: {len(unmapped)} entries with unmapped labels")
                unique_unmapped = []
                seen = set()
                for d in unmapped:
                    key = tuple(sorted(d.items()))
                    if key not in seen:
                        seen.add(key)
                        unique_unmapped.append(d)
                for d in unique_unmapped[:10]:
                    print(f"      - {d}")
                if len(unique_unmapped) > 10:
                    print(f"      ... and {len(unique_unmapped) - 10} more")
            else:
                print("    ✓ All labels mapped successfully")
        else:
            metadata_rows = 0
            print("    ⚠ Metadata CSV not found")

        expected = coll_config["expected"]
        if image_count == metadata_rows == expected:
            print(f"    ✓ All counts aligned at {expected}")
        else:
            print("    ⚠ Count mismatch:")
            print(f"      Expected: {expected}")
            print(f"      Images: {image_count}")
            print(f"      Metadata: {metadata_rows}")


def split_by_column(
    df: pl.DataFrame,
    config: DatasetConfig,
    train_ratio: float = 0.9,
) -> dict[str, pl.DataFrame]:
    df_filled = df.with_columns(
        pl.col(config.split_by_column).fill_null(
            pl.format(f"unknown_{{}}", pl.col(config.image_id_column))
        )
    )

    unique_ids = (
        df_filled.select(config.split_by_column)
        .unique()
        .sort(config.split_by_column)
        .to_series()
        .to_list()
    )
    random.seed(42)
    random.shuffle(unique_ids)

    split_idx = int(len(unique_ids) * train_ratio)
    train_ids = set(unique_ids[:split_idx])
    val_ids = set(unique_ids[split_idx:])

    train_df = df_filled.filter(pl.col(config.split_by_column).is_in(train_ids))
    val_df = df_filled.filter(pl.col(config.split_by_column).is_in(val_ids))

    return {"train": train_df, "val": val_df}


def create_train_val_splits(
    config: DatasetConfig,
    data_dir: Path,
    train_ratio: float = 0.9,
    max_distance_threshold: int = 2,
) -> dict[str, pl.DataFrame]:
    metadata_path = data_dir / "metadata.csv"
    images_dir = data_dir / "images"

    df = pl.read_csv(metadata_path)
    print(f"Initial dataset size: {len(df)}")

    for filter_fn in config.extra_filters:
        df = filter_fn(df)
        print(f"After {filter_fn.__name__}: {len(df)}")

    df = filter_by_label(df, config)
    print(f"After filter_by_label: {len(df)}")

    df = deduplicate_images(df, config, images_dir, max_distance_threshold)
    print(f"After deduplicate_images: {len(df)}")

    df = df.sort(config.image_id_column)
    splits = split_by_column(df, config, train_ratio=train_ratio)

    unique_ids = len(df.select(config.split_by_column).unique())
    print(f"Total unique {config.split_by_column}s: {unique_ids}")
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")

    return splits


def create_test_split(
    config: DatasetConfig,
    data_dir: Path,
    max_distance_threshold: int = 2,
) -> dict[str, pl.DataFrame]:
    metadata_path = data_dir / "metadata.csv"
    images_dir = data_dir / "images"

    df = pl.read_csv(metadata_path)
    print(f"Initial test dataset size: {len(df)}")

    for filter_fn in config.extra_filters:
        df = filter_fn(df)
        print(f"After {filter_fn.__name__}: {len(df)}")

    df = filter_by_label(df, config)
    print(f"After filter_by_label: {len(df)}")

    df = deduplicate_images(df, config, images_dir, max_distance_threshold)
    print(f"After deduplicate_images: {len(df)}")

    print(f"Final test split: {len(df)}")

    return {"test": df}


def df_to_manifest(
    df: pl.DataFrame, images_dir: Path, config: DatasetConfig
) -> list[dict]:
    manifests = []

    for row in tqdm(
        df.iter_rows(named=True),
        total=len(df),
        desc=f"Generating {config.name} manifests",
    ):
        image_id = row.get(config.image_id_column, "")
        manifest_entry = {
            "image_path": str(images_dir / f"{image_id}.jpg"),
            "label": row.get("label"),
            config.id_column_for_manifest: row.get(config.id_column_for_manifest, ""),
            "age_approx": row.get("age_approx"),
            "sex": row.get("sex", "").lower() if row.get("sex") else None,
            "anatom_site_general": row.get("anatom_site_general"),
        }
        manifests.append(manifest_entry)

    return manifests


def create_manifests(
    config: DatasetConfig,
    train_ratio: float = 0.9,
    max_distance_threshold: int = 2,
) -> None:
    print(f"Starting {config.name} data processing pipeline...")
    train_data_dir = Path(f"/data/raw/train/{config.name}")
    test_data_dir = Path(f"/data/raw/test/{config.name}")
    images_dir_train = train_data_dir / "images"
    images_dir_test = test_data_dir / "images"

    train_val_splits = create_train_val_splits(
        config,
        train_data_dir,
        train_ratio=train_ratio,
        max_distance_threshold=max_distance_threshold,
    )
    test_split = create_test_split(
        config,
        test_data_dir,
        max_distance_threshold=max_distance_threshold,
    )
    splits = {**train_val_splits, **test_split}

    print("\nConverting to manifests...")
    buffer_dir = Path("/data/buffer")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    splits_with_labels = {}
    for split, df in splits.items():
        df_with_labels = df.with_columns(
            pl.struct(config.diagnosis_columns)
            .map_elements(
                lambda x: get_label_from_row(x, config),
                return_dtype=pl.Int64,
            )
            .alias("label")
        )
        splits_with_labels[split] = df_with_labels

    for split, df in splits_with_labels.items():
        manifest_path = buffer_dir / f"{config.name}_split_{split}.jsonl"
        print(f"Writing {split} manifest to {manifest_path}...")

        current_images_dir = images_dir_test if split == "test" else images_dir_train
        manifest_list = df_to_manifest(df, current_images_dir, config)
        random.seed(40)
        random.shuffle(manifest_list)

        with jsonlines.open(manifest_path, mode="w") as writer:
            writer.write_all(manifest_list)

        print(f"Saved {len(manifest_list)} entries to {manifest_path}")

    print("\nGenerating balance summaries...")
    _generate_balance_summaries(
        splits_with_labels, buffer_dir, config.label_config, config.name
    )
    print(f"\n✓ {config.name} manifests saved to {buffer_dir}")


def generate_visualizations(
    config: DatasetConfig,
    manifests_dir: Path,
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        output_dir = manifests_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split_name in ["train", "val", "test"]:
        manifest_path = manifests_dir / f"{config.name}_split_{split_name}.jsonl"
        if not manifest_path.exists():
            continue

        entries = []
        with jsonlines.open(manifest_path) as reader:
            entries = list(reader)

        if entries:
            splits[split_name] = pl.DataFrame(entries)

    if not splits:
        print(f"No manifests found for {config.name} in {manifests_dir}")
        return

    print(f"Generating visualizations for {config.name}...")
    _generate_balance_summaries(splits, output_dir, config.label_config, config.name)
    print(f"✓ Visualizations saved to {output_dir}")


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

        labels = [entry.get("label") for entry in entries]
        valid_count = sum(1 for label in labels if label is not None)
        print(f"    ✓ {valid_count} entries with valid labels")

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
        print("\n  Checking for patient overlap between splits...")
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
            print("    ✓ No patient overlap between splits")


def _generate_balance_summaries(
    splits: dict[str, pl.DataFrame],
    output_dir: Path,
    label_config: dict,
    prefix: str,
) -> None:
    label_names = {label_id: cfg["name"] for label_id, cfg in label_config.items()}
    plot_label_distribution(splits, label_names, output_dir, prefix)
    plot_site_distribution(splits, label_names, output_dir, prefix)
    plot_gender_distribution(splits, label_names, output_dir, prefix)
    plot_age_distribution(splits, label_names, output_dir, prefix)


def plot_label_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path, prefix: str
) -> None:
    """Plots train vs val vs test label distribution on one plot with different colors."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    all_label_ids = sorted(label_names.keys())
    x = range(len(all_label_ids))
    width = 0.25

    for i, split_name in enumerate(["train", "val", "test"]):
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
            color=SPLIT_COLORS[split_name],
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
    plt.savefig(output_dir / f"{prefix}_balance_label.png", dpi=150)
    plt.close()


def plot_site_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path, prefix: str
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

    for ax, split_name in zip(axes, list(SPLIT_COLORS.keys())):
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
    plt.savefig(output_dir / f"{prefix}_balance_site.png", dpi=150)
    plt.close()


def plot_gender_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path, prefix: str
) -> None:
    """Plots train vs val vs test gender distribution on one chart with grouped bars."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    all_label_ids = sorted(label_names.keys())
    x = range(len(all_label_ids))
    width = 0.12

    for i, split_name in enumerate(["train", "val", "test"]):
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
            color=SPLIT_COLORS[split_name],
            alpha=0.8,
        )
        ax.bar(
            [xi + offset_female for xi in x],
            female_percentages,
            width,
            label=f"{split_name.capitalize()} Female (n={total_female})",
            color=SPLIT_COLORS[split_name],
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
    plt.savefig(output_dir / f"{prefix}_balance_gender.png", dpi=150)
    plt.close()


def plot_age_distribution(
    splits: dict[str, pl.DataFrame], label_names: dict, output_dir: Path, prefix: str
) -> None:
    """Plots train vs val vs test age distribution on one chart with violin plots."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    all_label_ids = sorted(label_names.keys())
    width = 0.25

    positions_list = []
    data_list = []
    colors_list = []

    for i, split_name in enumerate(["train", "val", "test"]):
        if split_name not in splits:
            continue

        df = splits[split_name]

        for j, label_id in enumerate(all_label_ids):
            label_data = (
                df.filter(pl.col("label") == label_id).select("age_approx").to_series()
            )
            label_data = label_data.filter(label_data.is_not_null())

            if len(label_data) > 0:
                position = j * (len(SPLIT_COLORS) * width + 0.2) + i * width
                positions_list.append(position)
                data_list.append(label_data.to_list())
                colors_list.append(SPLIT_COLORS[split_name])

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
        j * (len(SPLIT_COLORS) * width + 0.2) + width for j in range(len(all_label_ids))
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
            facecolor=SPLIT_COLORS[split_name], alpha=0.7, label=split_name.capitalize()
        )
        for split_name in SPLIT_COLORS
        if split_name in splits
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_balance_age.png", dpi=150)
    plt.close()
