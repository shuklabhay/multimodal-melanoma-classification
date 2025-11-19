import subprocess
from pathlib import Path

import jsonlines
from modal import App, Image, Volume

from src.pipelines.data_pipeline import df_to_manifest_dict, process_isic2019_train_data

app = App(
    "lesion-classification",
    image=Image.debian_slim(python_version="3.11").uv_sync(),
    volumes={
        "/data": Volume.from_name("data", create_if_missing=True),
    },
)


@app.function(timeout=86400)
def data(step: str) -> None:
    """
    Collect and sort data inside a Modal volume using the ISIC CLI.

    Args:
        step: workflow step to run ('all', 'download', or 'verify').
    """

    if step in ["all", "download"]:
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

    if step in ["all", "verify"]:
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


@app.function(timeout=86400)
def process() -> None:
    """Processes ISIC2019 training data to create train/val manifests and save to volume."""
    data_dir = Path("/data/raw/train/isic2019")
    images_dir = data_dir / "images"

    print("Starting data processing pipeline...")
    splits = process_isic2019_train_data(data_dir, train_ratio=0.9)

    print("\nConverting to manifests...")
    manifests_dir = Path("/data/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for split, df in splits.items():
        manifest_path = manifests_dir / f"split_{split}.jsonl"
        print(f"Writing {split} manifest to {manifest_path}...")

        manifest_list = df_to_manifest_dict(df, images_dir)

        with jsonlines.open(manifest_path, mode="w") as writer:
            writer.write_all(manifest_list)

        print(f"Saved {len(manifest_list)} entries to {manifest_path}")


@app.function()
def narratives(split: str = "train") -> None:
    """Generates or validates narratives inside Modal.

    Args:
        split: dataset split to process (default 'train').
    """
    raise NotImplementedError("Narratives task is not implemented.")


@app.function()
def visualize(report: str = "splits") -> None:
    """Produces Stage 0 QA visualizations inside Modal.

    Args:
        report: type of report to generate (default 'splits').
    """
    raise NotImplementedError("Visualize task is not implemented.")
