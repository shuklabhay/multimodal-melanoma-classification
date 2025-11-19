import subprocess
from pathlib import Path

from modal import App, Image, Volume

app = App(
    "lesion-classification",
    image=Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "curl")
    .uv_sync(),
    volumes={
        "/data": Volume.from_name("data", create_if_missing=True),
    },
)


@app.function()
def data(step: str) -> None:
    """Runs Stage 0 data workflows inside Modal.

    Args:
        step: workflow step to run ('all', 'download', 'extract', or 'verify').
    """

    if step in ["all", "download"]:
        print(f"Downloading ISIC 2019 from AWS S3...")
        download_dir = Path("/data/raw/isic2019")
        download_dir.mkdir(parents=True, exist_ok=True)

        isic_2019_files = {
            "images_zip": "https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip",
            "ground_truth": "https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv",
            "metadata": "https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Metadata.csv",
        }
        for name, url in isic_2019_files.items():
            filename = url.split("/")[-1]
            dest_path = download_dir / filename
            print(f"  Downloading {filename}...")
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "-A",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "-e",
                    "https://challenge.isic-archive.com/",
                    "--progress-bar",
                    "-o",
                    str(dest_path),
                    url,
                ],
                check=True,
            )
            print(f"    Saved to {dest_path}")

        print("Download complete!")

    if step in ["all", "extract"]:
        print(f"\nExtracting images...")
        zip_path = Path("/data/raw/isic2019/ISIC_2019_Training_Input.zip")
        extract_dir = Path("/data/raw/isic2019/images")
        extract_dir.mkdir(exist_ok=True)

        subprocess.run(
            ["unzip", "-q", str(zip_path), "-d", str(extract_dir)], check=True
        )
        print(f"  Extracted to {extract_dir}")

    if step in ["all", "verify"]:
        print(f"\nVerifying data...")
        images_dir = Path("/data/raw/isic2019/images")
        csv_dir = Path("/data/raw/isic2019")

        image_files = list(images_dir.rglob("*.jpg"))
        image_count = len(image_files)

        gt_path = csv_dir / "ISIC_2019_Training_GroundTruth.csv"
        metadata_path = csv_dir / "ISIC_2019_Training_Metadata.csv"

        print(f"  Images found: {image_count}")

        if gt_path.exists():
            with open(gt_path) as f:
                gt_rows = sum(1 for _ in f) - 1
            print(f"  Ground truth rows: {gt_rows}")
        else:
            gt_rows = 0
            print(f"  ⚠ Ground truth CSV not found")

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_rows = sum(1 for _ in f) - 1
            print(f"  Metadata rows: {metadata_rows}")
        else:
            metadata_rows = 0
            print(f"  ⚠ Metadata CSV not found")

        expected_images = 25331
        all_aligned = image_count == gt_rows == metadata_rows == expected_images

        if all_aligned:
            print(f"✓ All counts aligned at {expected_images}")
        else:
            print(f"⚠ Count mismatch detected:")
            print(f"    Expected: {expected_images}")
            print(f"    Images: {image_count}")
            print(f"    Ground truth: {gt_rows}")
            print(f"    Metadata: {metadata_rows}")


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
