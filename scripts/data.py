import os
import shutil
import subprocess
import sys
from pathlib import Path

from typer import Context, Option, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


@app.command()
def run_all() -> None:
    """Runs complete data pipeline: download, process images, generate narratives."""
    download_images()
    verify_images()
    process_images()
    generate_narratives()


@app.command()
def download_images() -> None:
    """Downloads ISIC 2019 dataset to Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::images",
            "--step",
            "download",
        ],
        check=True,
    )


@app.command()
def verify_images() -> None:
    """Verifies downloaded image data integrity on Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::images",
            "--step",
            "verify_download",
        ],
        check=True,
    )


@app.command()
def process_images() -> None:
    """Creates image manifests and visualizations, downloads them locally."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::process_images"],
        check=True,
    )
    _use_buffer(
        {
            "isic_split_train.jsonl": "assets/manifests/isic_split_train.jsonl",
            "isic_split_val.jsonl": "assets/manifests/isic_split_val.jsonl",
            "isic_split_test.jsonl": "assets/manifests/isic_split_test.jsonl",
            "label_distribution.png": "assets/figures/label_distribution.png",
            "age_distribution.png": "assets/figures/age_distribution.png",
            "sex_distribution.png": "assets/figures/sex_distribution.png",
            "site_distribution.png": "assets/figures/site_distribution.png",
        }
    )


@app.command()
def verify_image_manifests() -> None:
    """Verifies image manifests for data quality (runs locally)."""
    from src.pipelines.prep_images import verify_manifests

    verify_manifests()


@app.command("generate-narratives")
def generate_narratives(
    resume_mode: bool = Option(
        False,
        "--continue",
        help="Continue generating existing narratives instead of regenerating all",
    )
) -> None:
    """Generates narratives for all splits, downloads manifests locally."""
    if resume_mode:
        print("Running in continue mode - will skip existing narratives")
    else:
        print("Running in fresh mode - will regenerate all narratives")

    cmd = ["uv", "run", "modal", "run", "-m", "src.use_modal::narratives_entrypoint"]
    if resume_mode:
        cmd.append("--continue-from-existing")

    subprocess.run(cmd, check=True)
    _use_buffer(
        {
            "narrative_split_train.jsonl": "assets/manifests/narrative_split_train.jsonl",
            "narrative_split_val.jsonl": "assets/manifests/narrative_split_val.jsonl",
            "narrative_split_test.jsonl": "assets/manifests/narrative_split_test.jsonl",
            "balance_style.png": "assets/figures/balance_style.png",
            "balance_memory.png": "assets/figures/balance_memory.png",
            "balance_symptoms.png": "assets/figures/balance_symptoms.png",
        }
    )


@app.command()
def verify_narrative_manifests() -> None:
    """Verifies narrative manifests for data quality (runs locally)."""
    from src.pipelines.prep_narratives import verify_narrative_manifests

    verify_narrative_manifests()


def _use_buffer(file_map: dict[str, str]) -> None:
    """Downloads buffer from Modal, moves mapped files to destinations, deletes rest.

    Args:
        file_map: Dict mapping buffer filenames to local destination paths.
    """

    print("\nDownloading outputs from Modal Volume...")
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", "--force", "data", "buffer", "."],
        check=True,
    )

    buffer_dir = Path("buffer")
    if not buffer_dir.exists():
        return

    for file in buffer_dir.iterdir():
        if file.name in file_map:
            dest_path = Path(file_map[file.name])
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            file.rename(dest_path)
            print(f"  Moved {file.name} to {dest_path}")
        else:
            file.unlink()
            print(f"  Deleted {file.name}")

    shutil.rmtree(buffer_dir)
    subprocess.run(
        ["uv", "run", "modal", "volume", "rm", "data", "buffer", "-r"],
        check=True,
    )
    print("✓ Modal buffer cleared")


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for ISIC2019 data management."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== ISIC2019 Data Management ===")
    print("What would you like to do?")
    print("  1. Download images")
    print("  2. Verify download")
    print("  3. Create image manifests")
    print("  4. Generate narratives")
    print("  5. Run all")
    print("  6. Exit")
    choice = input("\nEnter your choice: ").strip().lower()

    steps_map = {
        "1": download_images,
        "2": verify_images,
        "3": process_images,
        "4": generate_narratives,
        "5": run_all,
    }

    if choice == "6":
        print("Exiting...")
        return

    if "," in choice:
        selected_steps = [s.strip() for s in choice.split(",")]
        invalid_steps = [s for s in selected_steps if s not in steps_map]

        if invalid_steps:
            print(f"Invalid step(s): {', '.join(invalid_steps)}. Please use 1-6.")
            return

        print(f"\nRunning steps: {', '.join(selected_steps)}\n")
        for step_num in selected_steps:
            print(f"Running step {step_num}...")
            steps_map[step_num]()
    elif choice in steps_map:
        steps_map[choice]()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
