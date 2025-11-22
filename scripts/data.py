import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

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
    _use_buffer({"jsonl": "assets/manifests", "png": "assets/figures"})


@app.command()
def verify_image_manifests() -> None:
    """Verifies image manifests for data quality (runs locally)."""
    from src.pipelines.prep_images import verify_manifests

    verify_manifests()


@app.command()
def generate_narratives() -> None:
    """Generates narratives for all splits, downloads manifests locally."""
    print("Generating narratives on Modal...")
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::narratives_entrypoint"],
        check=True,
    )
    _use_buffer({"narrative_jsonl": "assets/manifests"})


@app.command()
def verify_narrative_manifests() -> None:
    """Verifies narrative manifests for data quality (runs locally)."""
    from src.pipelines.prep_narratives import verify_narrative_manifests

    verify_narrative_manifests()


def _use_buffer(file_mappings: dict) -> None:
    """Downloads buffer from Modal, moves files to target directories, and cleans up.

    Args:
        file_mappings: Dict mapping file types to target directories.
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
        if file.suffix == ".jsonl":
            if "narrative_jsonl" in file_mappings and "narrative" in file.name:
                target_dir = Path(file_mappings["narrative_jsonl"])
                target_dir.mkdir(parents=True, exist_ok=True)
                file.rename(target_dir / file.name)
                print(f"  Moved {file.name} to {target_dir}")
            elif "jsonl" in file_mappings:
                target_dir = Path(file_mappings["jsonl"])
                target_dir.mkdir(parents=True, exist_ok=True)
                file.rename(target_dir / file.name)
                print(f"  Moved {file.name} to {target_dir}")
        elif file.suffix == ".png" and "png" in file_mappings:
            target_dir = Path(file_mappings["png"])
            target_dir.mkdir(parents=True, exist_ok=True)
            file.rename(target_dir / file.name)
            print(f"  Moved {file.name} to {target_dir}")

    buffer_dir.rmdir()
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
    print("\nEnter comma-separated choices for multiple (e.g., '1,2,3')")

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
