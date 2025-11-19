import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


@app.command()
def download() -> None:
    """Downloads ISIC 2019 dataset using ISIC CLI to Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::data",
            "--step",
            "download",
        ],
        check=True,
    )


@app.command()
def verify_download() -> None:
    """Verifies downloaded data integrity on Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::data",
            "--step",
            "verify_download",
        ],
        check=True,
    )


@app.command()
def verify_manifests() -> None:
    """Verifies created manifests for data quality (runs locally)."""
    from src.pipelines.data_pipeline import verify_manifests as verify_fn

    verify_fn()


@app.command()
def run_all() -> None:
    """Runs complete data setup workflow (download and verify)."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::data", "--step", "all"],
        check=True,
    )


@app.command()
def process() -> None:
    """Processes ISIC2019 data and creates split manifests locally."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::process"],
        check=True,
    )

    print("\nDownloading outputs from Modal Volume...")

    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            "--force",
            "data",
            "buffer",
            ".",
        ],
        check=True,
    )

    buffer_dir = Path("buffer")
    if buffer_dir.exists():
        manifests_dir = Path("assets/manifests")
        figures_dir = Path("assets/figures")
        manifests_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        for file in buffer_dir.iterdir():
            if file.suffix == ".jsonl":
                file.rename(manifests_dir / file.name)
                print(f"  Moved {file.name} to {manifests_dir}")
            elif file.suffix == ".png":
                file.rename(figures_dir / file.name)
                print(f"  Moved {file.name} to {figures_dir}")

        buffer_dir.rmdir()
        print(f"\nManifests saved to {manifests_dir}")
        print(f"Balance summaries saved to {figures_dir}")

    print("\nCleaning up Modal buffer...")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "rm",
            "data",
            "buffer",
            "-r",
        ],
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
    print("  1. Download data to Modal volume")
    print("  2. Verify downloaded data")
    print("  3. Create split manifests")
    print("  4. Verify manifests")
    print("  5. Exit")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        download()
    elif choice == "2":
        verify_download()
    elif choice == "3":
        process()
    elif choice == "4":
        verify_manifests()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
