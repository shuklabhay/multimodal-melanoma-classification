import subprocess
from pathlib import Path

from typer import Context, Typer

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
def verify() -> None:
    """Verifies data integrity on Modal volume."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::data", "--step", "verify"],
        check=True,
    )


@app.command()
def run_all() -> None:
    """Runs complete data setup workflow (download and verify)."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::data", "--step", "all"],
        check=True,
    )


@app.command()
def process() -> None:
    """Processes ISIC2019 data and creates train/val split manifests locally."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::process"],
        check=True,
    )

    print("\nDownloading manifests from Modal Volume...")
    manifests_dir = Path("assets/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            "data",
            "/data/manifests",
            "assets",
        ],
        check=True,
    )

    print(f"Manifests saved to {manifests_dir}")


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for ISIC2019 data management."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== ISIC2019 Data Management ===")
    print("What would you like to do?")
    print("  1. Download data to Modal volume")
    print("  2. Create train/val manifests")
    print("  3. Verify data")
    print("  4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        download()
    elif choice == "2":
        process()
    elif choice == "3":
        verify()
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
