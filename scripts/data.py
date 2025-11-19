import subprocess

from typer import Typer

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


if __name__ == "__main__":
    app()
