import modal
from modal import App, Image, Volume

from src.pipelines.prep_images import (
    create_manifests,
    download_isic2019,
    verify_download,
)
from src.pipelines.prep_narratives import create_narrative_manifests

app = App(
    "lesion-classification",
    image=Image.debian_slim(python_version="3.11").uv_sync(),
    volumes={
        "/data": Volume.from_name("data", create_if_missing=True),
    },
)


@app.function(timeout=86400)
def images(step: str) -> None:
    """Collect and sort image data inside a Modal volume using the ISIC CLI.

    Args:
        step: workflow step to run ('all', 'download', 'verify_download').
    """
    if step in ["all", "download"]:
        download_isic2019()

    if step in ["all", "verify_download"]:
        verify_download()


@app.function(timeout=86400)
def process_images() -> None:
    """Processes ISIC2019 image data to create train/val/test manifests and save to volume."""
    create_manifests(train_ratio=0.9)


@app.function(secrets=[modal.Secret.from_dotenv()], timeout=86400)
def process_narratives(narrative_options: dict) -> None:
    """Generates narratives for all image manifests using Nous API.

    Args:
        narrative_options: Dictionary containing loaded narrative options.
    """
    create_narrative_manifests(narrative_options)


@app.local_entrypoint()
def narratives_entrypoint(
    narrative_options_path: str = "assets/narrative_options",
) -> None:
    """Local entrypoint to generate narratives, loading options locally and passing to Modal.

    Args:
        narrative_options_path: Path to narrative options directory.
    """
    from pathlib import Path

    from src.pipelines.prep_narratives import load_narrative_options

    options_dir = Path(narrative_options_path)
    print(f"Loading narrative options from {options_dir}...")
    narrative_options = load_narrative_options(options_dir)

    print("Calling Modal to generate narratives...")
    process_narratives.remote(narrative_options)
    print("✓ Narrative generation complete on Modal")


@app.function()
def visualize(report: str = "splits") -> None:
    """Produces Stage 0 QA visualizations inside Modal.

    Args:
        report: type of report to generate (default 'splits').
    """
    raise NotImplementedError("Visualize task is not implemented.")
