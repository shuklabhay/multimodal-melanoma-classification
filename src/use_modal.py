import os

import modal
from modal import App, Image, Volume

from src.pipelines.prep_images import (
    create_manifests,
    download_isic2019,
    verify_download,
)
from src.pipelines.prep_narratives import generate_narrative_texts

app = App(
    "lesion-classification",
    image=Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_dir("assets", "/root/assets"),
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
def process_narratives() -> None:
    """Generates narratives from pre-generated manifests using Nous API.

    Expects narrative manifests to already exist in /root/assets/manifests.
    Only generates text content and saves to volume.
    """
    generate_narrative_texts()


@app.local_entrypoint()
def narratives_entrypoint() -> None:
    """Local entrypoint to generate narratives."""
    print("Calling Modal to generate narrative texts...")
    process_narratives.remote()
    print("✓ Narrative text generation complete on Modal")
