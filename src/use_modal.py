import subprocess
from pathlib import Path

import jsonlines
from modal import App, Image, Volume

from src.pipelines.data_pipeline import (
    create_manifests,
    download_isic2019,
    verify_download,
    verify_manifests,
)

app = App(
    "lesion-classification",
    image=Image.debian_slim(python_version="3.11").uv_sync(),
    volumes={
        "/data": Volume.from_name("data", create_if_missing=True),
    },
)


@app.function(timeout=86400)
def data(step: str) -> None:
    """Collect and sort data inside a Modal volume using the ISIC CLI.

    Args:
        step: workflow step to run ('all', 'download', 'verify_download', 'verify_manifests').
    """
    if step in ["all", "download"]:
        download_isic2019()

    if step in ["all", "verify_download"]:
        verify_download()

    if step in ["verify_manifests"]:
        verify_manifests()


@app.function(timeout=86400)
def process() -> None:
    """Processes ISIC2019 training data to create train/val manifests and save to volume."""
    create_manifests(train_ratio=0.9)


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
