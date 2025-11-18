from __future__ import annotations

from modal import App, Image, Volume

app = App(
    "lesion-classification",
    tags={"environment": "apresearch26"},
    image=Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_dir("assets", "/assets", copy=False)
    .add_local_dir("config", "/config", copy=False),
    volumes={
        "/data": Volume.from_name("data", create_if_missing=True),
        "/results": Volume.from_name("results", create_if_missing=True),
    },
)


@app.function()
def data(step: str = "download") -> None:
    """Runs Stage 0 data workflows for the provided step argument inside Modal and returns None."""
    raise NotImplementedError("Data task is not implemented.")


@app.function()
def narratives(split: str = "train") -> None:
    """Generates or validates narratives for the provided split argument inside Modal and returns None."""
    raise NotImplementedError("Narratives task is not implemented.")


@app.function()
def visualize(report: str = "splits") -> None:
    """Produces Stage 0 QA visualizations for the provided report argument inside Modal and returns None."""
    raise NotImplementedError("Visualize task is not implemented.")
