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


@app.function(timeout=600)
def fetch_visualizer_batches(
    num_batches: int = 2,
    batch_size: int = 4,
    image_size: tuple[int, int] = (224, 224),
) -> bytes:
    """Fetch batches from dataloaders on Modal for visualization.

    Args:
        num_batches: Number of batches to fetch per split.
        batch_size: Batch size for dataloaders.
        image_size: Image size for dataloaders.

    Returns:
        Pickle-compressed bytes of batch data.
    """
    import pickle
    import zlib

    from src.utils.data_utils import get_dataloaders

    train_ds, val_ds, test_ds = get_dataloaders(
        train_manifest="/root/assets/manifests/split_train.jsonl",
        val_manifest="/root/assets/manifests/split_val.jsonl",
        test_manifest="/root/assets/manifests/split_test.jsonl",
        batch_size=batch_size,
        image_size=image_size,
    )

    result = {"train": [], "val": [], "test": []}
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    for split_name, dataset in datasets.items():
        print(f"Fetching {num_batches} batches from {split_name}...")
        for i, ((images, texts), labels) in enumerate(dataset.take(num_batches)):
            images_np = images.numpy()
            texts_list = [t.numpy().decode("utf-8") for t in texts]
            labels_np = labels.numpy()
            result[split_name].append((images_np, texts_list, labels_np))
            print(f"  Batch {i + 1}/{num_batches} fetched")

    print("Compressing data for transfer...")
    return zlib.compress(pickle.dumps(result))


@app.local_entrypoint()
def visualizer_entrypoint(num_batches: int = 2, batch_size: int = 4) -> None:
    """Local entrypoint to fetch batches and run visualizer.

    Args:
        num_batches: Number of batches to fetch per split.
        batch_size: Batch size for dataloaders.
    """
    import pickle
    import zlib

    from src.utils.data_utils import run_visualizer

    print(f"Fetching {num_batches} batches (batch_size={batch_size}) from Modal...")
    compressed = fetch_visualizer_batches.remote(
        num_batches=num_batches, batch_size=batch_size
    )

    print("Decompressing data...")
    batches = pickle.loads(zlib.decompress(compressed))

    print("✓ Batches fetched, launching visualizer...")
    run_visualizer(batches, batch_size=batch_size)
