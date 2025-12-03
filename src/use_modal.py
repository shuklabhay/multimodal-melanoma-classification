import modal
from modal import App, Image, Volume

from src.pipelines.prep_embeddings import create_narrative_embeddings
from src.pipelines.prep_images import (
    create_manifests,
    download_isic2019,
    verify_download,
)
from src.pipelines.prep_narratives import generate_narrative_texts


def download_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.encode(["test"])


base_image = (
    Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_dir("assets", "/root/assets")
    .add_local_dir("config", "/root/config")
)
embedding_image = (
    Image.debian_slim(python_version="3.11")
    .uv_sync()
    .run_function(download_sentence_transformer)
    .add_local_dir("assets", "/root/assets")
    .add_local_dir("config", "/root/config")
)

app = App(
    "lesion-classification",
    image=base_image,
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
    """Generates narratives from pre-generated manifests using Nous API."""
    generate_narrative_texts()


@app.local_entrypoint()
def narratives_entrypoint() -> None:
    """Local entrypoint to generate narratives."""
    print("Calling Modal to generate narrative texts...")
    process_narratives.remote()
    print("✓ Narrative text generation complete on Modal")


@app.function(gpu="L4", image=embedding_image, timeout=86400)
def process_embeddings() -> None:
    """Generates narrative embeddings using sentence transformers on L4 GPU."""
    create_narrative_embeddings(batch_size=512)


@app.local_entrypoint()
def embeddings_entrypoint() -> None:
    """Local entrypoint to generate embeddings."""
    print("Calling Modal to generate narrative embeddings...")
    process_embeddings.remote()
    print("✓ Narrative embedding generation complete on Modal")


@app.function(timeout=86400)
def bundle_splits() -> None:
    """Bundle manifest data into NPZ files for each split."""
    from src.utils.data_utils import preprocess_and_bundle_data

    preprocess_and_bundle_data()


@app.function(timeout=60)
def list_available_checkpoints() -> list[dict]:
    """List all available checkpoints on the Modal volume.

    Returns:
        List of checkpoint metadata dicts.
    """
    from src.utils.eval_utils import list_checkpoints

    checkpoints = list_checkpoints()
    print(checkpoints)
    return checkpoints


@app.function(
    gpu="L4",
    timeout=86400,
    secrets=[
        modal.Secret.from_dict(
            {
                "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        ),
    ],
)
def generate_test_tsne() -> None:
    """Generate t-SNE visualization for test set embeddings."""
    from src.pipelines.prep_analysis import generate_test_tsne

    generate_test_tsne()


@app.function(
    gpu="L4",
    timeout=86400,
    secrets=[
        modal.Secret.from_dict(
            {
                "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        ),
    ],
)
def generate_similar_pairs() -> None:
    """Generate k-NN mixing analysis and similar pairs visualization."""
    from src.pipelines.prep_analysis import generate_similar_pairs

    generate_similar_pairs()


@app.function(
    gpu="L4",
    timeout=86400,
    secrets=[
        modal.Secret.from_dict(
            {
                "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        ),
    ],
)
def generate_class_separation_analysis() -> None:
    """Generate unified class separation analysis with metrics and distance plots."""
    from src.pipelines.prep_analysis import generate_class_separation_analysis

    generate_class_separation_analysis()


@app.function(
    gpu="L4",
    cpu=2,
    memory=24576,
    timeout=86400,
    retries=0,
    secrets=[
        modal.Secret.from_dict(
            {
                "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        ),
        modal.Secret.from_dotenv(),
    ],
)
def train_model(run_name: str, config_name: str, resume_from: str | None = None):
    """Train the lesion classification model on Modal with WandB tracking.

    Args:
        run_name: Name for the WandB run.
        config_name: Name of the config file (without .yaml extension).
        resume_from: Optional run name to resume training from. If provided,
            loads the checkpoint from that run and continues training.
    """
    from src.models.model_builder import load_config
    from src.utils.train_utils import train

    config_path = f"/root/config/{config_name}.yaml"
    config = load_config(config_path)

    train(
        run_name=run_name,
        config=config,
        config_name=config_name,
        resume_from=resume_from,
    )
