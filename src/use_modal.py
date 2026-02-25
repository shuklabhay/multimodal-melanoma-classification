import modal
from modal import App, Image, Volume

from src.pipelines.prep_embeddings import create_narrative_embeddings
from src.pipelines.dataset_config import ISIC2019_CONFIG
from src.pipelines.prep_images import (
    create_manifests as _create_manifests,
    download_dataset as _download_dataset,
    verify_download as _verify_download,
)
from src.pipelines.prep_narratives import generate_narrative_texts


def download_sentence_transformer():
    """Downloads and caches the sentence transformer model during image build."""
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


DATASET_CONFIGS = {
    "isic2019": ISIC2019_CONFIG,
}


@app.function(timeout=86400)
def download_images(dataset: str) -> None:
    """Downloads images and metadata for specified dataset from ISIC archive.

    Args:
        dataset: Dataset name key ('isic2019').
    """
    config = DATASET_CONFIGS[dataset]
    _download_dataset(config)


@app.function(timeout=86400)
def verify_download(dataset: str) -> None:
    """Verifies downloaded dataset by checking image counts and label mappings.

    Args:
        dataset: Dataset name key ('isic2019').
    """
    config = DATASET_CONFIGS[dataset]
    _verify_download(config)


@app.function(cpu=4, memory=4096, timeout=86400)
def create_manifests(dataset: str) -> None:
    """Creates train/val/test manifest files for specified dataset.

    Args:
        dataset: Dataset name key ('isic2019').
    """
    config = DATASET_CONFIGS[dataset]
    _create_manifests(config, train_ratio=0.9)


@app.function(secrets=[modal.Secret.from_dotenv()], timeout=86400)
def generate_narratives(dataset: str, split: str | None = None) -> None:
    """Generates narrative texts for specified dataset using LLM.

    Args:
        dataset: Dataset name key ('isic2019').
        split: Optional split name ("train", "val", or "test"). If None, runs all.
    """
    generate_narrative_texts(dataset, split=split)


@app.local_entrypoint()
def narratives_entrypoint(dataset: str, split: str | None = None) -> None:
    """Local entrypoint to trigger narrative generation on Modal.

    Args:
        dataset: Dataset name key ('isic2019').
        split: Optional split name ("train", "val", or "test"). If None, runs all.
    """
    split_label = split or "all"
    print(f"Calling Modal to generate narrative texts for {dataset} ({split_label})...")
    generate_narratives.remote(dataset, split)
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
    gpu="A100",
    cpu=2,
    memory=24576,
    timeout=86400,
    retries=0,
    secrets=[
        modal.Secret.from_dict(
            {
                "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "TF_CPP_MIN_LOG_LEVEL": "3",
            }
        ),
        modal.Secret.from_dotenv(),
    ],
)
def train_model(
    run_name: str,
    config_name: str,
    resume_from: str | None = None,
    model_type: str | None = None,
):
    """Train the lesion classification model on Modal with WandB tracking.

    Args:
        run_name: Name for the WandB run.
        config_name: Name of the config file (without .yaml extension).
        resume_from: Optional run name to resume training from. If provided,
            loads the checkpoint from that run and continues training.
        model_type: Optional per-run override for model type: "image", "fusion", or "text".
    """
    from src.models.model_builder import load_config
    from src.utils.train_utils import train

    config_path = f"/root/config/{config_name}.yaml"
    config = load_config(config_path)
    if model_type is not None:
        if model_type not in {"image", "fusion", "text"}:
            raise ValueError(
                f"Invalid model_type={model_type}. Use one of: image, fusion, text"
            )
        config["type"] = model_type

    train(
        run_name=run_name,
        config=config,
        config_name=config_name,
        resume_from=resume_from,
    )


@app.function(
    timeout=86400,
    memory=1536,
    secrets=[modal.Secret.from_dotenv()],
)
def evaluate_gpt5_on_test(
    modality: str,
    run_name: str,
) -> dict:
    """Run GPT-5 evaluation on test set from Modal volume.

    Args:
        modality: "image", "text", or "fusion".
        run_name: Name for the WandB run.

    Returns:
        Dictionary with metrics and results.
    """
    import asyncio
    import os

    import jax.numpy as jnp
    import wandb

    from src.utils.data_utils import load_manifest
    from src.utils.eval_utils import (
        calculate_classification_metrics,
        log_confusion_matrix,
    )
    from src.utils.llm_utils import process_classification_entries

    print(f"Starting GPT-5 evaluation for modality: {modality}")
    print(f"Run name: {run_name}")

    entries = load_manifest("/root/assets/manifests/split_test.jsonl")
    print(f"Loaded {len(entries)} test entries")

    predictions, labels = asyncio.run(
        process_classification_entries(entries, modality, batch_size=512)
    )

    predictions_arr = jnp.array(predictions)
    labels_arr = jnp.array(labels)

    metrics = calculate_classification_metrics(
        predictions_arr, labels_arr, num_classes=8
    )

    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=run_name,
        config={
            "model": "gpt-5-chat-latest",
            "modality": modality,
            "test_samples": len(entries),
        },
    )

    wandb.log(
        {
            "test/accuracy": metrics["accuracy"],
            "test/balanced_accuracy": metrics["balanced_accuracy"],
            "test/sensitivity": sum(sensitivity.values()) / len(sensitivity),
            "test/specificity": sum(specificity.values()) / len(specificity),
        }
    )

    for k, v in metrics["per_class_acc"].items():
        wandb.log({f"test_cw/bal_acc_{k}": v})

    log_confusion_matrix(predictions_arr, labels_arr, "Test Confusion Matrix")

    wandb.summary["test_accuracy"] = metrics["accuracy"]
    wandb.summary["test_balanced_accuracy"] = metrics["balanced_accuracy"]
    wandb.summary["test_sensitivity"] = sum(sensitivity.values()) / len(sensitivity)
    wandb.summary["test_specificity"] = sum(specificity.values()) / len(specificity)

    wandb.finish()

    print(f"\nResults for {modality}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "predictions": predictions,
        "labels": labels,
    }
