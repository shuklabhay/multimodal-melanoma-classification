import modal
from modal import App, Image, Volume

from src.pipelines.dataset_config import ISIC2019_CONFIG
from src.pipelines.prep_embeddings import create_narrative_embeddings
from src.pipelines.prep_images import create_manifests as _create_manifests
from src.pipelines.prep_images import download_dataset as _download_dataset
from src.pipelines.prep_images import verify_download as _verify_download
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
    gpu="A100-40GB",
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
    seed: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    train_bundle: str | None = None,
    val_bundle: str | None = None,
    test_bundle: str | None = None,
    class_weights_path: str | None = None,
):
    """Train the lesion classification model on Modal with WandB tracking.

    Args:
        run_name: Name for the WandB run.
        config_name: Name of the config file (without .yaml extension).
        resume_from: Optional run name to resume training from. If provided,
            loads the checkpoint from that run and continues training.
        model_type: Optional per-run override for model type: "image", "fusion", or "text".
        seed: Optional random seed override.
        batch_size: Optional batch size override.
        learning_rate: Optional optimizer learning-rate override.
        num_epochs: Optional max epochs override.
        early_stopping_patience: Optional early-stopping patience override.
        train_bundle: Optional train bundle path override.
        val_bundle: Optional validation bundle path override.
        test_bundle: Optional test bundle path override.
        class_weights_path: Optional class-weight path override.
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

    train_kwargs = {
        "run_name": run_name,
        "config": config,
        "config_name": config_name,
        "resume_from": resume_from,
    }
    if seed is not None:
        train_kwargs["seed"] = seed
    if batch_size is not None:
        train_kwargs["batch_size"] = batch_size
    if learning_rate is not None:
        train_kwargs["learning_rate"] = learning_rate
    if num_epochs is not None:
        train_kwargs["num_epochs"] = num_epochs
    if early_stopping_patience is not None:
        train_kwargs["early_stopping_patience"] = early_stopping_patience
    if train_bundle is not None:
        train_kwargs["train_bundle"] = train_bundle
    if val_bundle is not None:
        train_kwargs["val_bundle"] = val_bundle
    if test_bundle is not None:
        train_kwargs["test_bundle"] = test_bundle
    if class_weights_path is not None:
        train_kwargs["class_weights_path"] = class_weights_path

    train(**train_kwargs)


@app.function(
    gpu="A100-40GB",
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
def evaluate_checkpoint(
    run_name: str,
    config_name: str,
    model_type: str = "fusion",
    perturbation: str = "none",
    seed: int = 0,
    missing_rate: float = 0.5,
    bundle_path: str = "/data/bundled/test.npz",
    eval_run_name: str | None = None,
    eval_group: str = "ablations/eval",
    log_to_wandb: bool = True,
) -> dict:
    """Evaluate a saved checkpoint on a bundle with optional text perturbation.

    Args:
        run_name: Checkpoint run name under /data/checkpoints.
        config_name: Config file stem under /root/config.
        model_type: image|fusion|text override.
        perturbation: none|shuffle|zero|constant|missing|adversarial_swap.
        seed: RNG seed for perturbation operations.
        missing_rate: Only used when perturbation=missing.
        bundle_path: NPZ bundle path to evaluate.
        eval_run_name: Optional explicit WandB run name for this evaluation.
        eval_group: WandB group name for ablation/evaluation runs.
        log_to_wandb: Whether to log evaluation metrics to WandB.

    Returns:
        Result dict with metrics and perturbation stats.
    """
    import os

    import jax
    import numpy as np
    import wandb

    from src.paper.model_analysis import evaluate_checkpoint_bundle
    from src.pipelines.dataset_config import ISIC19_LABEL_CONFIG

    safe_tag = perturbation.replace("/", "_")
    json_path = f"/data/buffer/eval_{run_name}_{model_type}_{safe_tag}_seed{seed}.json"
    npz_path = f"/data/buffer/eval_{run_name}_{model_type}_{safe_tag}_seed{seed}.npz"

    result = evaluate_checkpoint_bundle(
        run_name=run_name,
        config_name=config_name,
        model_type=model_type,
        bundle_path=bundle_path,
        perturbation=perturbation,
        seed=seed,
        missing_rate=missing_rate,
        output_json=json_path,
        output_npz=npz_path,
    )

    # Record concrete runtime accelerator details for auditability.
    runtime_devices = sorted(
        {getattr(device, "device_kind", str(device)) for device in jax.devices()}
    )
    runtime_platforms = sorted({device.platform for device in jax.devices()})
    result["runtime_devices"] = runtime_devices
    result["runtime_platforms"] = runtime_platforms

    if not log_to_wandb:
        return result

    metrics = result.get("metrics", {})
    wb_name = (
        eval_run_name
        if eval_run_name is not None
        else f"abl_eval_{run_name}_{model_type}_{safe_tag}_seed{seed}"
    )

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=wb_name,
        group=eval_group,
        job_type="ablation_eval",
        config={
            "config_name": config_name,
            "source_run_name": run_name,
            "type": model_type,
            "perturbation": perturbation,
            "seed": seed,
            "missing_rate": missing_rate,
            "bundle_path": bundle_path,
            "runtime_devices": runtime_devices,
            "runtime_platforms": runtime_platforms,
        },
    )

    log_dict = {
        "test/loss": metrics.get("loss"),
        "test/accuracy": metrics.get("accuracy"),
        "test/balanced_accuracy": metrics.get("balanced_accuracy"),
        "test/sensitivity": metrics.get("sensitivity_mean"),
        "test/specificity": metrics.get("specificity_mean"),
        "test/auroc": metrics.get("auroc"),
        "test/ece": metrics.get("ece"),
        "test/brier": metrics.get("brier"),
    }
    for k, v in metrics.items():
        if k.startswith("bal_acc_"):
            log_dict[f"test_cw/{k}"] = v
        elif k.startswith("auroc_class_"):
            log_dict[f"test_cw/{k}"] = v

    wandb.log(log_dict)

    try:
        bundle = np.load(npz_path)
        preds = bundle["predictions"]
        labels = bundle["labels"]
        class_names = [
            ISIC19_LABEL_CONFIG[i]["name"] for i in range(len(ISIC19_LABEL_CONFIG))
        ]
        confusion = wandb.plot.confusion_matrix(
            preds=preds.tolist(),
            y_true=labels.tolist(),
            class_names=class_names,
            title="Test Confusion Matrix",
        )
        wandb.log({"test_confusion_matrix": confusion})
    except Exception as exc:
        print(f"Warning: failed to log confusion matrix for {wb_name}: {exc}")

    wandb.summary["test_loss"] = metrics.get("loss")
    wandb.summary["test_accuracy"] = metrics.get("accuracy")
    wandb.summary["test_balanced_accuracy"] = metrics.get("balanced_accuracy")
    wandb.summary["test_sensitivity"] = metrics.get("sensitivity_mean")
    wandb.summary["test_specificity"] = metrics.get("specificity_mean")
    wandb.summary["test_auroc"] = metrics.get("auroc")
    wandb.summary["test_ece"] = metrics.get("ece")
    wandb.summary["test_brier"] = metrics.get("brier")
    wandb.summary["perturbation"] = perturbation
    wandb.summary["source_run_name"] = run_name
    wandb.summary["missing_rate"] = missing_rate
    wandb.summary["runtime_devices"] = runtime_devices
    wandb.summary["runtime_platforms"] = runtime_platforms

    wandb.finish()
    return result


@app.function(
    gpu="A100-40GB",
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
def generate_explainability_cases(
    run_name: str,
    config_name: str,
    model_type: str = "fusion",
    output_dir: str = "/data/buffer",
    figure_name: str = "figure_explainability_cases.png",
    json_name: str = "explainability_cases.json",
    steps: int = 24,
) -> dict:
    """Generate qualitative explainability case studies and save to volume."""
    from src.paper.model_analysis import generate_explainability_case_studies

    return generate_explainability_case_studies(
        run_name=run_name,
        config_name=config_name,
        model_type=model_type,
        output_dir=output_dir,
        figure_name=figure_name,
        json_name=json_name,
        steps=steps,
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


@app.function(
    gpu="A100-40GB",
    image=embedding_image,
    cpu=4,
    memory=32768,
    timeout=86400,
)
def prepare_targeted_ablation_bundles(
    output_dir: str = "/data/bundled_ablation",
    base_bundle_dir: str = "/data/bundled",
    manifests_dir: str = "/root/assets/manifests",
    seed_csv: str = "0,1,2",
    batch_size: int = 384,
) -> dict:
    """Create targeted ablation bundles on Modal volume.

    Produces:
    - metadata-only train/val/test bundles
    - template-full train/val/test bundles (style-control with full narrative fields)
    - seed-specific train-shuffle bundles
    - field-only (symptoms/temporal/risk) test bundles
    """
    from src.paper.ablation_data import prepare_targeted_ablation_bundles as _prepare

    seeds = tuple(int(s.strip()) for s in seed_csv.split(",") if s.strip())
    return _prepare(
        base_bundle_dir=base_bundle_dir,
        manifests_dir=manifests_dir,
        output_dir=output_dir,
        seeds=seeds,
        batch_size=batch_size,
    )
