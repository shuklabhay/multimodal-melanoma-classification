import os
from typing import cast

import jax
import jax.numpy as jnp
import wandb
from flax import nnx, traverse_util
from safetensors.flax import load_file, save_file

from src.pipelines.label_config import LABEL_CONFIG

CHECKPOINT_BASE_DIR = "/data/checkpoints"


def calculate_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Calculate accuracy from logits and labels.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).

    Returns:
        Scalar accuracy value.
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def calculate_balanced_accuracy(
    logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> tuple[float, dict[str, float]]:
    """Calculate balanced accuracy (mean of per-class recalls).

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        num_classes: Number of classes.

    Returns:
        Tuple of (balanced_accuracy, per_class_accuracy_dict).
    """
    predictions = jnp.argmax(logits, axis=-1)
    per_class_acc = {}
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        short = LABEL_CONFIG[c]["short"]
        key = f"{c}{short}"
        if jnp.sum(mask) > 0:
            recall = jnp.sum((predictions == c) & mask) / jnp.sum(mask)
            recalls.append(float(recall))
            per_class_acc[key] = float(recall)
        else:
            per_class_acc[key] = 0.0
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0
    return balanced_acc, per_class_acc


def calculate_sensitivity_specificity(
    logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate per-class sensitivity (recall) and specificity.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        num_classes: Number of classes.

    Returns:
        Tuple of (sensitivity_dict, specificity_dict) with per-class values.
    """
    predictions = jnp.argmax(logits, axis=-1)
    sensitivity = {}
    specificity = {}

    for c in range(num_classes):
        tp = jnp.sum((predictions == c) & (labels == c))
        fn = jnp.sum((predictions != c) & (labels == c))
        tn = jnp.sum((predictions != c) & (labels != c))
        fp = jnp.sum((predictions == c) & (labels != c))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sensitivity[f"class_{c}"] = sens
        specificity[f"class_{c}"] = spec

    return sensitivity, specificity


def calculate_auroc(
    logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> tuple[float, dict[str, float]]:
    """Calculate AUROC (overall macro and per-class).

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        num_classes: Number of classes.

    Returns:
        Tuple of (macro_auroc, per_class_auroc_dict).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    per_class_auroc = {}

    for c in range(num_classes):
        class_probs = probs[:, c]
        class_labels = (labels == c).astype(jnp.float32)

        if jnp.sum(class_labels) == 0 or jnp.sum(1 - class_labels) == 0:
            per_class_auroc[f"class_{c}"] = 0.0
            continue

        sorted_indices = jnp.argsort(class_probs)[::-1]
        sorted_labels = class_labels[sorted_indices]

        tps = jnp.cumsum(sorted_labels)
        fps = jnp.cumsum(1 - sorted_labels)

        tpr = tps / jnp.sum(class_labels)
        fpr = fps / jnp.sum(1 - class_labels)

        tpr = jnp.concatenate([jnp.array([0.0]), tpr])
        fpr = jnp.concatenate([jnp.array([0.0]), fpr])

        auroc = jnp.sum((fpr[1:] - fpr[:-1]) * tpr[1:])
        per_class_auroc[f"class_{c}"] = float(auroc)

    macro_auroc = sum(per_class_auroc.values()) / len(per_class_auroc)
    return macro_auroc, per_class_auroc


def calculate_brier_score(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Calculate Brier score (mean squared error of probabilities).

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).

    Returns:
        Brier score (lower is better).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    num_classes = probs.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    brier = jnp.mean(jnp.sum((probs - one_hot) ** 2, axis=-1))
    return float(brier)


def calculate_ece(
    logits: jnp.ndarray, labels: jnp.ndarray, num_bins: int = 15
) -> float:
    """Calculate Expected Calibration Error.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        num_bins: Number of bins for calibration.

    Returns:
        ECE value (lower is better).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(jnp.float32)

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        bin_size = jnp.sum(in_bin)
        if bin_size > 0:
            bin_accuracy = jnp.sum(accuracies * in_bin) / bin_size
            bin_confidence = jnp.sum(confidences * in_bin) / bin_size
            ece += (bin_size / total_samples) * jnp.abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    losses: list[float],
    num_classes: int = 8,
) -> dict[str, float]:
    """Compute evaluation metrics from logits, labels, and losses.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        losses: List of per-batch loss values.
        num_classes: Number of classes.

    Returns:
        Dictionary containing computed metrics.
    """
    accuracy = float(calculate_accuracy(logits, labels))
    balanced_accuracy, per_class_acc = calculate_balanced_accuracy(
        logits, labels, num_classes
    )
    loss = sum(losses) / len(losses)

    sensitivity, specificity = calculate_sensitivity_specificity(
        logits, labels, num_classes
    )
    macro_auroc, per_class_auroc = calculate_auroc(logits, labels, num_classes)
    ece = calculate_ece(logits, labels)
    brier = calculate_brier_score(logits, labels)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "loss": loss,
        "sensitivity_mean": sum(sensitivity.values()) / len(sensitivity),
        "specificity_mean": sum(specificity.values()) / len(specificity),
        "auroc": macro_auroc,
        "ece": ece,
        "brier": brier,
    }

    for k, v in per_class_auroc.items():
        metrics[f"auroc_{k}"] = v

    for k, v in per_class_acc.items():
        metrics[f"bal_acc_{k}"] = v

    return metrics


def get_checkpoint_dir(run_name: str) -> str:
    """Get the checkpoint directory path for a run.

    Args:
        run_name: Name of the training run.

    Returns:
        Path to the checkpoint directory.
    """
    return os.path.join(CHECKPOINT_BASE_DIR, run_name)


def get_checkpoint_paths(run_name: str) -> tuple[str, str]:
    """Get paths for model and metadata files for a run.

    Args:
        run_name: Name of the training run.

    Returns:
        Tuple of (model_path, metadata_path).
    """
    checkpoint_dir = get_checkpoint_dir(run_name)
    model_path = os.path.join(checkpoint_dir, "best_model.safetensors")
    metadata_path = os.path.join(checkpoint_dir, "metadata.txt")
    return model_path, metadata_path


def save_checkpoint(
    model: nnx.Module,
    run_name: str,
    epoch: int,
    val_accuracy: float,
    learning_rate: float,
    batch_size: int,
    num_classes: int,
    seed: int,
    config_name: str,
    wandb_run_id: str,
) -> str:
    """Save model checkpoint to the Modal volume.

    Args:
        model: The model to save.
        run_name: Name of the training run.
        epoch: The current training epoch.
        val_accuracy: The validation accuracy achieved.
        learning_rate: The learning rate used for training.
        batch_size: The batch size used for training.
        num_classes: The number of output classes for the model.
        seed: The random seed used for model initialization.
        config_name: Name of the config file used.
        wandb_run_id: The WandB run ID for resuming logging.

    Returns:
        Path to the checkpoint directory.
    """
    checkpoint_dir = get_checkpoint_dir(run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path, metadata_path = get_checkpoint_paths(run_name)

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    _, params, _ = nnx.split(model, nnx.Param, ...)
    flat_params = traverse_util.flatten_dict(nnx.to_pure_dict(params), sep=".")
    flat_params = cast(dict[str, jax.Array], flat_params)
    save_file(flat_params, model_path)

    with open(metadata_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"epoch={epoch}\n")
        f.write(f"val_accuracy={val_accuracy}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_classes={num_classes}\n")
        f.write(f"seed={seed}\n")
        f.write(f"config_name={config_name}\n")
        f.write(f"checkpoint_path={checkpoint_dir}\n")
        f.write(f"wandb_run_id={wandb_run_id}\n")

    print(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def upload_best_to_wandb(run_name: str) -> None:
    """Upload the final best model from checkpoint directory to WandB artifact.

    Args:
        run_name: Name of the training run.
    """
    model_path, metadata_path = get_checkpoint_paths(run_name)
    checkpoint_dir = get_checkpoint_dir(run_name)

    if not os.path.exists(model_path):
        print(f"Warning: No checkpoint found at {model_path}")
        return

    metadata: dict[str, str | int | float] = {
        "run_name": run_name,
        "checkpoint_volume_path": checkpoint_dir,
    }
    with open(metadata_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            if key in ("epoch", "batch_size", "num_classes", "seed"):
                metadata[key] = int(value)
            elif key in ("val_accuracy", "learning_rate"):
                metadata[key] = float(value)
            else:
                metadata[key] = value

    artifact = wandb.Artifact(
        name=f"{run_name}_final_model",
        type="model",
        metadata=metadata,
    )
    artifact.add_file(model_path, name=f"{run_name}_best.safetensors")
    wandb.log_artifact(artifact)
    print(f"Uploaded final best model artifact: {run_name}_final_model")
    print(f"Checkpoint volume path: {checkpoint_dir}")


def load_checkpoint(model: nnx.Module, run_name: str) -> None:
    """Load model parameters from checkpoint directory.

    Args:
        model: The model to load parameters into (modified in place).
        run_name: Name of the training run to load checkpoint from.
    """
    model_path, _ = get_checkpoint_paths(run_name)
    print(f"Loading checkpoint from {model_path}")
    flat_params = load_file(model_path)

    _, params, _ = nnx.split(model, nnx.Param, ...)
    params_dict = nnx.to_pure_dict(params)
    flat_current = cast(
        dict[str, jax.Array], traverse_util.flatten_dict(params_dict, sep=".")
    )

    for key in list(flat_current.keys()):
        if key in flat_params:
            flat_current[key] = flat_params[key]

    nested_updated = traverse_util.unflatten_dict(flat_current, sep=".")

    def update_recursive(state: nnx.State, updates: dict) -> None:
        for k, v in updates.items():
            if isinstance(v, dict):
                update_recursive(getattr(state, k), v)
            else:
                setattr(state, k, nnx.Param(v))

    update_recursive(params, nested_updated)
    nnx.update(model, params)


def load_checkpoint_metadata(run_name: str) -> dict | None:
    """Load checkpoint metadata for a run.

    Args:
        run_name: Name of the training run.

    Returns:
        Dictionary with checkpoint metadata, or None if not found.
    """
    _, metadata_path = get_checkpoint_paths(run_name)

    if not os.path.exists(metadata_path):
        return None

    metadata = {}
    with open(metadata_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            if key in ("epoch", "batch_size", "num_classes", "seed"):
                metadata[key] = int(value)
            elif key in ("val_accuracy", "learning_rate"):
                metadata[key] = float(value)
            else:
                metadata[key] = value

    return metadata


def checkpoint_exists(run_name: str) -> bool:
    """Check if a checkpoint exists for a run.

    Args:
        run_name: Name of the training run.

    Returns:
        True if checkpoint exists, False otherwise.
    """
    model_path, metadata_path = get_checkpoint_paths(run_name)
    return os.path.exists(model_path) and os.path.exists(metadata_path)


def list_checkpoints() -> list[dict]:
    """List all available checkpoints with their metadata.

    Returns:
        List of checkpoint metadata dicts, sorted by run name.
    """
    if not os.path.exists(CHECKPOINT_BASE_DIR):
        return []

    checkpoints = []
    for run_dir in sorted(os.listdir(CHECKPOINT_BASE_DIR)):
        run_name = run_dir
        if checkpoint_exists(run_name):
            metadata = load_checkpoint_metadata(run_name)
            if metadata:
                checkpoints.append(metadata)

    return checkpoints


def log_confusion_matrix(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    title: str = "Confusion Matrix",
) -> None:
    """Log confusion matrix to WandB.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        title: Title for the confusion matrix plot.
    """

    class_names = [LABEL_CONFIG[i]["name"] for i in range(len(LABEL_CONFIG))]
    predictions = jnp.argmax(logits, axis=-1)
    cm = wandb.plot.confusion_matrix(
        preds=predictions.tolist(),
        y_true=labels.tolist(),
        class_names=class_names,
        title=title,
    )
    wandb.log({title.lower().replace(" ", "_"): cm})


def log_final_metrics(
    best_val_balanced_acc: float,
    best_epoch: int,
    test_metrics: dict[str, float],
    test_logits: jnp.ndarray,
    test_labels: jnp.ndarray,
) -> None:
    """Log final training metrics to WandB and finish the run.

    Args:
        best_val_balanced_acc: The best validation balanced accuracy achieved.
        best_epoch: The epoch at which best validation balanced accuracy was achieved.
        test_metrics: Dictionary containing test set metrics.
        test_logits: Test set logits for confusion matrix.
        test_labels: Test set labels for confusion matrix.
    """
    test_log_dict = {
        "test/loss": test_metrics["loss"],
        "test/accuracy": test_metrics["accuracy"],
        "test/balanced_accuracy": test_metrics["balanced_accuracy"],
        "test/sensitivity": test_metrics["sensitivity_mean"],
        "test/specificity": test_metrics["specificity_mean"],
        "test/auroc": test_metrics["auroc"],
        "test/ece": test_metrics["ece"],
        "test/brier": test_metrics["brier"],
    }
    for k, v in test_metrics.items():
        if k.startswith("bal_acc_"):
            test_log_dict[f"test_cw/{k}"] = v
    wandb.log(test_log_dict)

    log_confusion_matrix(test_logits, test_labels, "Test Confusion Matrix")

    wandb.summary["best_val_balanced_accuracy"] = best_val_balanced_acc
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["test_loss"] = test_metrics["loss"]
    wandb.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.summary["test_balanced_accuracy"] = test_metrics["balanced_accuracy"]
    wandb.summary["test_sensitivity"] = test_metrics["sensitivity_mean"]
    wandb.summary["test_specificity"] = test_metrics["specificity_mean"]
    wandb.summary["test_auroc"] = test_metrics["auroc"]
    wandb.summary["test_ece"] = test_metrics["ece"]
    wandb.summary["test_brier"] = test_metrics["brier"]
    wandb.finish()
