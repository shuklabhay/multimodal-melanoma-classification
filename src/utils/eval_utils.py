from typing import cast

import jax
import jax.numpy as jnp
import wandb
from flax import nnx, traverse_util
from safetensors.flax import save_file


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


def compute_metrics(
    logits: jnp.ndarray, labels: jnp.ndarray, losses: list[float]
) -> dict[str, float]:
    """Compute evaluation metrics from logits, labels, and losses.

    Args:
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Integer labels of shape (batch_size,).
        losses: List of per-batch loss values.

    Returns:
        Dictionary containing computed metrics.
    """
    accuracy = calculate_accuracy(logits, labels)
    loss = sum(losses) / len(losses)
    return {"accuracy": float(accuracy), "loss": loss}


def upload_best_model(
    model: nnx.Module,
    epoch: int,
    val_accuracy: float,
    learning_rate: float,
    batch_size: int,
    num_classes: int,
    seed: int,
) -> None:
    """Save and upload best model as WandB artifact.

    Args:
        model: The model to save.
        epoch: The current training epoch.
        val_accuracy: The validation accuracy achieved.
        learning_rate: The learning rate used for training.
        batch_size: The batch size used for training.
        num_classes: The number of output classes for the model.
        seed: The random seed used for model initialization.
    """
    local_path = "best_model.safetensors"
    _, params, _ = nnx.split(model, nnx.Param, ...)

    flat_params = traverse_util.flatten_dict(nnx.to_pure_dict(params), sep=".")
    flat_params = cast(dict[str, jax.Array], flat_params)
    save_file(flat_params, local_path)

    artifact = wandb.Artifact(
        name="best-model",
        type="model",
        metadata={
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "seed": seed,
        },
    )
    artifact.add_file(local_path)
    wandb.log_artifact(artifact)


def log_final_metrics(
    best_val_accuracy: float,
    best_epoch: int,
    test_metrics: dict[str, float],
) -> None:
    """Log final training metrics to WandB and finish the run.

    Args:
        best_val_accuracy: The best validation accuracy achieved during training.
        best_epoch: The epoch at which best validation accuracy was achieved.
        test_metrics: Dictionary containing test set metrics (accuracy, loss).
    """
    wandb.log(
        {
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
        }
    )
    wandb.summary["best_val_accuracy"] = best_val_accuracy
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["test_loss"] = test_metrics["loss"]
    wandb.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.finish()
