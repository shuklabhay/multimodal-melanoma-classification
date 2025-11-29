import os
from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb
from dotenv import load_dotenv
from flax import nnx
from tqdm import tqdm

load_dotenv()

from src.models.model_builder import DynamicModel, ModelConfig, build_model
from src.utils.data_utils import get_dataloaders
from src.utils.eval_utils import compute_metrics, log_final_metrics, upload_best_model

DatasetBatch = tuple[tuple[np.ndarray, np.ndarray], np.ndarray]


def create_train_state(
    config: ModelConfig,
    num_classes: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 0,
) -> tuple[DynamicModel, nnx.Optimizer]:
    """Initializes a model from config and an Adam optimizer.

    Args:
        config: Configuration dictionary with encoder and model settings.
        num_classes: The number of output classes for the classifier.
        learning_rate: The learning rate for the Adam optimizer.
        seed: Random seed for model initialization.

    Returns:
        A tuple containing the initialized model and the nnx.Optimizer.
    """
    model = build_model(config=config, num_classes=num_classes, seed=seed)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    return model, optimizer


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    images: jax.Array,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single training step.

    Args:
        model: The model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        images: Batch of input images.
        labels: Batch of integer labels.

    Returns:
        A tuple containing the loss and logits for the batch.
    """

    def loss_fn(model: nnx.Module) -> tuple[jax.Array, jax.Array]:
        logits = model(images, train=True)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, logits


@nnx.jit
def eval_step(
    model: nnx.Module,
    images: jax.Array,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single evaluation step.

    Args:
        model: The model to evaluate.
        images: Batch of input images.
        labels: Batch of integer labels.

    Returns:
        A tuple containing the loss and logits for the batch.
    """
    logits = model(images, train=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    return loss, logits


def train_epoch(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_ds: tf.data.Dataset,
    epoch: int,
) -> dict[str, float]:
    """Trains the model for one epoch.

    Args:
        model: The model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        train_ds: The training dataset.
        epoch: The current epoch number (for progress bar display).

    Returns:
        A dictionary containing training metrics (accuracy, loss).
    """
    all_logits = []
    all_labels = []
    all_losses = []

    batches = cast(Iterable[DatasetBatch], train_ds.as_numpy_iterator())
    pbar = tqdm(
        batches,
        desc=f"Epoch {epoch}",
        leave=False,
        total=int(train_ds.cardinality().numpy()),
    )
    for batch in pbar:
        (images, _texts), labels = batch
        images_jax = jnp.array(images)
        labels_jax = jnp.array(labels)

        loss, logits = train_step(model, optimizer, images_jax, labels_jax)
        all_logits.append(logits)
        all_labels.append(labels_jax)
        all_losses.append(float(loss))

        pbar.set_postfix(loss=f"{loss:.4f}")

    all_logits = jnp.concatenate(all_logits)
    all_labels = jnp.concatenate(all_labels)
    return compute_metrics(all_logits, all_labels, all_losses)


def eval_epoch(
    model: nnx.Module,
    ds: tf.data.Dataset,
) -> dict[str, float]:
    """Evaluates the model on a dataset.

    Args:
        model: The model to evaluate.
        ds: The dataset to evaluate on.

    Returns:
        A dictionary containing evaluation metrics (accuracy, loss).
    """
    all_logits = []
    all_labels = []
    all_losses = []

    batches = cast(Iterable[DatasetBatch], ds.as_numpy_iterator())
    for (images, _texts), labels in batches:
        images_jax = jnp.array(images)
        labels_jax = jnp.array(labels)

        loss, logits = eval_step(model, images_jax, labels_jax)
        all_logits.append(logits)
        all_labels.append(labels_jax)
        all_losses.append(float(loss))

    all_logits = jnp.concatenate(all_logits)
    all_labels = jnp.concatenate(all_labels)
    return compute_metrics(all_logits, all_labels, all_losses)


def train(
    run_name: str,
    config: ModelConfig,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_classes: int = 8,
    train_manifest: str = "/root/assets/manifests/split_train.jsonl",
    val_manifest: str = "/root/assets/manifests/split_val.jsonl",
    test_manifest: str = "/root/assets/manifests/split_test.jsonl",
    seed: int = 0,
    early_stopping_patience: int = 10,
) -> DynamicModel:
    """Full training loop with WandB tracking and early stopping.

    Args:
        run_name: Name for the WandB run.
        config: Model configuration dictionary.
        num_epochs: Maximum number of epochs to train.
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for the Adam optimizer.
        num_classes: Number of output classes for the classifier.
        train_manifest: Path to the training data manifest file.
        val_manifest: Path to the validation data manifest file.
        test_manifest: Path to the test data manifest file.
        seed: Random seed for reproducibility.
        early_stopping_patience: Number of epochs without improvement before stopping.

    Returns:
        The trained model.
    """
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=run_name,
        config={
            "config_name": run_name.split("_", 2)[-1],
            **config,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
        },
    )

    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        batch_size=batch_size,
    )

    print("Creating model and optimizer...")
    model, optimizer = create_train_state(
        config=config,
        num_classes=num_classes,
        learning_rate=learning_rate,
        seed=seed,
    )

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, optimizer, train_ds, epoch)
        val_metrics = eval_epoch(model, val_ds)

        wandb.log(
            {
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "epoch": epoch,
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            upload_best_model(
                model,
                epoch,
                best_val_accuracy,
                learning_rate,
                batch_size,
                num_classes,
                seed,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Running test evaluation...")
    test_metrics = eval_epoch(model, test_ds)

    log_final_metrics(best_val_accuracy, best_epoch, test_metrics)

    print(
        f"Training complete. Best val accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})"
    )
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    return model
