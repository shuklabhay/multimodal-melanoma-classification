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
from src.utils.data_utils import create_tf_dataset_from_bundle, get_dataloaders
from src.utils.eval_utils import (
    compute_metrics,
    load_best_from_wandb,
    log_final_metrics,
    upload_best_to_wandb,
)

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
    class_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single training step.

    Args:
        model: The model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        images: Batch of input images.
        labels: Batch of integer labels.
        class_weights: Per-class weights for loss weighting.

    Returns:
        A tuple containing the loss and logits for the batch.
    """

    def loss_fn(model: nnx.Module) -> tuple[jax.Array, jax.Array]:
        logits = model(images, train=True)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        )
        sample_weights = class_weights[labels]
        loss = jnp.mean(per_sample_loss * sample_weights)
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
    class_weights: jax.Array,
) -> dict[str, float]:
    """Trains the model for one epoch.

    Args:
        model: The model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        train_ds: The training dataset.
        epoch: The current epoch number (for progress bar display).
        class_weights: Per-class weights for loss weighting.

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

        loss, logits = train_step(
            model, optimizer, images_jax, labels_jax, class_weights
        )
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
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], jnp.ndarray, jnp.ndarray]:
    """Evaluates the model on a dataset.

    Args:
        model: The model to evaluate.
        ds: The dataset to evaluate on.
        return_predictions: If True, also return logits and labels.

    Returns:
        A dictionary containing evaluation metrics, or tuple of (metrics, logits, labels).
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
    metrics = compute_metrics(all_logits, all_labels, all_losses)

    if return_predictions:
        return metrics, all_logits, all_labels
    return metrics


def train(
    run_name: str,
    config: ModelConfig,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_classes: int = 8,
    train_bundle: str = "/data/bundled/train.npz",
    val_bundle: str = "/data/bundled/val.npz",
    test_bundle: str = "/data/bundled/test.npz",
    class_weights_path: str = "/root/assets/class_weights.npy",
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
        train_bundle: Path to the training data bundle NPZ.
        val_bundle: Path to the validation data bundle NPZ.
        test_bundle: Path to the test data bundle NPZ.
        class_weights_path: Path to class weights .npy file.
        seed: Random seed for reproducibility.
        early_stopping_patience: Number of epochs without improvement before stopping.

    Returns:
        The trained model.
    """
    print("Loading class weights...")
    class_weights = jnp.array(np.load(class_weights_path))
    print(f"Class weights: {class_weights}")

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
            "class_weights": class_weights.tolist(),
        },
    )

    print("Loading datasets...")
    train_ds, val_ds = get_dataloaders(
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        batch_size=batch_size,
    )

    print("Creating model and optimizer...")
    model, optimizer = create_train_state(
        config=config,
        num_classes=num_classes,
        learning_rate=learning_rate,
        seed=seed,
    )

    best_val_balanced_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, optimizer, train_ds, epoch, class_weights)
        val_metrics = cast(dict[str, float], eval_epoch(model, val_ds))

        wandb.log(
            {
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                "train/sensitivity": train_metrics["sensitivity_mean"],
                "train/specificity": train_metrics["specificity_mean"],
                "train/auroc": train_metrics["auroc"],
                "train/ece": train_metrics["ece"],
                "train/brier": train_metrics["brier"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                "val/sensitivity": val_metrics["sensitivity_mean"],
                "val/specificity": val_metrics["specificity_mean"],
                "val/auroc": val_metrics["auroc"],
                "val/ece": val_metrics["ece"],
                "val/brier": val_metrics["brier"],
            }
        )

        if val_metrics["balanced_accuracy"] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            print(
                f"Epoch {epoch}: New best! bal_acc={best_val_balanced_acc:.4f} "
                f"(prev best epoch {best_epoch})"
            )
            upload_best_to_wandb(
                model,
                epoch,
                best_val_balanced_acc,
                learning_rate,
                batch_size,
                num_classes,
                seed,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"Epoch {epoch}: No improvement ({epochs_without_improvement}/{early_stopping_patience})"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Unloading train/val datasets...")
    del train_ds, val_ds

    print("Loading best model checkpoint...")
    load_best_from_wandb(model)

    print("Loading test dataset...")
    test_ds = create_tf_dataset_from_bundle(test_bundle, batch_size=batch_size)

    print("Running test evaluation...")
    test_result = eval_epoch(model, test_ds, return_predictions=True)
    test_metrics, test_logits, test_labels = cast(
        tuple[dict[str, float], jnp.ndarray, jnp.ndarray], test_result
    )

    log_final_metrics(
        best_val_balanced_acc, best_epoch, test_metrics, test_logits, test_labels
    )

    print(
        f"Training complete. Best val balanced accuracy: {best_val_balanced_acc:.4f} (epoch {best_epoch})"
    )
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")

    return model
