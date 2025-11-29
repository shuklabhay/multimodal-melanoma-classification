from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from src.models.backbones import ResNet50Classifier
from src.utils.data_utils import get_dataloaders

DatasetBatch = tuple[tuple[np.ndarray, np.ndarray], np.ndarray]


def create_train_state(
    num_classes: int = 8, learning_rate: float = 1e-4, seed: int = 0
) -> tuple[ResNet50Classifier, nnx.Optimizer]:
    """Initializes a ResNet50Classifier model and an Adam optimizer.

    Args:
        num_classes: The number of output classes for the classifier.
        learning_rate: The learning rate for the Adam optimizer.
        seed: Random seed for model initialization.

    Returns:
        A tuple containing the initialized ResNet50Classifier model and the nnx.Optimizer.
    """
    rngs = nnx.Rngs(seed)
    model = ResNet50Classifier(num_classes=num_classes, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    return model, optimizer


@nnx.jit
def train_step(
    model: ResNet50Classifier,
    optimizer: nnx.Optimizer,
    images: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Performs a single training step, computing loss, gradients, and updating model parameters.

    Args:
        model: The ResNet50Classifier model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        images: A batch of input images.
        labels: A batch of corresponding integer labels.

    Returns:
        The scalar mean cross-entropy loss for the batch.
    """

    def loss_fn(model: ResNet50Classifier) -> jax.Array:
        logits = model(images, train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(
    model: ResNet50Classifier,
    images: jax.Array,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Computes the loss and accuracy for a batch of data without updating model parameters.

    Args:
        model: The ResNet50Classifier model for evaluation.
        images: A batch of input images.
        labels: A batch of corresponding integer labels.

    Returns:
        A tuple containing the scalar mean cross-entropy loss and the scalar accuracy for the batch.
    """
    logits = model(images, train=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return loss, accuracy


def train_epoch(
    model: ResNet50Classifier,
    optimizer: nnx.Optimizer,
    train_ds: tf.data.Dataset,
    epoch: int,
) -> float:
    """Iterates through the training dataset for one epoch, performing training steps and logging progress.

    Args:
        model: The ResNet50Classifier model to train.
        optimizer: The nnx.Optimizer for updating model parameters.
        train_ds: The TensorFlow dataset containing training data.
        epoch: The current epoch number for logging purposes.

    Returns:
        The average loss across all batches in the epoch.
    """
    losses = []
    batches = cast(Iterable[DatasetBatch], train_ds.as_numpy_iterator())

    pbar = tqdm(
        batches,
        desc=f"Epoch {epoch}",
        leave=False,
        total=int(train_ds.cardinality().numpy()),
    )
    for (images, _texts), labels in pbar:
        images_jax = jnp.array(images)
        labels_jax = jnp.array(labels)

        loss = train_step(model, optimizer, images_jax, labels_jax)
        losses.append(float(loss))
        pbar.set_postfix(loss=f"{loss:.4f}")

    return sum(losses) / len(losses) if losses else 0.0


def eval_epoch(
    model: ResNet50Classifier, val_ds: tf.data.Dataset
) -> tuple[float, float]:
    """Evaluates the model on a validation dataset, computing average loss and accuracy.

    Args:
        model: The ResNet50Classifier model to evaluate.
        val_ds: The TensorFlow dataset containing validation data.

    Returns:
        A tuple containing the average loss and average accuracy over the validation dataset.
    """
    losses = []
    accuracies = []

    batches = cast(Iterable[DatasetBatch], val_ds.as_numpy_iterator())
    for (images, _texts), labels in batches:
        images_jax = jnp.array(images)
        labels_jax = jnp.array(labels)

        loss, accuracy = eval_step(model, images_jax, labels_jax)
        losses.append(float(loss))
        accuracies.append(float(accuracy))

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return avg_loss, avg_accuracy


def train(
    num_epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_classes: int = 8,
    train_manifest: str = "/root/assets/manifests/split_train.jsonl",
    val_manifest: str = "/root/assets/manifests/split_val.jsonl",
    test_manifest: str = "/root/assets/manifests/split_test.jsonl",
    seed: int = 0,
) -> ResNet50Classifier:
    """Executes the full training loop, including data loading, model initialization, and epoch-wise training and evaluation.

    Args:
        num_epochs: The total number of epochs to train the model.
        batch_size: The batch size for training and validation data loaders.
        learning_rate: The learning rate for the optimizer.
        num_classes: The number of output classes for the classifier.
        train_manifest: Path to training manifest.
        val_manifest: Path to validation manifest.
        test_manifest: Path to test manifest.
        seed: Random seed for model initialization.

    Returns:
        The trained ResNet50Classifier model.
    """
    print("Loading datasets...")
    print(f"  Train: {train_manifest}")
    print(f"  Val: {val_manifest}")
    print(f"  Test: {test_manifest}")
    train_ds, val_ds, test_ds = get_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        batch_size=batch_size,
    )

    print("Creating model and optimizer...")
    model, optimizer = create_train_state(
        num_classes=num_classes,
        learning_rate=learning_rate,
        seed=seed,
    )

    print("Starting training...")
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss = train_epoch(model, optimizer, train_ds, epoch + 1)
        val_loss, val_accuracy = eval_epoch(model, val_ds)
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    print("Running test evaluation...")
    test_loss, test_accuracy = eval_epoch(model, test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model
