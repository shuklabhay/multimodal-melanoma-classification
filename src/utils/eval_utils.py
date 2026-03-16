import os
import random
from collections import Counter
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import wandb
from flax import nnx, traverse_util
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from safetensors.flax import load_file, save_file

from src.pipelines.dataset_config import ISIC19_LABEL_CONFIG

CHECKPOINT_BASE_DIR = "/data/checkpoints"


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation handling.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens.

    Args:
        tokens: List of tokens.
        n: N-gram size.

    Returns:
        List of n-gram tuples.
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_distinct_n(texts: list[str], n: int) -> float:
    """Calculate Distinct-n metric for a corpus of texts.

    Distinct-n measures lexical diversity as the ratio of unique n-grams
    to total n-grams across all texts. Higher values indicate more diversity.

    Args:
        texts: List of text strings.
        n: N-gram size (1 for unigrams, 2 for bigrams).

    Returns:
        Distinct-n score between 0 and 1.
    """
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = tokenize(text)
        all_ngrams.extend(get_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams


def calculate_self_bleu(texts: list[str], sample_size: int = 500) -> float:
    """Calculate Self-BLEU metric for a corpus of texts.

    Self-BLEU measures how similar generated texts are to each other.
    Lower values indicate more diversity (less repetition across samples).

    For efficiency, computes on a random sample if corpus is large.

    Args:
        texts: List of text strings.
        sample_size: Maximum number of texts to sample for computation.

    Returns:
        Self-BLEU score between 0 and 1. Lower is more diverse.
    """
    if len(texts) < 2:
        return 0.0

    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    tokenized = [tokenize(text) for text in texts]
    smoothing = SmoothingFunction().method1

    bleu_scores = []
    for i, hypothesis in enumerate(tokenized):
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if references and hypothesis:
            score = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
            bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


class NarrativeDiversityTracker:
    """Tracks narrative diversity metrics incrementally during generation.

    Maintains running counts for efficient Distinct-n computation and
    stores texts for Self-BLEU computation at the end.
    """

    def __init__(self) -> None:
        """Initialize the diversity tracker."""
        self.unigram_counts: Counter[tuple[str, ...]] = Counter()
        self.bigram_counts: Counter[tuple[str, ...]] = Counter()
        self.total_unigrams = 0
        self.total_bigrams = 0
        self.texts: list[str] = []
        self.num_narratives = 0

    def update(self, text: str) -> None:
        """Add a narrative text to the tracker.

        Args:
            text: Generated narrative text.
        """
        tokens = tokenize(text)
        unigrams = get_ngrams(tokens, 1)
        bigrams = get_ngrams(tokens, 2)

        self.unigram_counts.update(unigrams)
        self.bigram_counts.update(bigrams)
        self.total_unigrams += len(unigrams)
        self.total_bigrams += len(bigrams)
        self.texts.append(text)
        self.num_narratives += 1

    def distinct_1(self) -> float:
        """Get current Distinct-1 score.

        Returns:
            Distinct-1 score (unique unigrams / total unigrams).
        """
        if self.total_unigrams == 0:
            return 0.0
        return len(self.unigram_counts) / self.total_unigrams

    def distinct_2(self) -> float:
        """Get current Distinct-2 score.

        Returns:
            Distinct-2 score (unique bigrams / total bigrams).
        """
        if self.total_bigrams == 0:
            return 0.0
        return len(self.bigram_counts) / self.total_bigrams

    def self_bleu(self, sample_size: int = 500) -> float:
        """Compute Self-BLEU on collected texts.

        Args:
            sample_size: Maximum number of texts to sample.

        Returns:
            Self-BLEU score. Lower is more diverse.
        """
        return calculate_self_bleu(self.texts, sample_size)

    def get_metrics(self, compute_self_bleu: bool = True) -> dict[str, float]:
        """Get all diversity metrics.

        Args:
            compute_self_bleu: Whether to compute Self-BLEU (expensive).

        Returns:
            Dictionary with distinct_1, distinct_2, and optionally self_bleu.
        """
        metrics = {
            "distinct_1": self.distinct_1(),
            "distinct_2": self.distinct_2(),
            "num_narratives": float(self.num_narratives),
            "unique_unigrams": float(len(self.unigram_counts)),
            "unique_bigrams": float(len(self.bigram_counts)),
        }
        if compute_self_bleu:
            metrics["self_bleu"] = self.self_bleu()
        return metrics

    def reset(self) -> None:
        """Reset all tracked state."""
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.total_unigrams = 0
        self.total_bigrams = 0
        self.texts.clear()
        self.num_narratives = 0


class ClassificationMetrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    per_class_acc: dict[str, float]
    sensitivity: dict[str, float]
    specificity: dict[str, float]


def calculate_classification_metrics(
    predictions: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> ClassificationMetrics:
    """Calculate accuracy, balanced accuracy, sensitivity, and specificity from predictions.

    Args:
        predictions: Predicted class labels of shape (batch_size,).
        labels: Ground truth labels of shape (batch_size,).
        num_classes: Number of classes.

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - balanced_accuracy: Mean of per-class recalls
            - per_class_acc: Dict of per-class accuracies
            - sensitivity: Dict of per-class sensitivity values
            - specificity: Dict of per-class specificity values
            - sensitivity_mean: Mean sensitivity across classes
            - specificity_mean: Mean specificity across classes
    """
    accuracy = float(jnp.mean(predictions == labels))

    per_class_acc = {}
    sensitivity = {}
    specificity = {}
    recalls = []

    for c in range(num_classes):
        mask = labels == c
        short = ISIC19_LABEL_CONFIG[c]["short"]
        key = f"{c}{short}"

        tp = jnp.sum((predictions == c) & (labels == c))
        fn = jnp.sum((predictions != c) & (labels == c))
        tn = jnp.sum((predictions != c) & (labels != c))
        fp = jnp.sum((predictions == c) & (labels != c))

        if jnp.sum(mask) > 0:
            recall = float(tp / (tp + fn))
            recalls.append(recall)
            per_class_acc[key] = recall
        else:
            per_class_acc[key] = 0.0

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sensitivity[f"class_{c}"] = sens
        specificity[f"class_{c}"] = spec

    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "per_class_acc": per_class_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


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
    predictions = jnp.argmax(logits, axis=-1)

    classification_metrics = calculate_classification_metrics(
        predictions, labels, num_classes
    )

    loss = sum(losses) / len(losses)
    macro_auroc, per_class_auroc = calculate_auroc(logits, labels, num_classes)
    ece = calculate_ece(logits, labels)
    brier = calculate_brier_score(logits, labels)

    sensitivity = classification_metrics["sensitivity"]
    specificity = classification_metrics["specificity"]

    metrics = {
        "accuracy": classification_metrics["accuracy"],
        "balanced_accuracy": classification_metrics["balanced_accuracy"],
        "loss": loss,
        "sensitivity_mean": sum(sensitivity.values()) / len(sensitivity),
        "specificity_mean": sum(specificity.values()) / len(specificity),
        "auroc": macro_auroc,
        "ece": ece,
        "brier": brier,
    }

    for k, v in per_class_auroc.items():
        metrics[f"auroc_{k}"] = v

    for k, v in classification_metrics["per_class_acc"].items():
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

    params_state, batch_stats_state = nnx.state(model, nnx.Param, nnx.BatchStat)
    state_to_save = {
        "params": nnx.to_pure_dict(params_state),
        "batch_stats": nnx.to_pure_dict(batch_stats_state),
    }
    flat_params = traverse_util.flatten_dict(state_to_save, sep=".")
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


def load_checkpoint(model: nnx.Module, run_name: str) -> None:
    """Load model parameters from checkpoint directory.

    Args:
        model: The model to load parameters into (modified in place).
        run_name: Name of the training run to load checkpoint from.
    """
    model_path, _ = get_checkpoint_paths(run_name)
    print(f"Loading checkpoint from {model_path}")
    flat_params = load_file(model_path)

    params_state, batch_stats_state = nnx.state(model, nnx.Param, nnx.BatchStat)

    current_state = {
        "params": nnx.to_pure_dict(params_state),
        "batch_stats": nnx.to_pure_dict(batch_stats_state),
    }
    flat_current = cast(
        dict[str, jax.Array], traverse_util.flatten_dict(current_state, sep=".")
    )

    for key in list(flat_current.keys()):
        if key in flat_params:
            flat_current[key] = flat_params[key]

    nested_updated = traverse_util.unflatten_dict(flat_current, sep=".")

    def update_recursive(state: nnx.State, updates: dict, variable_type) -> None:
        for k, v in updates.items():
            if isinstance(v, dict):
                update_recursive(getattr(state, k), v, variable_type)
            else:
                setattr(state, k, variable_type(v))

    params_updates = nested_updated.get("params", {})
    batch_stats_updates = nested_updated.get("batch_stats", {})

    update_recursive(params_state, params_updates, nnx.Param)
    update_recursive(batch_stats_state, batch_stats_updates, nnx.BatchStat)

    nnx.update(model, params_state)
    nnx.update(model, batch_stats_state)


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
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    title: str = "Confusion Matrix",
) -> None:
    """Log confusion matrix to WandB.

    Args:
        predictions: Predicted class labels of shape (batch_size,).
        labels: Integer labels of shape (batch_size,).
        title: Title for the confusion matrix plot.
    """
    class_names = [
        ISIC19_LABEL_CONFIG[i]["name"] for i in range(len(ISIC19_LABEL_CONFIG))
    ]
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
    test_predictions: jnp.ndarray,
    test_labels: jnp.ndarray,
) -> None:
    """Log final training metrics to WandB and finish the run.

    Args:
        best_val_balanced_acc: The best validation balanced accuracy achieved.
        best_epoch: The epoch at which best validation balanced accuracy was achieved.
        test_metrics: Dictionary containing test set metrics.
        test_predictions: Test set predictions for confusion matrix.
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

    log_confusion_matrix(test_predictions, test_labels, "Test Confusion Matrix")

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
