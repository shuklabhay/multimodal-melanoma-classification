from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.models.backbones import ResNet50Backbone
from src.pipelines.label_config import LABEL_CONFIG


def load_bundle_raw(bundle_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw data from bundle without preprocessing.

    Args:
        bundle_path: Path to the NPZ bundle file.

    Returns:
        Tuple of (images, embeddings, labels) as numpy arrays.
    """
    print(f"Loading bundle from {bundle_path}...")
    bundle = np.load(bundle_path, allow_pickle=True)
    images = bundle["images"]
    embeddings = bundle["embeddings"]
    labels = bundle["labels"]
    print(f"Loaded {len(labels)} samples")
    return images, embeddings, labels


def preprocess_images_for_resnet(images: np.ndarray) -> np.ndarray:
    """Preprocess images for ResNet50.

    Args:
        images: Raw images of shape (N, H, W, 3) as uint8.

    Returns:
        Preprocessed images of shape (N, 224, 224, 3) as float32.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed = []
    for img in tqdm(images, desc="Preprocessing images"):
        img_tensor = tf.convert_to_tensor(img)
        img_float = tf.cast(img_tensor, tf.float32) / 255.0
        img_resized = tf.image.resize_with_crop_or_pad(img_float, 224, 224)
        img_normalized = (img_resized.numpy() - mean) / std
        processed.append(img_normalized)

    return np.array(processed, dtype=np.float32)


def extract_resnet_embeddings(images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Extract ResNet50 embeddings from preprocessed images.

    Args:
        images: Preprocessed images of shape (N, 224, 224, 3).
        batch_size: Batch size for inference.

    Returns:
        Embeddings of shape (N, 2048).
    """
    print("Loading ResNet50 backbone...")
    backbone = ResNet50Backbone()

    all_embeddings = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    print(f"Extracting embeddings ({num_batches} batches)...")
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting"):
        batch = images[i : i + batch_size]
        batch_jax = jnp.array(batch)
        embeddings = backbone(batch_jax, train=False)
        all_embeddings.append(np.array(embeddings))

    return np.concatenate(all_embeddings, axis=0)


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE dimensionality reduction.

    Args:
        embeddings: High-dimensional embeddings of shape (N, D).
        n_components: Number of output dimensions.
        perplexity: t-SNE perplexity (related to number of nearest neighbors).
        max_iter: Maximum number of iterations.
        random_state: Random seed for reproducibility.

    Returns:
        t-SNE embeddings of shape (N, n_components).
    """
    print(f"Computing t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
        n_jobs=-1,
    )
    result = tsne.fit_transform(embeddings)
    return np.asarray(result)


def compute_class_centroids(
    tsne_embeddings: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute centroids for each class in t-SNE space.

    Args:
        tsne_embeddings: t-SNE coordinates of shape (N, 2).
        labels: Class labels of shape (N,).
        num_classes: Number of classes.

    Returns:
        Centroids of shape (num_classes, 2).
    """
    centroids = np.zeros((num_classes, 2))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = tsne_embeddings[mask].mean(axis=0)
    return centroids


def plot_tsne_with_centroids(
    tsne_embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "t-SNE Visualization of Lesion Embeddings",
    figsize: tuple[int, int] = (14, 11),
    alpha: float = 0.6,
    point_size: int = 15,
) -> None:
    """Plot t-SNE visualization colored by class with centroids marked.

    Args:
        tsne_embeddings: t-SNE coordinates of shape (N, 2).
        labels: Class labels of shape (N,).
        output_path: Path to save the figure.
        title: Plot title.
        figsize: Figure size.
        alpha: Point transparency.
        point_size: Size of scatter points.
    """
    print("Creating t-SNE plot with centroids...")

    class_names = [LABEL_CONFIG[i]["name"] for i in range(len(LABEL_CONFIG))]
    num_classes = len(class_names)

    cmap = plt.cm.get_cmap("tab10", num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    centroids = compute_class_centroids(tsne_embeddings, labels, num_classes)

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        ax.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            c=[colors[class_idx]],
            label=f"{class_names[class_idx]} (n={mask.sum()})",
            alpha=alpha,
            s=point_size,
            edgecolors="none",
        )

    for class_idx in range(num_classes):
        ax.scatter(
            centroids[class_idx, 0],
            centroids[class_idx, 1],
            c=[colors[class_idx]],
            s=200,
            marker="X",
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def find_similar_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray,
    n_pairs: int = 10,
) -> list[tuple[int, int, float]]:
    """Find pairs of images from different classes that are most visually similar.

    Args:
        embeddings: ResNet embeddings of shape (N, 2048).
        labels: Class labels of shape (N,).
        images: Raw images of shape (N, H, W, 3).
        n_pairs: Number of pairs to return.

    Returns:
        List of (idx1, idx2, similarity) tuples, sorted by similarity descending.
    """
    print("Computing pairwise similarities...")
    similarities = cosine_similarity(embeddings)

    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                pairs.append((i, j, similarities[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n_pairs]


def plot_similar_pairs(
    pairs: list[tuple[int, int, float]],
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Plot pairs of visually similar images from different classes.

    Args:
        pairs: List of (idx1, idx2, similarity) tuples.
        images: Raw images of shape (N, H, W, 3).
        labels: Class labels of shape (N,).
        output_path: Path to save the figure.
    """
    print("Creating similar pairs visualization...")

    class_names = [LABEL_CONFIG[i]["name"] for i in range(len(LABEL_CONFIG))]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, 3 * n_pairs))

    for row, (idx1, idx2, sim) in enumerate(pairs):
        img1 = images[idx1]
        img2 = images[idx2]
        label1 = class_names[labels[idx1]]
        label2 = class_names[labels[idx2]]

        axes[row, 0].imshow(img1)
        axes[row, 0].set_title(f"{label1}", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img2)
        axes[row, 1].set_title(f"{label2}", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 0].set_ylabel(
            f"Sim: {sim:.3f}", fontsize=10, rotation=0, labelpad=50, va="center"
        )

    fig.suptitle(
        "Most Visually Similar Pairs (Different Classes)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved similar pairs plot to {output_path}")


def generate_test_tsne(
    bundle_path: str = "/data/bundled/test.npz",
    output_dir: str = "/data/buffer",
    batch_size: int = 128,
) -> None:
    """Generate t-SNE visualization for test set using ResNet50 embeddings.

    Args:
        bundle_path: Path to test bundle NPZ.
        output_dir: Directory to save output figures.
        batch_size: Batch size for ResNet inference.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images, _, labels = load_bundle_raw(bundle_path)

    processed_images = preprocess_images_for_resnet(images)

    resnet_embeddings = extract_resnet_embeddings(processed_images, batch_size)
    print(f"ResNet embeddings shape: {resnet_embeddings.shape}")

    tsne_coords = compute_tsne(resnet_embeddings)

    plot_tsne_with_centroids(
        tsne_coords,
        labels,
        f"{output_dir}/test_tsne.png",
        title="t-SNE of ResNet50 Embeddings (Test Set)",
    )

    similar_pairs = find_similar_pairs(resnet_embeddings, labels, images, n_pairs=10)
    plot_similar_pairs(
        similar_pairs,
        images,
        labels,
        f"{output_dir}/test_similar_pairs.png",
    )
