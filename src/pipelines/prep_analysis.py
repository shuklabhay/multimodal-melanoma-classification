import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.models.backbones import ResNet50Backbone
from src.pipelines.label_config import LABEL_CONFIG, SIMILAR_PAIRS, SPLIT_COLORS


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


def _preprocess_single_image(img: np.ndarray) -> np.ndarray:
    """Preprocess a single image for ResNet50.

    Args:
        img: Raw image of shape (H, W, 3) as uint8.

    Returns:
        Preprocessed image of shape (224, 224, 3) as float32.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_tensor = tf.convert_to_tensor(img)
    img_float = tf.math.divide(tf.cast(img_tensor, tf.float32), 255.0)
    img_resized = tf.image.resize_with_crop_or_pad(img_float, 224, 224)

    img_resized_np = np.asarray(img_resized)
    img_normalized = (img_resized_np - mean) / std
    return img_normalized


def preprocess_images_for_resnet(
    images: np.ndarray, num_workers: int = 4
) -> np.ndarray:
    """Preprocess images for ResNet50 using parallel workers.

    Args:
        images: Raw images of shape (N, H, W, 3) as uint8.
        num_workers: Number of parallel workers for preprocessing.

    Returns:
        Preprocessed images of shape (N, 224, 224, 3) as float32.
    """
    processed: list[np.ndarray | None] = [None] * len(images)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_preprocess_single_image, img): idx
            for idx, img in enumerate(images)
        }

        for future in tqdm(
            as_completed(futures), total=len(images), desc="Preprocessing images"
        ):
            idx = futures[future]
            processed[idx] = future.result()
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

    _, ax = plt.subplots(figsize=figsize)
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


def build_knn_graph(
    embeddings: np.ndarray, k: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN graph using original embeddings.

    Args:
        embeddings: Feature embeddings of shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        Tuple of (distances, indices) arrays of shape (N, k).
    """
    print(f"Building k-NN graph with k={k}...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    return distances[:, 1:], indices[:, 1:]


def compute_mixing_index(
    labels: np.ndarray, knn_indices: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute mixing index matrix showing fraction of k-NN from each class.

    Args:
        labels: Class labels of shape (N,).
        knn_indices: k-NN indices of shape (N, k).
        num_classes: Number of classes.

    Returns:
        Mixing matrix of shape (num_classes, num_classes) where [i, j] is the
        fraction of class i samples that have class j neighbors.
    """
    print("Computing mixing indices...")
    mixing_matrix = np.zeros((num_classes, num_classes))

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_samples = np.where(class_mask)[0]

        if len(class_samples) == 0:
            continue

        neighbor_classes = labels[knn_indices[class_samples]].flatten()
        for neighbor_class in range(num_classes):
            count = np.sum(neighbor_classes == neighbor_class)
            total_neighbors = len(class_samples) * knn_indices.shape[1]
            mixing_matrix[class_idx, neighbor_class] = count / total_neighbors

    return mixing_matrix


def rank_class_pairs(
    mixing_matrix: np.ndarray, class_names: list[str]
) -> list[tuple[str, str, float]]:
    """Rank pairwise look-alike scores.

    Args:
        mixing_matrix: Mixing matrix of shape (num_classes, num_classes).
        class_names: List of class names.

    Returns:
        List of (class_i, class_j, score) tuples, sorted by score descending.
    """
    num_classes = len(class_names)
    pairs = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            score = (mixing_matrix[i, j] + mixing_matrix[j, i]) / 2
            pairs.append((class_names[i], class_names[j], score))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def find_cross_class_examples(
    class_a: int,
    class_b: int,
    labels: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_examples: int = 5,
) -> list[tuple[int, int, float]]:
    """Find specific examples where class A samples have class B neighbors.

    Args:
        class_a: First class index.
        class_b: Second class index.
        labels: Class labels of shape (N,).
        knn_indices: k-NN indices of shape (N, k).
        knn_distances: k-NN distances of shape (N, k).
        n_examples: Number of examples to return.

    Returns:
        List of (idx_a, idx_b, distance) tuples.
    """
    examples = []
    class_a_samples = np.where(labels == class_a)[0]

    for idx_a in class_a_samples:
        neighbors = knn_indices[idx_a]
        neighbor_labels = labels[neighbors]
        class_b_mask = neighbor_labels == class_b

        if np.any(class_b_mask):
            class_b_neighbors = neighbors[class_b_mask]
            class_b_distances = knn_distances[idx_a][class_b_mask]
            closest_idx = np.argmin(class_b_distances)
            idx_b = class_b_neighbors[closest_idx]
            dist = class_b_distances[closest_idx]
            examples.append((idx_a, idx_b, dist))

    examples.sort(key=lambda x: x[2])
    return examples[:n_examples]


def plot_specific_pairs(
    target_pairs: list[tuple[int, int]],
    labels: np.ndarray,
    images: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    output_path: str,
    n_examples: int = 5,
) -> None:
    """Plot specific class pairs with multiple examples in columns with custom spacing.

    Args:
        target_pairs: List of (class_a, class_b) tuples to visualize.
        labels: Class labels of shape (N,).
        images: Raw images of shape (N, H, W, 3).
        knn_indices: k-NN indices of shape (N, k).
        knn_distances: k-NN distances of shape (N, k).
        output_path: Path to save the figure.
        n_examples: Number of examples per pair.
    """
    print("Creating specific pairs visualization...")

    class_shorts = [LABEL_CONFIG[i]["short"] for i in range(len(LABEL_CONFIG))]
    n_pairs = len(target_pairs)

    width_ratios = []
    for i in range(n_pairs):
        width_ratios.extend([1, 1])
        if i < n_pairs - 1:
            width_ratios.append(0.5)

    fig = plt.figure(figsize=(n_pairs * 3.5, n_examples * 1.8))
    gs = fig.add_gridspec(
        n_examples,
        n_pairs * 2 + (n_pairs - 1),
        wspace=0.05,
        hspace=0.1,
        width_ratios=width_ratios,
    )

    for pair_idx, (class_a, class_b) in enumerate(target_pairs):
        examples = find_cross_class_examples(
            class_a, class_b, labels, knn_indices, knn_distances, n_examples
        )

        pair_name = f"{class_shorts[class_a]} vs {class_shorts[class_b]}"
        print(f"  {pair_name}: Found {len(examples)} examples")

        col_start = pair_idx * 3

        for ex_idx, (idx_a, idx_b, dist) in enumerate(examples):
            ax_a = fig.add_subplot(gs[ex_idx, col_start])
            ax_a.imshow(images[idx_a])
            ax_a.axis("off")
            if ex_idx == 0:
                ax_a.set_title(
                    f"{class_shorts[class_a]}", fontsize=11, fontweight="bold"
                )

            ax_b = fig.add_subplot(gs[ex_idx, col_start + 1])
            ax_b.imshow(images[idx_b])
            ax_b.axis("off")
            if ex_idx == 0:
                ax_b.set_title(
                    f"{class_shorts[class_b]}", fontsize=11, fontweight="bold"
                )

        for ex_idx in range(len(examples), n_examples):
            ax_a = fig.add_subplot(gs[ex_idx, col_start])
            ax_a.axis("off")
            ax_b = fig.add_subplot(gs[ex_idx, col_start + 1])
            ax_b.axis("off")

    fig.suptitle(
        "k-NN Mixing: Most Confused Class Pairs",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved specific pairs plot to {output_path}")


def generate_test_tsne(
    bundle_path: str = "/data/bundled/test.npz",
    output_dir: str = "/data/buffer",
    batch_size: int = 128,
) -> None:
    """Generate t-SNE visualization for test set.

    Args:
        bundle_path: Path to test bundle NPZ.
        output_dir: Directory to save output figures.
        batch_size: Batch size for ResNet inference.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images, _, labels = load_bundle_raw(bundle_path)

    processed_images = preprocess_images_for_resnet(images)
    resnet_embeddings = extract_resnet_embeddings(processed_images, batch_size)
    tsne_coords = compute_tsne(resnet_embeddings)

    plot_tsne_with_centroids(
        tsne_coords,
        labels,
        f"{output_dir}/test_tsne.png",
        title="t-SNE of ResNet50 Embeddings (Test Set)",
    )


def generate_similar_pairs(
    bundle_path: str = "/data/bundled/test.npz",
    output_dir: str = "/data/buffer",
    batch_size: int = 128,
    k: int = 20,
) -> None:
    """Generate k-NN mixing analysis and similar pairs visualization for test set.

    Args:
        bundle_path: Path to test bundle NPZ.
        output_dir: Directory to save output figures.
        batch_size: Batch size for ResNet inference.
        k: Number of nearest neighbors for mixing analysis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images, _, labels = load_bundle_raw(bundle_path)

    processed_images = preprocess_images_for_resnet(images)
    resnet_embeddings = extract_resnet_embeddings(processed_images, batch_size)

    knn_distances, knn_indices = build_knn_graph(resnet_embeddings, k=k)
    num_classes = len(LABEL_CONFIG)
    mixing_matrix = compute_mixing_index(labels, knn_indices, num_classes)

    class_names = [LABEL_CONFIG[i]["name"] for i in range(num_classes)]
    ranked_pairs = rank_class_pairs(mixing_matrix, class_names)

    print("\nTop 10 most confused class pairs (by k-NN mixing):")
    for i, (name_a, name_b, score) in enumerate(ranked_pairs[:10], 1):
        print(f"  {i}. {name_a} ↔ {name_b}: {score:.4f}")

    class_to_idx = {LABEL_CONFIG[i]["short"]: i for i in range(num_classes)}

    target_pairs = [
        (class_to_idx[short_a], class_to_idx[short_b])
        for short_a, short_b in SIMILAR_PAIRS
    ]

    plot_specific_pairs(
        target_pairs,
        labels,
        images,
        knn_indices,
        knn_distances,
        f"{output_dir}/test_similar_pairs.png",
        n_examples=5,
    )


def compute_separation_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute class-wise silhouette and Davies-Bouldin scores on embeddings.

    Args:
        embeddings: Feature embeddings of shape (N, D).
        labels: Class labels of shape (N,).

    Returns:
        Dictionary with overall and class-wise separation metrics.
    """
    num_classes = len(LABEL_CONFIG)
    class_names = [LABEL_CONFIG[i]["name"] for i in range(num_classes)]
    class_shorts = [LABEL_CONFIG[i]["short"] for i in range(num_classes)]

    print("Computing overall silhouette score...")
    overall_silhouette = silhouette_score(embeddings, labels)

    print("Computing overall Davies-Bouldin index...")
    overall_dbi = davies_bouldin_score(embeddings, labels)

    print("Computing per-sample silhouette scores...")
    sample_silhouettes = silhouette_samples(embeddings, labels)

    class_silhouettes = {}
    class_counts = {}
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_silhouettes[class_shorts[class_idx]] = float(
                sample_silhouettes[mask].mean()
            )
            class_counts[class_shorts[class_idx]] = int(mask.sum())
        else:
            class_silhouettes[class_shorts[class_idx]] = None
            class_counts[class_shorts[class_idx]] = 0

    return {
        "overall_silhouette": float(overall_silhouette),
        "overall_davies_bouldin": float(overall_dbi),
        "class_silhouette": class_silhouettes,
        "class_counts": class_counts,
        "class_names": {class_shorts[i]: class_names[i] for i in range(num_classes)},
    }


def compute_pairwise_distances(
    embeddings: np.ndarray, labels: np.ndarray, class_a: int, class_b: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute intra-class and inter-class distances for a pair of classes.

    Args:
        embeddings: Feature embeddings of shape (N, D).
        labels: Class labels of shape (N,).
        class_a: First class index.
        class_b: Second class index.

    Returns:
        Tuple of (intra_distances, inter_distances) arrays.
    """

    mask_a = labels == class_a
    mask_b = labels == class_b

    emb_a = embeddings[mask_a]
    emb_b = embeddings[mask_b]

    dist_aa = cosine_distances(emb_a, emb_a)
    dist_bb = cosine_distances(emb_b, emb_b)

    intra_a = dist_aa[np.triu_indices(len(emb_a), k=1)]
    intra_b = dist_bb[np.triu_indices(len(emb_b), k=1)]
    intra_distances = np.concatenate([intra_a, intra_b])

    dist_ab = cosine_distances(emb_a, emb_b)
    inter_distances = dist_ab.flatten()
    return intra_distances, inter_distances


def compute_overlap_coefficient(
    intra_distances: np.ndarray, inter_distances: np.ndarray, n_bins: int = 100
) -> float:
    """Compute overlap coefficient between two distance distributions.

    The overlap coefficient is the area of intersection divided by the
    minimum of the two distribution areas.

    Args:
        intra_distances: Intra-class distances.
        inter_distances: Inter-class distances.
        n_bins: Number of histogram bins.

    Returns:
        Overlap coefficient in [0, 1]. Higher means more overlap.
    """
    all_distances = np.concatenate([intra_distances, inter_distances])
    bins = np.linspace(all_distances.min(), all_distances.max(), n_bins + 1)

    hist_intra, _ = np.histogram(intra_distances, bins=bins, density=True)
    hist_inter, _ = np.histogram(inter_distances, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    intersection = np.minimum(hist_intra, hist_inter).sum() * bin_width
    return float(intersection)


def generate_class_separation_analysis(
    bundle_dir: str = "/data/bundled",
    output_dir: str = "/data/buffer",
    batch_size: int = 128,
) -> None:
    """Generate unified class separation analysis with metrics and distance distributions.

    Creates a combined figure with:
    - Top row: Class-wise silhouette scores (bar chart) and overall metrics (bar chart)
    - Bottom rows: Distance distributions for similar condition pairs

    All plots show train/val/test splits with consistent colors.

    Args:
        bundle_dir: Directory containing train.npz, val.npz, test.npz bundles.
        output_dir: Directory to save outputs.
        batch_size: Batch size for ResNet inference.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_classes = len(LABEL_CONFIG)
    class_shorts = [LABEL_CONFIG[i]["short"] for i in range(num_classes)]
    class_to_idx = {LABEL_CONFIG[i]["short"]: i for i in range(num_classes)}

    # Collect embeddings and metrics for all splits
    split_data: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}

    for split in ["train", "val", "test"]:
        bundle_path = f"{bundle_dir}/{split}.npz"
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")

        images, _, labels = load_bundle_raw(bundle_path)
        processed_images = preprocess_images_for_resnet(images)
        resnet_embeddings = extract_resnet_embeddings(processed_images, batch_size)
        print(f"ResNet embeddings shape: {resnet_embeddings.shape}")

        metrics = compute_separation_metrics(resnet_embeddings, labels)
        all_metrics[split] = metrics
        split_data[split] = {"embeddings": resnet_embeddings, "labels": labels}

        print(f"\n{split.upper()} Results:")
        print(f"  Overall Silhouette: {metrics['overall_silhouette']:.4f}")
        print(f"  Overall Davies-Bouldin: {metrics['overall_davies_bouldin']:.4f}")

    overlap_coefficients: dict[str, dict] = {
        split: {} for split in ["train", "val", "test"]
    }
    for split in ["train", "val", "test"]:
        embeddings = split_data[split]["embeddings"]
        labels = split_data[split]["labels"]
        for short_a, short_b in SIMILAR_PAIRS:
            class_a = class_to_idx[short_a]
            class_b = class_to_idx[short_b]
            intra_dist, inter_dist = compute_pairwise_distances(
                embeddings, labels, class_a, class_b
            )
            overlap = compute_overlap_coefficient(intra_dist, inter_dist)
            pair_key = f"{short_a}_vs_{short_b}"
            overlap_coefficients[split][pair_key] = overlap

    n_pairs = len(SIMILAR_PAIRS)
    n_dist_cols = 3
    n_dist_rows = (n_pairs + n_dist_cols - 1) // n_dist_cols

    fig = plt.figure(figsize=(15, 5 + 4 * n_dist_rows))
    gs = fig.add_gridspec(1 + n_dist_rows, 3, height_ratios=[1.2] + [1] * n_dist_rows)

    ax_silhouette = fig.add_subplot(gs[0, 0:2])
    x = np.arange(num_classes)
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        scores = [
            all_metrics[split]["class_silhouette"].get(short, 0) or 0
            for short in class_shorts
        ]
        ax_silhouette.bar(
            x + i * width,
            scores,
            width,
            label=split.capitalize(),
            color=SPLIT_COLORS[split],
            alpha=0.8,
        )
    ax_silhouette.set_xlabel("Class", fontsize=10)
    ax_silhouette.set_ylabel("Silhouette Score", fontsize=10)
    ax_silhouette.set_title(
        "Class-wise Silhouette Scores", fontsize=12, fontweight="bold"
    )
    ax_silhouette.set_xticks(x + width)
    ax_silhouette.set_xticklabels(class_shorts, fontsize=9)
    ax_silhouette.legend(fontsize=9)
    ax_silhouette.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax_overall = fig.add_subplot(gs[0, 2])
    metrics_names = ["Silhouette", "Davies-Bouldin"]
    x_metrics = np.arange(len(metrics_names))
    for i, split in enumerate(["train", "val", "test"]):
        values = [
            all_metrics[split]["overall_silhouette"],
            all_metrics[split]["overall_davies_bouldin"],
        ]
        ax_overall.bar(
            x_metrics + i * width,
            values,
            width,
            label=split.capitalize(),
            color=SPLIT_COLORS[split],
            alpha=0.8,
        )
    ax_overall.set_ylabel("Score", fontsize=10)
    ax_overall.set_title("Overall Separation Metrics", fontsize=12, fontweight="bold")
    ax_overall.set_xticks(x_metrics + width)
    ax_overall.set_xticklabels(metrics_names, fontsize=9)
    ax_overall.legend(fontsize=9)

    print("\nComputing distance distributions for similar condition pairs...")
    for pair_idx, (short_a, short_b) in enumerate(SIMILAR_PAIRS):
        row = 1 + pair_idx // n_dist_cols
        col = pair_idx % n_dist_cols
        ax = fig.add_subplot(gs[row, col])

        class_a = class_to_idx[short_a]
        class_b = class_to_idx[short_b]

        for split in ["train", "val", "test"]:
            embeddings = split_data[split]["embeddings"]
            labels = split_data[split]["labels"]
            intra_dist, inter_dist = compute_pairwise_distances(
                embeddings, labels, class_a, class_b
            )

            ax.hist(
                inter_dist,
                bins=40,
                alpha=0.5,
                label=f"{split.capitalize()}",
                density=True,
                color=SPLIT_COLORS[split],
            )

        pair_key = f"{short_a}_vs_{short_b}"
        test_overlap = overlap_coefficients["test"][pair_key]
        ax.set_xlabel("Cosine Distance", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(
            f"Inter-class Distance: {short_a} vs {short_b}\nOverlap: {test_overlap:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        print(f"  {short_a} vs {short_b}: test_overlap={test_overlap:.4f}")

    for pair_idx in range(n_pairs, n_dist_rows * n_dist_cols):
        row = 1 + pair_idx // n_dist_cols
        col = pair_idx % n_dist_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    fig.suptitle(
        "Class Separation Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/class_separation_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"\n✓ Saved figure to {output_dir}/class_separation_analysis.png")

    combined_metrics = {
        "separation_metrics": all_metrics,
        "overlap_coefficients": overlap_coefficients,
    }
    metrics_path = f"{output_dir}/class_separation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
