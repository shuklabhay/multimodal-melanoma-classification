import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def load_manifest(manifest_path: str) -> list[dict]:
    """Loads JSONL manifest file and returns list of data items.

    Args:
        manifest_path: Path to JSONL manifest file.

    Returns:
        List of dictionaries containing image_path, text_path, and label.
    """
    manifest = []
    with open(manifest_path) as f:
        for line in f:
            manifest.append(json.loads(line))
    return manifest


def resize_shortest_edge(image: tf.Tensor, target_size: int) -> tf.Tensor:
    """Resizes the image so the shortest side matches target_size."""
    shape = cast(list[int], tf.shape(image))
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)

    scale = tf.cast(target_size, tf.float32) / tf.minimum(h, w)
    new_h = tf.cast(h * scale, tf.int32)
    new_w = tf.cast(w * scale, tf.int32)

    return tf.image.resize(image, [new_h, new_w], method="bicubic")


def isic_augment(image: tf.Tensor) -> tf.Tensor:
    """Applies ISIC-specific augmentations: random crop, color jitter, and flips.

    Args:
        image: Input image tensor with shape (H, W, 3).

    Returns:
        Augmented image tensor with shape (224, 224, 3).
    """
    shape = cast(list[tf.Tensor], tf.shape(image))
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    rand_val = tf.random.uniform((), dtype=tf.float32)
    scale = tf.add(0.7, tf.multiply(0.3, rand_val))
    new_h = tf.cast(h * scale, tf.int32)
    new_w = tf.cast(w * scale, tf.int32)

    image = cast(tf.Tensor, tf.image.random_crop(image, [new_h, new_w, 3]))
    image = cast(tf.Tensor, tf.image.resize(image, [224, 224]))

    image = cast(tf.Tensor, tf.image.random_flip_left_right(image))
    image = cast(tf.Tensor, tf.image.random_flip_up_down(image))

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = cast(tf.Tensor, tf.image.rot90(image, k=k))

    image = cast(tf.Tensor, tf.image.random_brightness(image, max_delta=0.6))
    image = cast(tf.Tensor, tf.image.random_contrast(image, lower=0.6, upper=1.4))
    image = cast(tf.Tensor, tf.image.random_saturation(image, lower=0.6, upper=1.4))
    image = cast(tf.Tensor, tf.image.random_hue(image, max_delta=0.1))

    image = cast(tf.Tensor, tf.clip_by_value(image, 0.0, 1.0))

    return image


def create_tf_dataset_from_bundle(
    bundle_path: str,
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a preprocessed NPZ bundle.

    Args:
        bundle_path: Path to the NPZ bundle file.
        batch_size: Batch size for the dataset.
        augment: If True, applies augmentation (for training).

    Returns:
        A tf.data.Dataset yielding ((image, embedding), label) tuples.
    """
    print(f"Loading bundle from {bundle_path}...")
    bundle = np.load(bundle_path, allow_pickle=True)
    images = bundle["images"]
    embeddings = bundle["embeddings"]
    labels = bundle["labels"]
    print(f"✓ Loaded {len(labels)} samples from bundle")

    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def process_sample(
        image: tf.Tensor, embedding: tf.Tensor, label: tf.Tensor
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        image = tf.divide(tf.cast(image, tf.float32), 255.0)

        if augment:
            image = isic_augment(image)
        else:
            image = cast(tf.Tensor, tf.image.resize_with_crop_or_pad(image, 224, 224))

        image = tf.divide(tf.subtract(image, mean), std)
        return (image, embedding), label

    dataset = tf.data.Dataset.from_tensor_slices((images, embeddings, labels))
    dataset = dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    num_batches = (len(labels) + batch_size - 1) // batch_size
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_batches))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataloaders(
    train_bundle: str,
    val_bundle: str,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Creates train and validation dataloaders from bundled NPZ files.

    Args:
        train_bundle: Path to training bundle NPZ.
        val_bundle: Path to validation bundle NPZ.
        batch_size: Batch size for all datasets.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_ds = create_tf_dataset_from_bundle(
        train_bundle,
        batch_size=batch_size,
        augment=True,
    )
    val_ds = create_tf_dataset_from_bundle(val_bundle, batch_size=batch_size)

    return train_ds, val_ds


def use_buffer(file_map: dict[str, str]) -> None:
    """Downloads buffer from Modal, moves mapped files to destinations, deletes rest.

    Args:
        file_map: Dict mapping buffer filenames to local destination paths.
    """

    print("\nDownloading outputs from Modal Volume...")
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", "--force", "data", "buffer", "."],
        check=True,
    )

    buffer_dir = Path("buffer")
    if not buffer_dir.exists():
        return

    for file in buffer_dir.iterdir():
        if file.name in file_map:
            dest_path = Path(file_map[file.name])
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            file.rename(dest_path)
            print(f"  Moved {file.name} to {dest_path}")
        else:
            file.unlink()
            print(f"  Deleted {file.name}")

    shutil.rmtree(buffer_dir)
    subprocess.run(
        ["uv", "run", "modal", "volume", "rm", "data", "buffer", "-r"],
        check=True,
    )
    print("✓ Modal buffer cleared")


def _process_item(
    item: dict, target_h: int, target_w: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Process a single manifest item (image + embedding).

    Args:
        item: Manifest item with image_path, embedding_path, and label.
        target_h: Target image height.
        target_w: Target image width.

    Returns:
        Tuple of (processed_image, embedding, label).
    """
    image = tf.io.read_file(item["image_path"])
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = cast(tf.Tensor, image)

    image = resize_shortest_edge(image, max(target_h, target_w))
    image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)
    image = cast(tf.Tensor, tf.cast(image, tf.uint8))

    embedding_data = np.load(item["embedding_path"])
    embedding = embedding_data["embedding"]

    return (
        image.numpy(),
        embedding,
        item["label"],
    )


def preprocess_and_bundle_data(
    manifest_dir: str = "/root/assets/manifests",
    output_dir: str = "/data/bundled",
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
) -> None:
    """Bundles manifest data into NPZ files for each split.

    Args:
        manifest_dir: Directory containing split_train.jsonl, split_val.jsonl, split_test.jsonl.
        output_dir: Directory to save bundled NPZ files.
        image_size: Target image size (height, width).
        num_workers: Number of parallel workers for processing.
    """
    manifest_path = Path(manifest_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_h, target_w = 256, 256

    splits = ["train", "val", "test"]

    for split_name in splits:
        manifest_file = manifest_path / f"split_{split_name}.jsonl"
        print(f"\nBundling {split_name} split...")
        manifest = load_manifest(str(manifest_file))

        results: list[tuple[int, np.ndarray, np.ndarray, int]] = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_item, item, target_h, target_w): idx
                for idx, item in enumerate(manifest)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(manifest),
                desc=f"Processing {split_name}",
            ):
                idx = futures[future]
                image, embedding, label = future.result()
                results.append((idx, image, embedding, label))

        results.sort(key=lambda x: x[0])

        all_images = [r[1] for r in results]
        all_embeddings = [r[2] for r in results]
        all_labels = [r[3] for r in results]

        bundle_path = output_path / f"{split_name}.npz"
        np.savez(
            bundle_path,
            images=np.array(all_images, dtype=np.uint8),
            embeddings=np.array(all_embeddings, dtype=np.float32),
            labels=np.array(all_labels, dtype=np.int32),
        )
        print(f"✓ Saved {split_name}")
