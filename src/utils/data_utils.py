import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import tensorflow as tf


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

    return tf.image.resize(image, [new_h, new_w])


def create_tf_dataset(
    manifest_path: str,
    batch_size: int,
    image_size: tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a combined manifest.

    Args:
        manifest_path: Path to the combined manifest JSONL file.
        batch_size: Batch size for the dataset.
        image_size: Target image size (height, width).
        is_training: Whether this is a training dataset.

    Returns:
        A tf.data.Dataset yielding ((image, text), label) tuples.
    """
    manifest = load_manifest(manifest_path)
    target_h, target_w = image_size

    print(f"Loading {len(manifest)} images into RAM...")
    all_images = []
    all_texts = []
    all_labels = []

    for i, item in enumerate(manifest):
        image = tf.io.read_file(item["image_path"])
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = cast(tf.Tensor, image)

        if is_training:
            resize_target = max(target_h, target_w) + 32
        else:
            resize_target = max(target_h, target_w)
        image = resize_shortest_edge(image, resize_target)

        if not is_training:
            image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)
            image = tf.cast(image, tf.float32) / 255.0
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = (image - mean) / std

        text = tf.io.read_file(item["text_path"])
        text = tf.strings.strip(text)

        all_images.append(image.numpy())  # type: ignore[reportOptionalCall]
        all_texts.append(text.numpy())
        all_labels.append(item["label"])

        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i + 1}/{len(manifest)} images")

    print(f"✓ All {len(manifest)} images loaded into RAM")

    def train_augment(image, text, label):
        """Apply random crop and normalization for training."""
        image = tf.image.random_crop(image, [target_h, target_w, 3])
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return (image, text), label

    def eval_passthrough(image, text, label):
        """Just restructure the tuple for eval (already processed)."""
        return (image, text), label

    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_texts, all_labels))

    if is_training:
        dataset = dataset.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(eval_passthrough, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    num_batches = (len(manifest) + batch_size - 1) // batch_size
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_batches))

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataloaders(
    train_manifest: str = "assets/manifests/split_train.jsonl",
    val_manifest: str = "assets/manifests/split_val.jsonl",
    test_manifest: str = "assets/manifests/split_test.jsonl",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Creates train, validation, and test dataloaders.

    Args:
        train_manifest: Path to training manifest.
        val_manifest: Path to validation manifest.
        test_manifest: Path to test manifest.
        batch_size: Batch size for all datasets.
        image_size: Target image size for resizing.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_ds = create_tf_dataset(
        train_manifest,
        batch_size=batch_size,
        image_size=image_size,
        is_training=True,
    )

    val_ds = create_tf_dataset(
        val_manifest,
        batch_size=batch_size,
        image_size=image_size,
    )

    test_ds = create_tf_dataset(
        test_manifest,
        batch_size=batch_size,
        image_size=image_size,
    )

    return train_ds, val_ds, test_ds


def _use_buffer(file_map: dict[str, str]) -> None:
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
