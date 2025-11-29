import json
import shutil
import subprocess
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

    return tf.image.resize(image, [new_h, new_w])


def create_tf_dataset_from_bundle(
    bundle_path: str,
    batch_size: int,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a preprocessed NPZ bundle.

    Args:
        bundle_path: Path to the NPZ bundle file.
        batch_size: Batch size for the dataset.

    Returns:
        A tf.data.Dataset yielding ((image, text), label) tuples.
    """
    print(f"Loading bundle from {bundle_path}...")
    bundle = np.load(bundle_path, allow_pickle=True)
    images = bundle["images"]
    texts = bundle["texts"]
    labels = bundle["labels"]
    print(f"✓ Loaded {len(labels)} samples from bundle")

    def format_batch(image, text, label):
        return (image, text), label

    dataset = tf.data.Dataset.from_tensor_slices((images, texts, labels))
    dataset = dataset.map(format_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    num_batches = (len(labels) + batch_size - 1) // batch_size
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_batches))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataloaders(
    train_bundle: str = "/data/bundled/train.npz",
    val_bundle: str = "/data/bundled/val.npz",
    test_bundle: str = "/data/bundled/test.npz",
    batch_size: int = 32,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Creates train, validation, and test dataloaders from bundled NPZ files.

    Args:
        train_bundle: Path to training bundle NPZ.
        val_bundle: Path to validation bundle NPZ.
        test_bundle: Path to test bundle NPZ.
        batch_size: Batch size for all datasets.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_ds = create_tf_dataset_from_bundle(train_bundle, batch_size=batch_size)
    val_ds = create_tf_dataset_from_bundle(val_bundle, batch_size=batch_size)
    test_ds = create_tf_dataset_from_bundle(test_bundle, batch_size=batch_size)

    return train_ds, val_ds, test_ds


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


def preprocess_and_bundle_data(
    manifest_dir: str = "/root/assets/manifests",
    output_dir: str = "/data/bundled",
    image_size: tuple[int, int] = (224, 224),
) -> None:
    """Bundles manifest data into NPZ files for each split.

    Args:
        manifest_dir: Directory containing split_train.jsonl, split_val.jsonl, split_test.jsonl.
        output_dir: Directory to save bundled NPZ files.
        image_size: Target image size (height, width).
    """
    manifest_path = Path(manifest_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_h, target_w = image_size
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    splits = [
        ("train", True),
        ("val", False),
        ("test", False),
    ]

    for split_name, is_training in splits:
        manifest_file = manifest_path / f"split_{split_name}.jsonl"
        print(f"\nBundling {split_name} split from {manifest_file}...")
        manifest = load_manifest(str(manifest_file))

        all_images = []
        all_texts = []
        all_labels = []

        for item in tqdm(manifest, desc=f"Processing {split_name}"):
            image = tf.io.read_file(item["image_path"])
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = cast(tf.Tensor, image)

            if is_training:
                resize_target = max(target_h, target_w) + 32
            else:
                resize_target = max(target_h, target_w)
            image = resize_shortest_edge(image, resize_target)

            if is_training:
                image = tf.image.random_crop(image, [target_h, target_w, 3])
            else:
                image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)

            image = tf.cast(image, tf.float32) / 255.0
            image = (image - mean) / std

            text = tf.io.read_file(item["text_path"])
            text = tf.strings.strip(text)

            all_images.append(image.numpy())
            all_texts.append(text.numpy())
            all_labels.append(item["label"])

        bundle_path = output_path / f"{split_name}.npz"
        np.savez(
            bundle_path,
            images=np.array(all_images, dtype=np.float32),
            texts=np.array(all_texts, dtype=object),
            labels=np.array(all_labels, dtype=np.int32),
        )
        print(f"✓ Saved {split_name} bundle to {bundle_path} ({len(manifest)} samples)")
