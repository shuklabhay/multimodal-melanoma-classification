import json

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


def create_tf_dataset(
    manifest_path: str,
    batch_size: int,
    image_size: tuple[int, int] = (224, 224),
    prefetch_size: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a combined manifest.

    Args:
        manifest_path: Path to the combined manifest JSONL file.
        batch_size: Batch size for the dataset.
        image_size: Target image size (height, width) for resizing.
        prefetch_size: Number of batches to prefetch.

    Returns:
        A tf.data.Dataset yielding ((image, text), label) tuples.
    """
    manifest = load_manifest(manifest_path)

    def generator():
        """Generator function to yield (image_path, text_path, label)."""
        for item in manifest:
            yield (item["image_path"], item["text_path"], item["label"])

    def load_and_process(image_path, text_path, label):
        """Loads and processes image and text data.

        Args:
            image_path: Path to the image file.
            text_path: Path to the text file.
            label: Integer label.

        Returns:
            Tuple of ((image, text), label).
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)

        image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
        image = tf.cast(image, tf.float32) / 255.0

        text = tf.io.read_file(text_path)
        text = tf.strings.strip(text)
        return (image, text), label

    dataset = tf.data.Dataset.from_generator(
        generator, output_types=(tf.string, tf.string, tf.int32)
    )
    dataset = dataset.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_size)

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
