import json
from typing import cast

import numpy as np
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
    prefetch_size: int = tf.data.AUTOTUNE,
    is_training: bool = False,
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from a combined manifest.

    Args:
        manifest_path: Path to the combined manifest JSONL file.
        batch_size: Batch size for the dataset.
        image_size: Target image size (height, width).
        prefetch_size: Number of batches to prefetch.
        is_training: Whether this is a training dataset.

    Returns:
        A tf.data.Dataset yielding ((image, text), label) tuples.
    """
    manifest = load_manifest(manifest_path)

    def generator():
        """Generator function to yield (image_path, text_path, label)."""
        for item in manifest:
            yield (item["image_path"], item["text_path"], item["label"])

    def load_and_process(image_path, text_path, label):
        """Loads and processes image and text data."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = cast(tf.Tensor, image)

        target_h, target_w = image_size
        if is_training:
            resize_target = max(target_h, target_w) + 32
            image = resize_shortest_edge(image, resize_target)
            image = tf.image.random_crop(image, [target_h, target_w, 3])

        else:
            resize_target = max(target_h, target_w)
            image = resize_shortest_edge(image, resize_target)
            image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)

        image = tf.math.divide(tf.cast(image, tf.float32), 255.0)

        text = tf.io.read_file(text_path)
        text = tf.strings.strip(text)
        return (image, text), label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # type: ignore[call-arg]
            tf.TensorSpec(shape=(), dtype=tf.string),  # type: ignore[call-arg]
            tf.TensorSpec(shape=(), dtype=tf.int32),  # type: ignore[call-arg]
        ),
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


class DataloaderVisualizer:
    """Tkinter app to visualize dataloader outputs."""

    def __init__(
        self,
        batches: dict[str, list[tuple[np.ndarray, list[str], np.ndarray]]],
        batch_size: int = 8,
    ):
        """Initialize the visualizer.

        Args:
            batches: Dict mapping split name to list of (images, texts, labels) tuples.
                     images: np.ndarray of shape (batch, H, W, 3) with values 0-1
                     texts: list of text strings
                     labels: np.ndarray of shape (batch,) with label indices
            batch_size: Number of images per batch.
        """
        import tkinter as tk

        self.tk = tk
        self.batches = batches
        self.batch_size = batch_size
        self.current_split = "train"
        self.current_batch_idx = 0

        self.root = tk.Tk()
        self.root.title("Dataloader Visualizer")
        self.root.configure(bg="#1e1e1e")

        self._setup_ui()
        self._update_display()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        from tkinter import ttk

        self.ttk = ttk
        tk = self.tk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6)
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TFrame", background="#1e1e1e")

        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X)

        ttk.Label(control_frame, text="Split:").pack(side=tk.LEFT, padx=5)
        self.split_var = tk.StringVar(value="train")
        split_combo = ttk.Combobox(
            control_frame,
            textvariable=self.split_var,
            values=["train", "val", "test"],
            state="readonly",
            width=10,
        )
        split_combo.pack(side=tk.LEFT, padx=5)
        split_combo.bind("<<ComboboxSelected>>", self._on_split_change)

        ttk.Button(control_frame, text="<< Prev", command=self._prev_batch).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Next >>", command=self._next_batch).pack(
            side=tk.LEFT, padx=5
        )

        self.batch_label = ttk.Label(control_frame, text="Batch 1/1")
        self.batch_label.pack(side=tk.LEFT, padx=20)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#2d2d2d", highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        scrollbar_x = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set
        )

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner_frame, anchor="nw"
        )

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.root.bind("<Left>", lambda e: self._prev_batch())
        self.root.bind("<Right>", lambda e: self._next_batch())

    def _on_frame_configure(self, event) -> None:
        """Update scroll region when inner frame changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        """Update inner frame width when canvas resizes."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_split_change(self, event) -> None:
        """Handle split selection change."""
        self.current_split = self.split_var.get()
        self.current_batch_idx = 0
        self._update_display()

    def _prev_batch(self) -> None:
        """Go to previous batch."""
        if self.current_batch_idx > 0:
            self.current_batch_idx -= 1
            self._update_display()

    def _next_batch(self) -> None:
        """Go to next batch."""
        max_idx = len(self.batches.get(self.current_split, [])) - 1
        if self.current_batch_idx < max_idx:
            self.current_batch_idx += 1
            self._update_display()

    def _update_display(self) -> None:
        """Update the display with current batch."""
        import io

        from PIL import Image

        from src.pipelines.label_config import LABEL_CONFIG

        tk = self.tk
        ttk = self.ttk

        for widget in self.inner_frame.winfo_children():
            widget.destroy()

        split_batches = self.batches.get(self.current_split, [])
        total_batches = len(split_batches)

        self.batch_label.configure(
            text=f"Batch {self.current_batch_idx + 1}/{total_batches}"
        )

        if not split_batches:
            ttk.Label(
                self.inner_frame, text="No batches available", font=("Arial", 14)
            ).pack(pady=50)
            return

        images, texts, labels = split_batches[self.current_batch_idx]
        batch_size = len(images)

        cols = min(4, batch_size)
        self.photo_refs = []

        for idx in range(batch_size):
            row = idx // cols
            col = idx % cols

            cell_frame = ttk.Frame(self.inner_frame)
            cell_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            img_array = images[idx]
            img_array = (img_array * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)

            display_size = 200
            pil_img = pil_img.resize(
                (display_size, display_size), Image.Resampling.LANCZOS
            )

            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            photo = tk.PhotoImage(data=buffer.getvalue())
            self.photo_refs.append(photo)

            img_label = ttk.Label(cell_frame, image=photo)
            img_label.pack()

            label_idx = int(labels[idx])
            label_name = LABEL_CONFIG.get(label_idx, {}).get(
                "name", f"Unknown ({label_idx})"
            )

            info_label = ttk.Label(
                cell_frame,
                text=f"Label: {label_idx} ({label_name})",
                font=("Arial", 10, "bold"),
                wraplength=200,
            )
            info_label.pack(pady=(5, 0))

            text_content = texts[idx] if idx < len(texts) else ""

            text_label = ttk.Label(
                cell_frame,
                text=text_content,
                font=("Arial", 8),
                wraplength=200,
                foreground="#aaaaaa",
            )
            text_label.pack(pady=(2, 0))

        for col in range(cols):
            self.inner_frame.columnconfigure(col, weight=1)

    def run(self) -> None:
        """Run the visualizer app."""
        self.root.geometry("1200x900")
        self.root.mainloop()


def fetch_dataloader_batches(
    num_batches: int = 5,
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, list[tuple[np.ndarray, list[str], np.ndarray]]]:
    """Fetch batches from all dataloaders and convert to numpy arrays.

    Args:
        num_batches: Number of batches to fetch per split.
        batch_size: Batch size for dataloaders.
        image_size: Image size for dataloaders.

    Returns:
        Dict mapping split name to list of (images, texts, labels) tuples.
    """
    train_ds, val_ds, test_ds = get_dataloaders(
        batch_size=batch_size, image_size=image_size
    )

    result: dict[str, list[tuple[np.ndarray, list[str], np.ndarray]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    for split_name, dataset in datasets.items():
        for i, ((images, texts), labels) in enumerate(dataset.take(num_batches)):
            images_np = images.numpy()
            texts_list = [t.numpy().decode("utf-8") for t in texts]
            labels_np = labels.numpy()
            result[split_name].append((images_np, texts_list, labels_np))

    return result


def run_visualizer(
    batches: dict[str, list[tuple[np.ndarray, list[str], np.ndarray]]],
    batch_size: int = 4,
) -> None:
    """Run the dataloader visualizer app.

    Args:
        batches: Pre-fetched batches from fetch_dataloader_batches.
        batch_size: Batch size used.
    """
    app = DataloaderVisualizer(batches, batch_size=batch_size)
    app.run()
