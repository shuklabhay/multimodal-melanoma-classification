import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import aiofiles
import numpy as np
from tqdm import tqdm

from src.models.backbones import SentenceTransformerEncoder


def writer_worker(write_queue: Queue, pbar: tqdm) -> None:
    """Worker thread that writes embeddings to disk.

    Args:
        write_queue: Queue of (embedding, path) tuples to write. None signals shutdown.
        pbar: Progress bar to update after each write.
    """
    while True:
        item = write_queue.get()
        if item is None:
            break
        embedding, path = item
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embedding=embedding)
        pbar.update(1)
        write_queue.task_done()


async def load_texts_continuous(
    items: list[dict], text_queue: asyncio.Queue, batch_size: int
) -> None:
    """Continuously loads text files and puts batches onto queue.

    Args:
        items: List of manifest items with text_path keys.
        text_queue: Async queue to put (batch_items, texts) tuples.
        batch_size: Number of items per batch.
    """

    async def load_single(item: dict) -> str:
        async with aiofiles.open(item["text_path"]) as f:
            content = await f.read()
            return content.strip()

    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        texts = await asyncio.gather(*[load_single(item) for item in batch_items])
        await text_queue.put((batch_items, texts))

    await text_queue.put(None)


async def generate_embeddings_for_split(
    narrative_manifest_path: str,
    output_dir: str,
    split_name: str,
    encoder: SentenceTransformerEncoder,
    batch_size: int = 128,
) -> list[dict]:
    """Generates embeddings using concurrent load/encode/write pipeline.

    Args:
        narrative_manifest_path: Path to narrative manifest JSONL file.
        output_dir: Base output directory for embeddings.
        split_name: Name of the split (train/val/test).
        encoder: SentenceTransformerEncoder instance.
        batch_size: Batch size for encoding.

    Returns:
        List of manifest entries with text_path and embedding_path.
    """
    async with aiofiles.open(narrative_manifest_path) as f:
        content = await f.read()
        narrative_items = [json.loads(line) for line in content.strip().split("\n")]

    parent_dir = "train" if split_name in ["train", "val"] else "test"
    prefix = Path(narrative_manifest_path).stem.split("_narratives_")[0]
    embedding_dir = Path(output_dir) / parent_dir / f"{prefix}_text_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    for item in narrative_items:
        filename = Path(item["text_path"]).stem
        item["embedding_path"] = str(embedding_dir / f"{filename}.npz")

    print(f"\nProcessing {split_name} split ({len(narrative_items)} items)...")

    pbar = tqdm(total=len(narrative_items), desc=f"{split_name}")

    write_queue: Queue = Queue(maxsize=batch_size * 4)
    writers = [
        Thread(target=writer_worker, args=(write_queue, pbar), daemon=True)
        for _ in range(8)
    ]
    for w in writers:
        w.start()

    text_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    loader_task = asyncio.create_task(
        load_texts_continuous(narrative_items, text_queue, batch_size)
    )

    manifest_entries = []
    loop = asyncio.get_event_loop()

    while True:
        batch_data = await text_queue.get()
        if batch_data is None:
            break

        batch_items, texts = batch_data

        embeddings = await loop.run_in_executor(
            None,
            lambda t=texts: encoder.encode_batch(
                t, batch_size=batch_size, show_progress=False
            ),
        )

        for item, emb in zip(batch_items, embeddings):
            write_queue.put((emb, item["embedding_path"]))
            manifest_entries.append(
                {
                    "text_path": item["text_path"],
                    "embedding_path": item["embedding_path"],
                    "label": item["label"],
                }
            )

    await loader_task

    write_queue.join()
    for _ in writers:
        write_queue.put(None)
    for w in writers:
        w.join()

    pbar.close()
    return manifest_entries


def create_narrative_embeddings(
    narrative_manifest_dir: str = "/root/assets/manifests",
    output_dir: str = "/data/raw",
    batch_size: int = 128,
) -> None:
    """Creates narrative embeddings and embedding manifests for all splits.

    Args:
        narrative_manifest_dir: Directory containing narrative manifests.
        output_dir: Base output directory (will create {split}/embeddings subdirs).
        batch_size: Batch size for embedding generation.
    """
    print("Starting embedding creation...")
    encoder = SentenceTransformerEncoder()
    manifest_entries_all = {}

    for split in ["train", "val", "test"]:
        narrative_manifest_path = (
            Path(narrative_manifest_dir) / f"narrative_split_{split}.jsonl"
        )

        manifest_entries = asyncio.run(
            generate_embeddings_for_split(
                str(narrative_manifest_path),
                output_dir,
                split,
                encoder,
                batch_size,
            )
        )
        manifest_entries_all[split] = manifest_entries

    buffer_dir = Path("/data/buffer")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    for split, entries in manifest_entries_all.items():
        manifest_path = buffer_dir / f"embedding_split_{split}.jsonl"
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"✓ Created embedding manifest: {manifest_path.name}")

    print("\n✓ All embeddings and manifests created successfully!")
