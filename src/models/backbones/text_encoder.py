from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encodes texts into embeddings using batched processing."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
