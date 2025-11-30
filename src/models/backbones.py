from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.core.frozen_dict import unfreeze
from sentence_transformers import SentenceTransformer
from transformers import FlaxResNetModel
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPoolingAndNoAttention,
)


def recursive_to_nnx(data, variable_type):
    if isinstance(data, dict):
        return nnx.Dict(
            {k: recursive_to_nnx(v, variable_type) for k, v in data.items()}
        )
    return variable_type(data)


def recursive_from_nnx(node):
    if isinstance(node, nnx.Dict):
        return {k: recursive_from_nnx(v) for k, v in node.items()}
    if hasattr(node, "value"):
        return node.value
    return node


class ResNet50Backbone(nnx.Module):
    def __init__(self):
        hf_model = cast(
            FlaxResNetModel, FlaxResNetModel.from_pretrained("microsoft/resnet-50")
        )
        self.backbone_module = hf_model.module

        all_params = unfreeze(hf_model.params)
        self.backbone_params = nnx.Dict(
            {
                "params": recursive_to_nnx(all_params.get("params", {}), nnx.Param),
                "batch_stats": recursive_to_nnx(
                    all_params.get("batch_stats", {}), nnx.BatchStat
                ),
            }
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass through ResNet50.

        Args:
            x: Input images of shape (batch, height, width, channels).
            train: Whether in training mode (unused, kept for API consistency).

        Returns:
            Embeddings of shape (batch, 2048).
        """
        legacy_vars = recursive_from_nnx(self.backbone_params)
        backbone_output = self.backbone_module.apply(
            legacy_vars, pixel_values=x, mutable=False
        )
        backbone_output = cast(
            FlaxBaseModelOutputWithPoolingAndNoAttention, backbone_output
        )

        embedding = backbone_output.pooler_output
        assert embedding is not None, "pooler_output should not be None"

        embedding = jnp.squeeze(embedding, axis=(-1, -2))
        return embedding


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initializes sentence transformer for text embedding generation.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def encode_batch(
        self, texts: list[str], batch_size: int = 128, show_progress: bool = True
    ) -> np.ndarray:
        """Encodes texts into embeddings using batched processing.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Numpy array of embeddings with shape (num_texts, embedding_dim).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
