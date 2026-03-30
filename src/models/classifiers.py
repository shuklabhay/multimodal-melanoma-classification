import math

import jax
import jax.numpy as jnp
from flax import nnx


class ConcatMLPClassifier(nnx.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        input_dim = image_dim + text_dim
        if input_dim <= 0:
            raise ValueError("ConcatMLPClassifier requires image and/or text features")

        self.ln1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(
        self,
        image_features: jnp.ndarray | None,
        text_features: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Forward pass through concat + 2-layer MLP.

        Args:
            image_features: Optional image features of shape (batch_size, image_dim).
            text_features: Optional text features of shape (batch_size, text_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        feature_parts = []
        if image_features is not None:
            feature_parts.append(image_features)
        if text_features is not None:
            feature_parts.append(text_features)

        if not feature_parts:
            raise ValueError("At least one modality must be provided to classifier")

        features = (
            feature_parts[0]
            if len(feature_parts) == 1
            else jnp.concatenate(feature_parts, axis=-1)
        )

        x = self.ln1(features)
        x = nnx.relu(self.linear1(x))
        x = self.ln2(x)
        x = self.linear2(x)
        return x


class CrossAttentionMLPClassifier(nnx.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
        attn_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        if image_dim <= 0 or text_dim <= 0:
            raise ValueError(
                "CrossAttentionMLPClassifier requires image and text features"
            )
        if attn_dim <= 0 or attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be positive and divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.image_proj = nnx.Linear(image_dim, attn_dim, rngs=rngs)
        self.text_proj = nnx.Linear(text_dim, attn_dim, rngs=rngs)
        self.token_norm = nnx.LayerNorm(attn_dim, rngs=rngs)
        self.q_proj = nnx.Linear(attn_dim, attn_dim, rngs=rngs)
        self.k_proj = nnx.Linear(attn_dim, attn_dim, rngs=rngs)
        self.v_proj = nnx.Linear(attn_dim, attn_dim, rngs=rngs)
        self.out_proj = nnx.Linear(attn_dim, attn_dim, rngs=rngs)
        self.fusion_norm = nnx.LayerNorm(2 * attn_dim, rngs=rngs)
        self.linear1 = nnx.Linear(2 * attn_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def _multi_head_attention(self, tokens: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = tokens.shape
        q = self.q_proj(tokens).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(tokens).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(tokens).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attended = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return self.out_proj(attended)

    def __call__(
        self,
        image_features: jnp.ndarray | None,
        text_features: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Run two-token attention fusion before the shared MLP head."""
        if image_features is None or text_features is None:
            raise ValueError(
                "CrossAttentionMLPClassifier requires both image and text features"
            )

        tokens = jnp.concatenate(
            [
                self.image_proj(image_features)[:, None, :],
                self.text_proj(text_features)[:, None, :],
            ],
            axis=1,
        )
        tokens = tokens + self._multi_head_attention(self.token_norm(tokens))
        features = self.fusion_norm(tokens.reshape(tokens.shape[0], -1))
        x = nnx.relu(self.linear1(features))
        x = self.ln2(x)
        x = self.linear2(x)
        return x
