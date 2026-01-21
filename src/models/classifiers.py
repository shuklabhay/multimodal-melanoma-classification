import jax.numpy as jnp
from flax import nnx

from src.models.fusion import ResidualFiLMFusionBlock


class FiLMClassifierDepth2(nnx.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.fusion1 = ResidualFiLMFusionBlock(image_dim, text_dim, rngs)
        self.linear1 = nnx.Linear(image_dim, hidden_dim, rngs=rngs)
        self.fusion2 = ResidualFiLMFusionBlock(hidden_dim, text_dim, rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(
        self, image_features: jnp.ndarray, text_features: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through 2-layer FiLM classifier with image-text fusion at each layer.

        Args:
            image_features: Image embeddings of shape (batch_size, image_dim).
            text_features: Text embeddings of shape (batch_size, text_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.fusion1(image_features, text_features)
        x = nnx.relu(self.linear1(x))
        x = self.fusion2(x, text_features)
        x = self.linear2(x)
        return x


class ClassifierDepth2(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.ln1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through 2-layer classifier with layer normalization and ReLU activation.

        Args:
            features: Input features of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.ln1(features)
        x = nnx.relu(self.linear1(x))
        x = self.ln2(x)
        x = self.linear2(x)
        return x
