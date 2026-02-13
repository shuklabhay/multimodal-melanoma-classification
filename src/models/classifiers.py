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
