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

        self.aux_image_head = nnx.Linear(image_dim, num_classes, rngs=rngs)
        self.aux_text_head = nnx.Linear(text_dim, num_classes, rngs=rngs)

    def __call__(
        self, image_features: jnp.ndarray, text_features: jnp.ndarray
    ) -> jnp.ndarray:
        x = self.fusion1(image_features, text_features)
        x = nnx.relu(self.linear1(x))
        x = self.fusion2(x, text_features)
        x = self.linear2(x)
        return x

    def forward_image_only(self, image_features: jnp.ndarray) -> jnp.ndarray:
        return self.aux_image_head(image_features)

    def forward_text_only(self, text_features: jnp.ndarray) -> jnp.ndarray:
        return self.aux_text_head(text_features)


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
        x = self.ln1(features)
        x = nnx.relu(self.linear1(x))
        x = self.ln2(x)
        x = self.linear2(x)
        return x
