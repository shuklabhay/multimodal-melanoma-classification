import jax.numpy as jnp
from flax import nnx


class FiLMBlock(nnx.Module):
    def __init__(self, feature_dim: int, conditioning_dim: int, rngs: nnx.Rngs):
        self.gamma_proj = nnx.Linear(conditioning_dim, feature_dim, rngs=rngs)
        self.beta_proj = nnx.Linear(conditioning_dim, feature_dim, rngs=rngs)

    def __call__(
        self, visual_features: jnp.ndarray, conditioning: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply FiLM modulation.

        Args:
            visual_features: Visual features of shape (batch, feature_dim).
            conditioning: Conditioning signal of shape (batch, conditioning_dim).

        Returns:
            Modulated features of shape (batch, feature_dim).
        """
        gamma = self.gamma_proj(conditioning)
        beta = self.beta_proj(conditioning)
        return gamma * visual_features + beta
