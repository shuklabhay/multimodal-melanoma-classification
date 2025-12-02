import jax.numpy as jnp
from flax import nnx


class ResidualFiLMFusionBlock(nnx.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int | None = None,
    ):
        self.image_dim = image_dim
        self.text_dim = text_dim

        self.hidden: nnx.Linear | None = None
        if hidden_dim is not None:
            self.hidden = nnx.Linear(text_dim, hidden_dim, rngs=rngs)
            proj_input_dim = hidden_dim
        else:
            proj_input_dim = text_dim

        self.proj = nnx.Linear(proj_input_dim, 2 * image_dim, rngs=rngs)

        self.proj.kernel.value = jnp.zeros_like(self.proj.kernel.value)
        gamma_bias = jnp.ones(image_dim)
        beta_bias = jnp.zeros(image_dim)
        if self.proj.bias is not None:
            self.proj.bias.value = jnp.concatenate([gamma_bias, beta_bias])

    def __call__(
        self, image_features: jnp.ndarray, text_features: jnp.ndarray
    ) -> jnp.ndarray:
        x = text_features
        if self.hidden is not None:
            x = nnx.relu(self.hidden(x))

        gamma_beta = self.proj(x)
        gamma = gamma_beta[:, : self.image_dim]
        beta = gamma_beta[:, self.image_dim :]

        fused = image_features * gamma + beta
        return fused + image_features
