import jax.numpy as jnp
from flax import nnx

from src.models.fusion import FiLMBlock


class Classifier2FiLM(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        conditioning_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.film1 = FiLMBlock(input_dim, conditioning_dim, rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.film2 = FiLMBlock(hidden_dim, conditioning_dim, rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, features: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            features: Input features of shape (batch, input_dim).
            conditioning: Conditioning signal of shape (batch, conditioning_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.film1(features, conditioning)
        x = nnx.relu(self.linear1(x))
        x = self.film2(x, conditioning)
        x = self.linear2(x)
        return x


class Classifier4FiLM(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        conditioning_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.film1 = FiLMBlock(input_dim, conditioning_dim, rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.film2 = FiLMBlock(hidden_dim, conditioning_dim, rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.film3 = FiLMBlock(hidden_dim, conditioning_dim, rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.film4 = FiLMBlock(hidden_dim, conditioning_dim, rngs)
        self.linear4 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, features: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            features: Input features of shape (batch, input_dim).
            conditioning: Conditioning signal of shape (batch, conditioning_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.film1(features, conditioning)
        x = nnx.relu(self.linear1(x))
        x = self.film2(x, conditioning)
        x = nnx.relu(self.linear2(x))
        x = self.film3(x, conditioning)
        x = nnx.relu(self.linear3(x))
        x = self.film4(x, conditioning)
        x = self.linear4(x)
        return x


class ClassifierDepth2(nnx.Module):
    def __init__(
        self,
        image_enc_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.linear1 = nnx.Linear(image_enc_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            features: Input features of shape (batch, input_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = nnx.relu(self.linear1(features))
        x = self.linear2(x)
        return x


class ClassifierDepth4(nnx.Module):
    def __init__(
        self,
        image_enc_dim: int,
        num_classes: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 1024,
    ):
        self.linear1 = nnx.Linear(image_enc_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            features: Input features of shape (batch, input_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = nnx.relu(self.linear1(features))
        x = nnx.relu(self.linear2(x))
        x = nnx.relu(self.linear3(x))
        x = self.linear4(x)
        return x
