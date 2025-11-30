from typing import TypedDict

import jax.numpy as jnp
import yaml
from flax import nnx

from src.models.backbones import ResNet50Backbone
from src.models.classifiers import ClassifierDepth2, ClassifierDepth4

ENCODER_OUTPUT_DIMS = {
    "resnet50": 2048,
    "mpnet-base": 768,
}


class ModelConfig(TypedDict):
    image_enc: str | None
    text_enc: str | None
    fusion_type: str
    num_classifier_blocks: int


class DynamicModel(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        self.config = config

        image_enc = config["image_enc"]
        text_enc = config["text_enc"]
        fusion_type = config["fusion_type"]
        num_classifier_blocks = config["num_classifier_blocks"]

        image_enc_dim: int
        if image_enc == "resnet50":
            self.image_encoder = ResNet50Backbone()
            image_enc_dim = ENCODER_OUTPUT_DIMS["resnet50"]

        if num_classifier_blocks == 2:
            self.classifier = ClassifierDepth2(
                image_enc_dim=image_enc_dim,
                num_classes=num_classes,
                rngs=rngs,
            )

    def __call__(self, images: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        if self.image_encoder is not None:
            features = self.image_encoder(images, train=train)
        else:
            raise NotImplementedError("Text-only model not yet implemented")

        logits = self.classifier(features)
        return logits


def load_config(config_path: str) -> ModelConfig:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The parsed config dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(
    config: ModelConfig,
    num_classes: int = 8,
    seed: int = 0,
) -> DynamicModel:
    """Build a model from config.

    Args:
        config: Configuration dictionary with encoder and model settings.
        num_classes: The number of output classes for the classifier.
        seed: Random seed for model initialization.

    Returns:
        The initialized model.
    """
    rngs = nnx.Rngs(seed)
    return DynamicModel(config=config, num_classes=num_classes, rngs=rngs)
