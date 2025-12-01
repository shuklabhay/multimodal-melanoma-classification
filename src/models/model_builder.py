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
        num_classifier_blocks = config["num_classifier_blocks"]

        classifier_input_dim: int

        if image_enc == "resnet50" and text_enc is None:
            self.image_encoder = ResNet50Backbone()
            classifier_input_dim = ENCODER_OUTPUT_DIMS["resnet50"]
        elif text_enc == "mpnet-base" and image_enc is None:
            self.image_encoder = None
            classifier_input_dim = ENCODER_OUTPUT_DIMS["mpnet-base"]
        else:
            raise NotImplementedError("Fusion or invalid config combination")

        if num_classifier_blocks == 2:
            self.classifier = ClassifierDepth2(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                rngs=rngs,
            )
        elif num_classifier_blocks == 4:
            self.classifier = ClassifierDepth4(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported num_classifier_blocks: {num_classifier_blocks}")

    def __call__(
        self,
        images: jnp.ndarray | None,
        embeddings: jnp.ndarray | None,
        train: bool = False,
    ) -> jnp.ndarray:
        if self.config["image_enc"] is not None and self.config["text_enc"] is None:
            if images is None:
                raise ValueError("Model configured for images but no images provided")
            features = self.image_encoder(images, train=train)
            return self.classifier(features)

        elif self.config["text_enc"] is not None and self.config["image_enc"] is None:
            if embeddings is None:
                raise ValueError("Model configured for text but no embeddings provided")
            return self.classifier(embeddings)

        else:
            raise NotImplementedError("Fusion path not implemented yet")


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
