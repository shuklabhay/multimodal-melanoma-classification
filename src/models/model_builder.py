from typing import TypedDict

import jax.numpy as jnp
import yaml
from flax import nnx

from src.models.backbones import ResNet50Backbone
from src.models.classifiers import (
    ClassifierDepth2,
    FiLMClassifierDepth2,
)

ENCODER_OUTPUT_DIMS = {
    "resnet50": 2048,
    "mpnet-base": 768,
}


class ModelConfig(TypedDict, total=False):
    image_enc: str | None
    text_enc: str | None
    fusion_type: str
    classifier: str
    modality_dropout_normal: float
    modality_dropout_zero_text: float
    modality_dropout_swap_text: float
    aux_image_loss_weight: float
    aux_text_loss_weight: float


class DynamicModel(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        self.config = config

        image_enc = config.get("image_enc")
        text_enc = config.get("text_enc")
        fusion_type = config.get("fusion_type", "none")
        classifier_name = config.get("classifier")

        if image_enc == "resnet50" and text_enc is None:
            self.image_encoder = ResNet50Backbone()
            if classifier_name == "ClassifierDepth2":
                self.classifier = ClassifierDepth2(
                    input_dim=ENCODER_OUTPUT_DIMS["resnet50"],
                    num_classes=num_classes,
                    rngs=rngs,
                )
            else:
                raise ValueError(f"Unsupported classifier: {classifier_name}")

        elif text_enc == "mpnet-base" and image_enc is None:
            if classifier_name == "ClassifierDepth2":
                self.classifier = ClassifierDepth2(
                    input_dim=ENCODER_OUTPUT_DIMS["mpnet-base"],
                    num_classes=num_classes,
                    rngs=rngs,
                )
            else:
                raise ValueError(f"Unsupported classifier: {classifier_name}")

        elif (
            image_enc == "resnet50"
            and text_enc == "mpnet-base"
            and fusion_type == "film"
        ):
            self.image_encoder = ResNet50Backbone()
            if classifier_name == "FiLMClassifierDepth2":
                self.classifier = FiLMClassifierDepth2(
                    image_dim=ENCODER_OUTPUT_DIMS["resnet50"],
                    text_dim=ENCODER_OUTPUT_DIMS["mpnet-base"],
                    num_classes=num_classes,
                    rngs=rngs,
                )
            else:
                raise ValueError(f"Unsupported classifier for fusion: {classifier_name}")

        else:
            raise NotImplementedError(
                f"Unsupported config: image_enc={image_enc}, text_enc={text_enc}, fusion_type={fusion_type}"
            )

    def __call__(
        self,
        images: jnp.ndarray | None,
        embeddings: jnp.ndarray | None,
        train: bool = False,
        return_aux: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        image_enc = self.config.get("image_enc")
        text_enc = self.config.get("text_enc")

        if image_enc is not None and text_enc is None:
            if images is None:
                raise ValueError("Model configured for images but no images provided")
            assert isinstance(self.classifier, ClassifierDepth2)
            features = self.image_encoder(images, train=train)
            logits = self.classifier(features)
            if return_aux:
                return logits, None, None
            return logits

        elif text_enc is not None and image_enc is None:
            if embeddings is None:
                raise ValueError("Model configured for text but no embeddings provided")
            assert isinstance(self.classifier, ClassifierDepth2)
            logits = self.classifier(embeddings)
            if return_aux:
                return logits, None, None
            return logits

        elif image_enc is not None and text_enc is not None:
            if images is None:
                raise ValueError("Fusion model requires images")
            if embeddings is None:
                raise ValueError("Fusion model requires text embeddings")
            assert isinstance(self.classifier, FiLMClassifierDepth2)
            image_features = self.image_encoder(images, train=train)
            logits = self.classifier(image_features, embeddings)
            if return_aux:
                aux_image_logits = self.classifier.forward_image_only(image_features)
                aux_text_logits = self.classifier.forward_text_only(embeddings)
                return logits, aux_image_logits, aux_text_logits
            return logits

        else:
            raise NotImplementedError("Invalid config")


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
