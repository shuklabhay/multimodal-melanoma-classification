from typing import Literal, TypedDict, cast

import jax.numpy as jnp
import yaml
from flax import nnx

from src.models.backbones import (
    ImageEncoderName,
    build_image_backbone,
    get_image_encoder_output_dim,
)
from src.models.classifiers import ConcatMLPClassifier, CrossAttentionMLPClassifier

TEXT_ENCODER_OUTPUT_DIMS = {
    "mpnet-base": 768,
}

ModelType = Literal["image", "fusion", "text"]
LegacyRunMode = Literal["image_only", "multimodal", "text_only"]


class ModelConfig(TypedDict, total=False):
    type: ModelType
    run_mode: LegacyRunMode
    image_enc: ImageEncoderName | None
    text_enc: str | None
    text_feature_dim: int
    fusion_type: str
    fusion_attn_dim: int
    fusion_num_heads: int
    mlp_hidden_dim: int
    modality_dropout_normal: float
    modality_dropout_zero_text: float
    modality_dropout_swap_text: float


RUN_MODE_TO_MODEL_TYPE: dict[LegacyRunMode, ModelType] = {
    "image_only": "image",
    "multimodal": "fusion",
    "text_only": "text",
}


def get_model_type(config: ModelConfig) -> ModelType:
    """Resolve training/inference model type from config.

    Prefers `type` and supports legacy `run_mode` for backward compatibility.
    """
    model_type = config.get("type")
    if model_type is not None:
        if model_type not in {"image", "fusion", "text"}:
            raise ValueError("Config `type` must be one of: image, fusion, text")
        return cast(ModelType, model_type)

    run_mode = config.get("run_mode")
    if run_mode in RUN_MODE_TO_MODEL_TYPE:
        return RUN_MODE_TO_MODEL_TYPE[cast(LegacyRunMode, run_mode)]

    raise ValueError(
        "Config must set `type` to one of: image, fusion, text "
        "(or legacy `run_mode`: image_only, multimodal, text_only)"
    )


class DynamicModel(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        self.config = config

        model_type = get_model_type(config)

        image_enc = config.get("image_enc")
        text_enc = config.get("text_enc")

        if model_type == "image":
            if image_enc is None:
                raise ValueError("type=image requires image_enc")
            text_enc = None
        elif model_type == "fusion":
            if image_enc is None:
                raise ValueError("type=fusion requires image_enc")
            if text_enc is None:
                raise ValueError("type=fusion requires text_enc")
        elif model_type == "text":
            image_enc = None
            if text_enc is None:
                raise ValueError("type=text requires text_enc")

        if image_enc is None and text_enc is None:
            raise ValueError("At least one encoder must be enabled")

        text_feature_dim = config.get("text_feature_dim")
        if text_enc is not None and text_feature_dim is None:
            if text_enc not in TEXT_ENCODER_OUTPUT_DIMS:
                raise ValueError(f"Unsupported text encoder: {text_enc}")

        self.model_type = model_type
        self.active_image_enc = image_enc
        self.active_text_enc = text_enc

        image_dim = 0
        if image_enc is not None:
            image_encoder_name = cast(ImageEncoderName, image_enc)
            self.image_encoder = build_image_backbone(image_encoder_name)
            image_dim = get_image_encoder_output_dim(image_encoder_name)
        else:
            self.image_encoder = nnx.data(None)

        text_dim = 0
        if text_enc is not None:
            text_dim = (
                text_feature_dim
                if text_feature_dim is not None
                else TEXT_ENCODER_OUTPUT_DIMS[text_enc]
            )

        fusion_type = config.get("fusion_type", "concat")
        if model_type == "fusion" and fusion_type == "cross_attention":
            self.classifier = CrossAttentionMLPClassifier(
                image_dim=image_dim,
                text_dim=text_dim,
                num_classes=num_classes,
                hidden_dim=config.get("mlp_hidden_dim", 1024),
                attn_dim=config.get("fusion_attn_dim", 512),
                num_heads=config.get("fusion_num_heads", 8),
                rngs=rngs,
            )
        elif fusion_type == "concat":
            self.classifier = ConcatMLPClassifier(
                image_dim=image_dim,
                text_dim=text_dim,
                num_classes=num_classes,
                hidden_dim=config.get("mlp_hidden_dim", 1024),
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def classify_from_features(
        self,
        image_features: jnp.ndarray | None,
        text_features: jnp.ndarray | None,
    ) -> jnp.ndarray:
        return self.classifier(image_features, text_features)

    def __call__(
        self,
        images: jnp.ndarray | None,
        embeddings: jnp.ndarray | None,
        train: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through configured encoders and concat classifier.

        Args:
            images: Input images of shape (batch, height, width, channels), or None if text-only.
            embeddings: Text embeddings of shape (batch, embed_dim), or None if image-only.
            train: Whether in training mode.

        Returns:
            Logits of shape (batch, num_classes).
        """
        image_features = None
        if self.active_image_enc is not None:
            if images is None:
                raise ValueError("Model configured for images but no images provided")
            if self.image_encoder is None:
                raise RuntimeError("Image encoder is missing")
            image_features = self.image_encoder(images, train=train)

        text_features = None
        if self.active_text_enc is not None:
            if embeddings is None:
                raise ValueError(
                    "Model configured for text but no text embeddings provided"
                )
            text_features = embeddings

        return self.classify_from_features(image_features, text_features)


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
