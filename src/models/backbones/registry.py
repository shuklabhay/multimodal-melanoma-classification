from __future__ import annotations

from typing import Literal

from flax import nnx

from src.models.backbones.convnext_tiny import ConvNeXtTinyBackbone
from src.models.backbones.efficientnet_b4 import EfficientNetB4Backbone
from src.models.backbones.mobilenet_v3_large import MobileNetV3LargeBackbone
from src.models.backbones.mobilenet_v4_conv_large import MobileNetV4ConvLargeBackbone
from src.models.backbones.resnet50 import ResNet50Backbone
from src.models.backbones.vit_small import ViTSmallBackbone

ImageEncoderName = Literal[
    "resnet50",
    "convnext_tiny",
    "efficientnet_b4",
    "vit_small",
    "mobilenet_v3_large",
    "mobilenet_v4_conv_large",
]

IMAGE_ENCODER_OUTPUT_DIMS: dict[ImageEncoderName, int] = {
    "resnet50": 2048,
    "convnext_tiny": 768,
    "efficientnet_b4": 1792,
    "vit_small": 384,
    "mobilenet_v3_large": 960,
    "mobilenet_v4_conv_large": 960,
}


def get_image_encoder_output_dim(image_encoder: ImageEncoderName) -> int:
    return IMAGE_ENCODER_OUTPUT_DIMS[image_encoder]


def build_image_backbone(image_encoder: ImageEncoderName) -> nnx.Module:
    if image_encoder == "resnet50":
        return ResNet50Backbone()
    if image_encoder == "convnext_tiny":
        return ConvNeXtTinyBackbone()
    if image_encoder == "efficientnet_b4":
        return EfficientNetB4Backbone()
    if image_encoder == "vit_small":
        return ViTSmallBackbone()
    if image_encoder == "mobilenet_v3_large":
        return MobileNetV3LargeBackbone()
    if image_encoder == "mobilenet_v4_conv_large":
        return MobileNetV4ConvLargeBackbone()

    raise ValueError(f"Unsupported image encoder: {image_encoder}")
