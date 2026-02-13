from __future__ import annotations

from typing import Literal, TypedDict, cast

import jax.nn as jnn
import jax.numpy as jnp
import torch

from src.models.backbones.common import StateDictNCHWBackbone


class MobileNetV3BlockSpec(TypedDict):
    input_channels: int
    kernel: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    activation: Literal["RE", "HS"]
    stride: int
    dilation: int


MOBILENET_V3_LARGE_IMAGENET1K_V2_URL = (
    "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"
)

MOBILENET_V3_LARGE_BLOCK_SPECS: list[MobileNetV3BlockSpec] = [
    {
        "input_channels": 16,
        "kernel": 3,
        "expanded_channels": 16,
        "out_channels": 16,
        "use_se": False,
        "activation": "RE",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 16,
        "kernel": 3,
        "expanded_channels": 64,
        "out_channels": 24,
        "use_se": False,
        "activation": "RE",
        "stride": 2,
        "dilation": 1,
    },
    {
        "input_channels": 24,
        "kernel": 3,
        "expanded_channels": 72,
        "out_channels": 24,
        "use_se": False,
        "activation": "RE",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 24,
        "kernel": 5,
        "expanded_channels": 72,
        "out_channels": 40,
        "use_se": True,
        "activation": "RE",
        "stride": 2,
        "dilation": 1,
    },
    {
        "input_channels": 40,
        "kernel": 5,
        "expanded_channels": 120,
        "out_channels": 40,
        "use_se": True,
        "activation": "RE",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 40,
        "kernel": 5,
        "expanded_channels": 120,
        "out_channels": 40,
        "use_se": True,
        "activation": "RE",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 40,
        "kernel": 3,
        "expanded_channels": 240,
        "out_channels": 80,
        "use_se": False,
        "activation": "HS",
        "stride": 2,
        "dilation": 1,
    },
    {
        "input_channels": 80,
        "kernel": 3,
        "expanded_channels": 200,
        "out_channels": 80,
        "use_se": False,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 80,
        "kernel": 3,
        "expanded_channels": 184,
        "out_channels": 80,
        "use_se": False,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 80,
        "kernel": 3,
        "expanded_channels": 184,
        "out_channels": 80,
        "use_se": False,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 80,
        "kernel": 3,
        "expanded_channels": 480,
        "out_channels": 112,
        "use_se": True,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 112,
        "kernel": 3,
        "expanded_channels": 672,
        "out_channels": 112,
        "use_se": True,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 112,
        "kernel": 5,
        "expanded_channels": 672,
        "out_channels": 160,
        "use_se": True,
        "activation": "HS",
        "stride": 2,
        "dilation": 1,
    },
    {
        "input_channels": 160,
        "kernel": 5,
        "expanded_channels": 960,
        "out_channels": 160,
        "use_se": True,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
    {
        "input_channels": 160,
        "kernel": 5,
        "expanded_channels": 960,
        "out_channels": 160,
        "use_se": True,
        "activation": "HS",
        "stride": 1,
        "dilation": 1,
    },
]


class MobileNetV3LargeBackbone(StateDictNCHWBackbone):
    def __init__(self):
        state_dict = cast(
            dict[str, torch.Tensor],
            torch.hub.load_state_dict_from_url(
                MOBILENET_V3_LARGE_IMAGENET1K_V2_URL,
                map_location="cpu",
                progress=False,
                check_hash=True,
            ),
        )
        self.block_specs = MOBILENET_V3_LARGE_BLOCK_SPECS
        super().__init__(state_dict, include_prefixes=("features.",))

    @staticmethod
    def _hard_swish(x: jnp.ndarray) -> jnp.ndarray:
        return x * jnp.clip(x + 3.0, 0.0, 6.0) / 6.0

    @staticmethod
    def _hard_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x + 3.0, 0.0, 6.0) / 6.0

    def _activation(self, x: jnp.ndarray, activation: Literal["RE", "HS"]) -> jnp.ndarray:
        if activation == "HS":
            return self._hard_swish(x)
        return jnn.relu(x)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x_nchw = jnp.transpose(x, (0, 3, 1, 2))

        x_nchw = self._conv2d_nchw(
            x_nchw,
            self._get_param("features.0.0.weight"),
            bias=None,
            stride=2,
            padding=1,
        )
        x_nchw = self._batch_norm_nchw(
            x_nchw,
            "features.0.1",
            eps=1e-3,
            momentum=0.01,
            train=train,
        )
        x_nchw = self._hard_swish(x_nchw)

        for block_idx, spec in enumerate(self.block_specs, start=1):
            block_prefix = f"features.{block_idx}.block"
            residual = x_nchw
            inner_idx = 0

            activation = cast(Literal["RE", "HS"], spec["activation"])
            in_channels = int(spec["input_channels"])
            expanded_channels = int(spec["expanded_channels"])
            out_channels = int(spec["out_channels"])
            kernel = int(spec["kernel"])
            stride = int(spec["stride"])
            dilation = int(spec["dilation"])
            use_se = bool(spec["use_se"])

            if expanded_channels != in_channels:
                expand_prefix = f"{block_prefix}.{inner_idx}"
                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    self._get_param(f"{expand_prefix}.0.weight"),
                    bias=None,
                    stride=1,
                    padding=0,
                )
                x_nchw = self._batch_norm_nchw(
                    x_nchw,
                    f"{expand_prefix}.1",
                    eps=1e-3,
                    momentum=0.01,
                    train=train,
                )
                x_nchw = self._activation(x_nchw, activation)
                inner_idx += 1

            depthwise_prefix = f"{block_prefix}.{inner_idx}"
            depthwise_stride = 1 if dilation > 1 else stride
            depthwise_padding = ((kernel - 1) // 2) * dilation
            depthwise_weight = self._get_param(f"{depthwise_prefix}.0.weight")
            x_nchw = self._conv2d_nchw(
                x_nchw,
                depthwise_weight,
                bias=None,
                stride=depthwise_stride,
                padding=depthwise_padding,
                dilation=dilation,
                groups=int(depthwise_weight.shape[0]),
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{depthwise_prefix}.1",
                eps=1e-3,
                momentum=0.01,
                train=train,
            )
            x_nchw = self._activation(x_nchw, activation)
            inner_idx += 1

            if use_se:
                se_prefix = f"{block_prefix}.{inner_idx}"
                se = jnp.mean(x_nchw, axis=(2, 3), keepdims=True)
                se = self._conv2d_nchw(
                    se,
                    self._get_param(f"{se_prefix}.fc1.weight"),
                    self._get_param(f"{se_prefix}.fc1.bias"),
                    stride=1,
                    padding=0,
                )
                se = jnn.relu(se)
                se = self._conv2d_nchw(
                    se,
                    self._get_param(f"{se_prefix}.fc2.weight"),
                    self._get_param(f"{se_prefix}.fc2.bias"),
                    stride=1,
                    padding=0,
                )
                se = self._hard_sigmoid(se)
                x_nchw = x_nchw * se
                inner_idx += 1

            project_prefix = f"{block_prefix}.{inner_idx}"
            x_nchw = self._conv2d_nchw(
                x_nchw,
                self._get_param(f"{project_prefix}.0.weight"),
                bias=None,
                stride=1,
                padding=0,
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{project_prefix}.1",
                eps=1e-3,
                momentum=0.01,
                train=train,
            )

            if stride == 1 and in_channels == out_channels:
                x_nchw = x_nchw + residual

        x_nchw = self._conv2d_nchw(
            x_nchw,
            self._get_param("features.16.0.weight"),
            bias=None,
            stride=1,
            padding=0,
        )
        x_nchw = self._batch_norm_nchw(
            x_nchw,
            "features.16.1",
            eps=1e-3,
            momentum=0.01,
            train=train,
        )
        x_nchw = self._hard_swish(x_nchw)

        return jnp.mean(x_nchw, axis=(2, 3))
