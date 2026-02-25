from __future__ import annotations

import math
from typing import cast

import jax.nn as jnn
import jax.numpy as jnp
from transformers import EfficientNetConfig, EfficientNetModel

from src.models.backbones.common import StateDictNCHWBackbone


class EfficientNetB4Backbone(StateDictNCHWBackbone):
    def __init__(self):
        model = EfficientNetModel.from_pretrained("google/efficientnet-b4")
        self.config = cast(EfficientNetConfig, model.config)
        self.block_specs = self._build_block_specs(self.config)
        super().__init__(model.state_dict())

    @staticmethod
    def _round_filters(config: EfficientNetConfig, num_channels: int) -> int:
        divisor = config.depth_divisor
        scaled = num_channels * config.width_coefficient
        new_dim = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
        if new_dim < 0.9 * scaled:
            new_dim += divisor
        return int(new_dim)

    @staticmethod
    def _round_repeats(config: EfficientNetConfig, repeats: int) -> int:
        return int(math.ceil(config.depth_coefficient * repeats))

    @staticmethod
    def _correct_pad(kernel_size: int, adjust: bool) -> tuple[int, int, int, int]:
        correct = kernel_size // 2
        if adjust:
            return (correct - 1, correct, correct - 1, correct)
        return (correct, correct, correct, correct)

    def _build_block_specs(
        self, config: EfficientNetConfig
    ) -> list[dict[str, int | bool]]:
        specs: list[dict[str, int | bool]] = []

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(
            self._round_repeats(config, n) for n in config.num_block_repeats
        )

        curr_block_num = 0
        for i in range(num_base_blocks):
            in_dim = self._round_filters(config, config.in_channels[i])
            out_dim = self._round_filters(config, config.out_channels[i])
            base_stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]
            repeats = self._round_repeats(config, config.num_block_repeats[i])

            for j in range(repeats):
                stride = 1 if j > 0 else base_stride
                in_ch = out_dim if j > 0 else in_dim
                id_skip = j == 0
                adjust_padding = curr_block_num not in config.depthwise_padding
                _drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                specs.append(
                    {
                        "in_channels": in_ch,
                        "out_channels": out_dim,
                        "stride": stride,
                        "kernel_size": kernel_size,
                        "expand_ratio": expand_ratio,
                        "id_skip": id_skip,
                        "adjust_padding": adjust_padding,
                    }
                )
                curr_block_num += 1

        return specs

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x_nchw = jnp.transpose(x, (0, 3, 1, 2))

        x_nchw = self._pad_nchw(x_nchw, left=0, right=1, top=0, bottom=1)
        x_nchw = self._conv2d_nchw(
            x_nchw,
            self._get_param("embeddings.convolution.weight"),
            bias=None,
            stride=2,
            padding=0,
        )
        x_nchw = self._batch_norm_nchw(
            x_nchw,
            "embeddings.batchnorm",
            eps=float(self.config.batch_norm_eps),
            momentum=float(self.config.batch_norm_momentum),
            train=train,
        )
        x_nchw = jnn.silu(x_nchw)

        for block_idx, spec in enumerate(self.block_specs):
            prefix = f"encoder.blocks.{block_idx}"
            residual = x_nchw

            expand_ratio = int(spec["expand_ratio"])
            stride = int(spec["stride"])
            kernel_size = int(spec["kernel_size"])
            adjust_padding = bool(spec["adjust_padding"])

            if expand_ratio != 1:
                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    self._get_param(f"{prefix}.expansion.expand_conv.weight"),
                    bias=None,
                    stride=1,
                    padding=0,
                )
                x_nchw = self._batch_norm_nchw(
                    x_nchw,
                    f"{prefix}.expansion.expand_bn",
                    eps=float(self.config.batch_norm_eps),
                    momentum=float(self.config.batch_norm_momentum),
                    train=train,
                )
                x_nchw = jnn.silu(x_nchw)

            if stride == 2:
                left, right, top, bottom = self._correct_pad(
                    kernel_size, adjust_padding
                )
                x_nchw = self._pad_nchw(
                    x_nchw,
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                )
                depthwise_padding = 0
            else:
                depthwise_padding = kernel_size // 2

            depthwise_weight = self._get_param(
                f"{prefix}.depthwise_conv.depthwise_conv.weight"
            )
            x_nchw = self._conv2d_nchw(
                x_nchw,
                depthwise_weight,
                bias=None,
                stride=stride,
                padding=depthwise_padding,
                groups=int(depthwise_weight.shape[0]),
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{prefix}.depthwise_conv.depthwise_norm",
                eps=float(self.config.batch_norm_eps),
                momentum=float(self.config.batch_norm_momentum),
                train=train,
            )
            x_nchw = jnn.silu(x_nchw)

            se = jnp.mean(x_nchw, axis=(2, 3), keepdims=True)
            se = self._conv2d_nchw(
                se,
                self._get_param(f"{prefix}.squeeze_excite.reduce.weight"),
                self._get_param(f"{prefix}.squeeze_excite.reduce.bias"),
                stride=1,
                padding=0,
            )
            se = jnn.silu(se)
            se = self._conv2d_nchw(
                se,
                self._get_param(f"{prefix}.squeeze_excite.expand.weight"),
                self._get_param(f"{prefix}.squeeze_excite.expand.bias"),
                stride=1,
                padding=0,
            )
            se = jnn.sigmoid(se)
            x_nchw = x_nchw * se

            x_nchw = self._conv2d_nchw(
                x_nchw,
                self._get_param(f"{prefix}.projection.project_conv.weight"),
                bias=None,
                stride=1,
                padding=0,
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{prefix}.projection.project_bn",
                eps=float(self.config.batch_norm_eps),
                momentum=float(self.config.batch_norm_momentum),
                train=train,
            )

            if stride == 1 and not bool(spec["id_skip"]):
                x_nchw = x_nchw + residual

        x_nchw = self._conv2d_nchw(
            x_nchw,
            self._get_param("encoder.top_conv.weight"),
            bias=None,
            stride=1,
            padding=0,
        )
        x_nchw = self._batch_norm_nchw(
            x_nchw,
            "encoder.top_bn",
            eps=float(self.config.batch_norm_eps),
            momentum=float(self.config.batch_norm_momentum),
            train=train,
        )
        x_nchw = jnn.silu(x_nchw)

        return jnp.mean(x_nchw, axis=(2, 3))
