from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
from transformers import ConvNextModel

from src.models.backbones.common import StateDictNCHWBackbone


class ConvNeXtTinyBackbone(StateDictNCHWBackbone):
    def __init__(self):
        model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
        self.depths = list(model.config.depths)
        self.layer_norm_eps = float(model.config.layer_norm_eps)
        super().__init__(model.state_dict())

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        del train

        x_nchw = jnp.transpose(x, (0, 3, 1, 2))

        stem_weight = self._get_param("embeddings.patch_embeddings.weight")
        stem_bias = self._get_param("embeddings.patch_embeddings.bias")
        x_nchw = self._conv2d_nchw(
            x_nchw,
            stem_weight,
            stem_bias,
            stride=4,
            padding=0,
        )

        x_nchw = self._layer_norm_nchw(
            x_nchw,
            self._get_param("embeddings.layernorm.weight"),
            self._get_param("embeddings.layernorm.bias"),
            eps=1e-6,
        )

        for stage_idx, depth in enumerate(self.depths):
            if stage_idx > 0:
                norm_prefix = f"encoder.stages.{stage_idx}.downsampling_layer.0"
                conv_prefix = f"encoder.stages.{stage_idx}.downsampling_layer.1"

                x_nchw = self._layer_norm_nchw(
                    x_nchw,
                    self._get_param(f"{norm_prefix}.weight"),
                    self._get_param(f"{norm_prefix}.bias"),
                    eps=1e-6,
                )
                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    self._get_param(f"{conv_prefix}.weight"),
                    self._get_param(f"{conv_prefix}.bias"),
                    stride=2,
                    padding=0,
                )

            for layer_idx in range(depth):
                prefix = f"encoder.stages.{stage_idx}.layers.{layer_idx}"
                residual = x_nchw

                dw_weight = self._get_param(f"{prefix}.dwconv.weight")
                dw_bias = self._get_param(f"{prefix}.dwconv.bias")
                groups = int(dw_weight.shape[0])
                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    dw_weight,
                    dw_bias,
                    stride=1,
                    padding=3,
                    groups=groups,
                )

                x_nhwc = jnp.transpose(x_nchw, (0, 2, 3, 1))
                x_nhwc = self._layer_norm_nhwc(
                    x_nhwc,
                    self._get_param(f"{prefix}.layernorm.weight"),
                    self._get_param(f"{prefix}.layernorm.bias"),
                    eps=1e-6,
                )

                x_nhwc = self._linear_last_dim(
                    x_nhwc,
                    self._get_param(f"{prefix}.pwconv1.weight"),
                    self._get_param(f"{prefix}.pwconv1.bias"),
                )
                x_nhwc = jnn.gelu(x_nhwc)
                x_nhwc = self._linear_last_dim(
                    x_nhwc,
                    self._get_param(f"{prefix}.pwconv2.weight"),
                    self._get_param(f"{prefix}.pwconv2.bias"),
                )

                layer_scale = self._get_param(f"{prefix}.layer_scale_parameter")
                x_nhwc = x_nhwc * layer_scale.reshape((1, 1, 1, -1))

                x_nchw = jnp.transpose(x_nhwc, (0, 3, 1, 2))
                x_nchw = x_nchw + residual

        pooled = jnp.mean(x_nchw, axis=(2, 3))
        pooled = self._layer_norm_nhwc(
            pooled,
            self._get_param("layernorm.weight"),
            self._get_param("layernorm.bias"),
            eps=self.layer_norm_eps,
        )
        return pooled
