from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from flax import nnx
from flax.core.frozen_dict import unfreeze
from transformers import FlaxResNetModel
from transformers.modeling_flax_outputs import \
    FlaxBaseModelOutputWithPoolingAndNoAttention

from src.models.backbones.common import recursive_from_nnx, recursive_to_nnx


class ResNet50Backbone(nnx.Module):
    def __init__(self):
        hf_model = cast(
            FlaxResNetModel,
            FlaxResNetModel.from_pretrained("microsoft/resnet-50"),
        )
        self.backbone_module = hf_model.module

        all_params = unfreeze(hf_model.params)
        self.backbone_params = nnx.Dict(
            {
                "params": recursive_to_nnx(all_params.get("params", {}), nnx.Param),
                "batch_stats": recursive_to_nnx(
                    all_params.get("batch_stats", {}), nnx.BatchStat
                ),
            }
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        legacy_vars = recursive_from_nnx(self.backbone_params)

        if train:
            backbone_output, mutated = self.backbone_module.apply(
                legacy_vars,
                pixel_values=x,
                mutable=["batch_stats"],
            )
            updated_batch_stats = mutated.get("batch_stats", {})
            self.backbone_params["batch_stats"] = recursive_to_nnx(
                updated_batch_stats,
                nnx.BatchStat,
            )
        else:
            backbone_output = self.backbone_module.apply(
                legacy_vars,
                pixel_values=x,
                mutable=False,
            )

        backbone_output = cast(
            FlaxBaseModelOutputWithPoolingAndNoAttention,
            backbone_output,
        )

        embedding = backbone_output.pooler_output
        assert embedding is not None, "pooler_output should not be None"
        return jnp.squeeze(embedding, axis=(-1, -2))
