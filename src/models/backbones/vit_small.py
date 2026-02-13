from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from flax import nnx
from flax.core.frozen_dict import unfreeze
from transformers import FlaxViTModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling

from src.models.backbones.common import recursive_from_nnx, recursive_to_nnx


class ViTSmallBackbone(nnx.Module):
    def __init__(self):
        hf_model = cast(
            FlaxViTModel,
            FlaxViTModel.from_pretrained(
                "WinKawaks/vit-small-patch16-224",
                from_pt=True,
            ),
        )
        self.backbone_module = hf_model.module
        all_params = unfreeze(hf_model.params)
        self.backbone_params = nnx.Dict(
            {"params": recursive_to_nnx(all_params, nnx.Param)}
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        del train
        legacy_vars = recursive_from_nnx(self.backbone_params)
        backbone_output = self.backbone_module.apply(
            legacy_vars,
            pixel_values=x,
            mutable=False,
        )
        backbone_output = cast(FlaxBaseModelOutputWithPooling, backbone_output)

        if backbone_output.pooler_output is not None:
            return backbone_output.pooler_output

        return backbone_output.last_hidden_state[:, 0]
