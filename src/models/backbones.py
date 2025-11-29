from typing import cast

import jax.numpy as jnp
from flax import nnx
from flax.core.frozen_dict import unfreeze
from transformers import FlaxResNetModel
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPoolingAndNoAttention,
)


def recursive_to_nnx(data, variable_type):
    if isinstance(data, dict):
        return nnx.Dict(
            {k: recursive_to_nnx(v, variable_type) for k, v in data.items()}
        )
    return variable_type(data)


def recursive_from_nnx(node):
    if isinstance(node, nnx.Dict):
        return {k: recursive_from_nnx(v) for k, v in node.items()}
    if hasattr(node, "value"):
        return node.value
    return node


class ResNet50Classifier(nnx.Module):
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        hf_model = cast(
            FlaxResNetModel, FlaxResNetModel.from_pretrained("microsoft/resnet-50")
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
        self.head = nnx.Linear(2048, num_classes, rngs=rngs)

    def __call__(self, x, train: bool = False):
        legacy_vars = recursive_from_nnx(self.backbone_params)
        backbone_output = self.backbone_module.apply(
            legacy_vars, pixel_values=x, mutable=False
        )
        backbone_output = cast(
            FlaxBaseModelOutputWithPoolingAndNoAttention, backbone_output
        )

        embedding = backbone_output.pooler_output
        assert embedding is not None, "pooler_output should not be None"
        embedding = jnp.squeeze(embedding, axis=(-1, -2))

        logits = self.head(embedding)
        return logits
