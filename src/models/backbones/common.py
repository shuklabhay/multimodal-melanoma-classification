from __future__ import annotations

import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
from flax import nnx


def recursive_to_nnx(data, variable_type):
    """Recursively converts nested dicts to nnx.Dict with variable wrappers."""
    if isinstance(data, dict):
        return nnx.Dict({k: recursive_to_nnx(v, variable_type) for k, v in data.items()})
    return variable_type(data)


def recursive_from_nnx(node):
    """Recursively converts nnx.Dict structures back to plain Python dicts."""
    if isinstance(node, nnx.Dict):
        return {k: recursive_from_nnx(v) for k, v in node.items()}
    if hasattr(node, "value"):
        return node.value
    return node


class StateDictNCHWBackbone(nnx.Module):
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        include_prefixes: tuple[str, ...] | None = None,
    ):
        params: dict[str, nnx.Param] = {}
        stats: dict[str, nnx.BatchStat] = {}

        for key, value in state_dict.items():
            if include_prefixes is not None and not key.startswith(include_prefixes):
                continue
            if key.endswith("num_batches_tracked"):
                continue

            array = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            safe_key = self._safe_key(key)

            if key.endswith("running_mean") or key.endswith("running_var"):
                stats[safe_key] = nnx.BatchStat(array)
            else:
                params[safe_key] = nnx.Param(array)

        self.params = nnx.Dict(params)
        self.batch_stats = nnx.Dict(stats)

    @staticmethod
    def _safe_key(key: str) -> str:
        return key.replace(".", "__")

    def _has_param(self, key: str) -> bool:
        return hasattr(self.params, self._safe_key(key))

    def _get_param(self, key: str) -> jnp.ndarray:
        return self.params[self._safe_key(key)][...]

    def _get_batch_stat(self, key: str) -> jnp.ndarray:
        return self.batch_stats[self._safe_key(key)][...]

    def _to_pair(self, value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, tuple):
            return value
        return value, value

    def _pad_nchw(
        self,
        x: jnp.ndarray,
        *,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> jnp.ndarray:
        return jnp.pad(
            x,
            ((0, 0), (0, 0), (top, bottom), (left, right)),
            mode="constant",
        )

    def _conv2d_nchw(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray | None,
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> jnp.ndarray:
        stride_pair = self._to_pair(stride)
        dilation_pair = self._to_pair(dilation)
        padding_pair = self._to_pair(padding)

        y = lax.conv_general_dilated(
            lhs=x,
            rhs=weight,
            window_strides=stride_pair,
            padding=((padding_pair[0], padding_pair[0]), (padding_pair[1], padding_pair[1])),
            lhs_dilation=(1, 1),
            rhs_dilation=dilation_pair,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=groups,
        )

        if bias is not None:
            y = y + bias.reshape((1, -1, 1, 1))
        return y

    def _batch_norm_nchw(
        self,
        x: jnp.ndarray,
        prefix: str,
        *,
        eps: float,
        momentum: float,
        train: bool,
    ) -> jnp.ndarray:
        weight = self._get_param(f"{prefix}.weight")
        bias = self._get_param(f"{prefix}.bias")

        running_mean_key = self._safe_key(f"{prefix}.running_mean")
        running_var_key = self._safe_key(f"{prefix}.running_var")
        running_mean = self.batch_stats[running_mean_key][...]
        running_var = self.batch_stats[running_var_key][...]

        if train:
            batch_mean = jnp.mean(x, axis=(0, 2, 3))
            batch_var = jnp.var(x, axis=(0, 2, 3))

            updated_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
            updated_var = (1.0 - momentum) * running_var + momentum * batch_var

            self.batch_stats[running_mean_key].value = updated_mean
            self.batch_stats[running_var_key].value = updated_var

            mean = batch_mean
            var = batch_var
        else:
            mean = running_mean
            var = running_var

        x_hat = (x - mean.reshape((1, -1, 1, 1))) / jnp.sqrt(
            var.reshape((1, -1, 1, 1)) + eps
        )
        return x_hat * weight.reshape((1, -1, 1, 1)) + bias.reshape((1, -1, 1, 1))

    def _layer_norm_nhwc(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray,
        *,
        eps: float,
    ) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        return ((x - mean) / jnp.sqrt(var + eps)) * weight + bias

    def _layer_norm_nchw(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray,
        *,
        eps: float,
    ) -> jnp.ndarray:
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        x_nhwc = self._layer_norm_nhwc(x_nhwc, weight, bias, eps=eps)
        return jnp.transpose(x_nhwc, (0, 3, 1, 2))

    def _linear_last_dim(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray,
    ) -> jnp.ndarray:
        y = jnp.einsum("...i,oi->...o", x, weight)
        return y + bias
