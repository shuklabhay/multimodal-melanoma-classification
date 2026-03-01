from __future__ import annotations

from typing import cast

import jax.nn as jnn
import jax.numpy as jnp
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.models.backbones.common import StateDictNCHWBackbone

MOBILENET_V4_CONV_LARGE_REPO = "timm/mobilenetv4_conv_large.e500_r256_in1k"


class MobileNetV4ConvLargeBackbone(StateDictNCHWBackbone):
    """timm MobileNetV4-Conv-Large backbone (pretrained on ImageNet-1k)."""

    def __init__(self):
        path = hf_hub_download(MOBILENET_V4_CONV_LARGE_REPO, "model.safetensors")
        state_dict = cast(dict[str, torch.Tensor], load_file(path))
        self.stage_block_indices = self._collect_stage_block_indices(state_dict)
        super().__init__(state_dict)

    @staticmethod
    def _collect_stage_block_indices(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[int, list[int]]:
        stage_to_blocks: dict[int, set[int]] = {}
        for key in state_dict:
            if not key.startswith("blocks."):
                continue
            parts = key.split(".")
            stage = int(parts[1])
            block = int(parts[2])
            stage_to_blocks.setdefault(stage, set()).add(block)

        return {stage: sorted(blocks) for stage, blocks in stage_to_blocks.items()}

    @staticmethod
    def _stage_stride(stage: int) -> int:
        stage_first_strides = {
            0: 2,
            1: 2,
            2: 2,
            3: 2,
            4: 1,
        }
        return stage_first_strides[stage]

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x_nchw = jnp.transpose(x, (0, 3, 1, 2))

        x_nchw = self._conv2d_nchw(
            x_nchw,
            self._get_param("conv_stem.weight"),
            bias=None,
            stride=2,
            padding=1,
        )
        x_nchw = self._batch_norm_nchw(
            x_nchw,
            "bn1",
            eps=1e-5,
            momentum=0.1,
            train=train,
        )
        x_nchw = jnn.relu(x_nchw)

        for pos, block_idx in enumerate(self.stage_block_indices.get(0, [])):
            stride = self._stage_stride(0) if pos == 0 else 1
            prefix = f"blocks.0.{block_idx}"
            residual = x_nchw

            exp_w = self._get_param(f"{prefix}.conv_exp.weight")
            k = int(exp_w.shape[-1])
            x_nchw = self._conv2d_nchw(
                x_nchw,
                exp_w,
                bias=None,
                stride=stride,
                padding=k // 2,
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{prefix}.bn1",
                eps=1e-5,
                momentum=0.1,
                train=train,
            )
            x_nchw = jnn.relu(x_nchw)

            x_nchw = self._conv2d_nchw(
                x_nchw,
                self._get_param(f"{prefix}.conv_pwl.weight"),
                bias=None,
                stride=1,
                padding=0,
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{prefix}.bn2",
                eps=1e-5,
                momentum=0.1,
                train=train,
            )

            if stride == 1 and residual.shape[1] == x_nchw.shape[1]:
                x_nchw = x_nchw + residual

        for stage in [1, 2, 3]:
            block_ids = self.stage_block_indices.get(stage, [])
            for pos, block_idx in enumerate(block_ids):
                stride = self._stage_stride(stage) if pos == 0 else 1
                prefix = f"blocks.{stage}.{block_idx}"
                residual = x_nchw

                has_dw_mid = self._has_param(f"{prefix}.dw_mid.conv.weight")

                dw_start_w = self._get_param(f"{prefix}.dw_start.conv.weight")
                dw_start_k = int(dw_start_w.shape[-1])
                dw_start_stride = stride if not has_dw_mid else 1
                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    dw_start_w,
                    bias=None,
                    stride=dw_start_stride,
                    padding=dw_start_k // 2,
                    groups=int(dw_start_w.shape[0]),
                )
                x_nchw = self._batch_norm_nchw(
                    x_nchw,
                    f"{prefix}.dw_start.bn",
                    eps=1e-5,
                    momentum=0.1,
                    train=train,
                )

                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    self._get_param(f"{prefix}.pw_exp.conv.weight"),
                    bias=None,
                    stride=1,
                    padding=0,
                )
                x_nchw = self._batch_norm_nchw(
                    x_nchw,
                    f"{prefix}.pw_exp.bn",
                    eps=1e-5,
                    momentum=0.1,
                    train=train,
                )
                x_nchw = jnn.relu(x_nchw)

                if has_dw_mid:
                    dw_mid_w = self._get_param(f"{prefix}.dw_mid.conv.weight")
                    dw_mid_k = int(dw_mid_w.shape[-1])
                    x_nchw = self._conv2d_nchw(
                        x_nchw,
                        dw_mid_w,
                        bias=None,
                        stride=stride,
                        padding=dw_mid_k // 2,
                        groups=int(dw_mid_w.shape[0]),
                    )
                    x_nchw = self._batch_norm_nchw(
                        x_nchw,
                        f"{prefix}.dw_mid.bn",
                        eps=1e-5,
                        momentum=0.1,
                        train=train,
                    )
                    x_nchw = jnn.relu(x_nchw)

                x_nchw = self._conv2d_nchw(
                    x_nchw,
                    self._get_param(f"{prefix}.pw_proj.conv.weight"),
                    bias=None,
                    stride=1,
                    padding=0,
                )
                x_nchw = self._batch_norm_nchw(
                    x_nchw,
                    f"{prefix}.pw_proj.bn",
                    eps=1e-5,
                    momentum=0.1,
                    train=train,
                )

                if self._has_param(f"{prefix}.dw_end.conv.weight"):
                    dw_end_w = self._get_param(f"{prefix}.dw_end.conv.weight")
                    dw_end_k = int(dw_end_w.shape[-1])
                    x_nchw = self._conv2d_nchw(
                        x_nchw,
                        dw_end_w,
                        bias=None,
                        stride=1,
                        padding=dw_end_k // 2,
                        groups=int(dw_end_w.shape[0]),
                    )
                    x_nchw = self._batch_norm_nchw(
                        x_nchw,
                        f"{prefix}.dw_end.bn",
                        eps=1e-5,
                        momentum=0.1,
                        train=train,
                    )

                if stride == 1 and residual.shape[1] == x_nchw.shape[1]:
                    x_nchw = x_nchw + residual

        for block_idx in self.stage_block_indices.get(4, []):
            prefix = f"blocks.4.{block_idx}"
            w = self._get_param(f"{prefix}.conv.weight")
            k = int(w.shape[-1])
            x_nchw = self._conv2d_nchw(
                x_nchw,
                w,
                bias=None,
                stride=1,
                padding=k // 2,
            )
            x_nchw = self._batch_norm_nchw(
                x_nchw,
                f"{prefix}.bn1",
                eps=1e-5,
                momentum=0.1,
                train=train,
            )
            x_nchw = jnn.relu(x_nchw)

        return jnp.mean(x_nchw, axis=(2, 3))
