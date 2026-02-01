# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""Backbone modules."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision  # type: ignore
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter  # type: ignore

from .utils import NestedTensor, build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict,  # pylint: disable=too-many-positional-arguments
                              prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Does the forward prop
        Args:
            x: Tensor to be normalized
        Returns:
            The input tensor normalized
        """
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """Base class for ResNet implementation
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor) -> dict[str, NestedTensor]:
        """Does the forward prop through the resnet model
        Args:
            tensor_list: Input data
        Returns:
            The output from intermediate layers and the final output
        """
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation])
        assert name not in (
            'resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class ResnetBackbone(nn.Sequential):
    """Combines the Backbone with the position embedding module"""

    def __init__(self, type_backbone: str, hidden_dim: int, type_embedding: str,
                 return_interm_layers: bool = False, *, train_backbone: bool = True, dilation: bool = False):
        position_embedding = build_position_encoding(
            hidden_dim, type_embedding)
        backbone = Backbone(type_backbone, train_backbone,
                            return_interm_layers, dilation)
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, input: NestedTensor) -> tuple[list[NestedTensor], list[torch.Tensor]]:  # pylint: disable=W0622
        """Does the forward prop through the resnet backbone and computes position embedding for them
        Args:
            input: input data to process
        Returns:
            the outputs from the resnet and the position embeddings calculated
        """
        xs = self[0](input)
        out: list[NestedTensor] = []
        pos = []
        for _, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos
