# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110
from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """
    LayerScale is a simple learnable scaling mechanism applied to the output of 
    residual blocks, typically used in vision transformers. It scales the input 
    by a learnable parameter `gamma`.

    Args:
        dim (int): The dimension of the input tensor to be scaled.
        init_values (float or Tensor): The initial values for the scaling parameter `gamma`.
                                       This can be a float or a tensor.
        inplace (bool): If True, performs the scaling operation in-place on the input tensor.

    Attributes:
        inplace (bool): Whether to perform the scaling operation in-place.
        gamma (Tensor): A learnable parameter used to scale the input tensor.
    """

    inplace: bool
    gamma: nn.Parameter
    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        """
        Initializes the LayerScale module.

        Args:
            dim (int): The dimension of the input tensor.
            init_values (float or Tensor): Initial scaling values for the `gamma` parameter.
                                           Typically a small value like 1e-5.
            inplace (bool): If True, modifies the input tensor in place during the scaling operation.
        """

        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LayerScale module.

        Args:
            x (Tensor): The input tensor to be scaled.

        Returns:
            Tensor: The scaled input tensor.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
