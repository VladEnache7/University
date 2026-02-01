# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py
from __future__ import annotations

from typing import Callable

from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,h,w) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: int |  tuple[int, int] = 224,
        patch_size: int |  tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768, *,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        patch_h, patch_w = self.patch_size

        assert h % patch_h == 0, f"Input image height {h} is not a multiple of patch height {patch_h}"
        assert w % patch_w == 0, f"Input image width {w} is not a multiple of patch width: {patch_w}"

        x = self.proj(x)  # B C h w
        h, w = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, h, w, self.embed_dim)  # B h w C
        return x

    def flops(self) -> float:
        ho, wo = self.patches_resolution
        flops = ho * wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += ho * wo * self.embed_dim
        return flops
