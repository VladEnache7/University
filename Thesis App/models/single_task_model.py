from __future__ import annotations
import torch
from torch import nn


class SingleTaskModel(nn.Module):
    """
    An implementation of the :class:`nn.Module`.
    """
    backbone: nn.Module
    head: nn.Module
    def __init__(self, backbone: nn.Module, head: nn.Module, freeze_backbone: bool = True) -> None:
        """
        Args:
            backbone: implementation of a :class:`models.BackboneInteface`
            head: implementation of a :class:`models.HeadInteface`
            freeze_backbone: if set to True, the backbones parameters won't be updated
        """
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        forward function for backbone
        Args:
            inputs: the inputs of the model
        Returns:
            the output of the entire model
        """
        backbone_out = self.backbone(inputs)
        return self.head(backbone_out)
