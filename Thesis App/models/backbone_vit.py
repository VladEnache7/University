from functools import partial

import torch

from models.layers import NestedTensorBlock as Block
from models.layers.attention import MemEffAttention

from .vision_transformer import DinoVisionTransformer


class BackboneViT(torch.nn.Module):
    """
    Generic Backbone for Vision Transformer models.
    Instantiates a DinoVisionTransformer with configurable options.
    """

    def __init__(self, vit_type: str, patch_tokens: bool = True):
        """
        Args:
            model_config (dict): Configuration for Vision Transformer model, e.g., 
                                 {"patch_size": 16, "embed_dim": 384, ...}
            patch_tokens (bool): Whether to return only the class token or to also add the patch tokens.
        """
        super().__init__()
        self.patch_tokens = patch_tokens
        self.model_config = {
            'vit_small': {"patch_size": 16, "embed_dim": 384, "depth": 12,
                   "num_heads": 6, "block_fn":partial(Block, attn_class=MemEffAttention)},
            'vit_base': {"patch_size": 16, "embed_dim": 768, "depth": 12,
                   "num_heads": 12, "block_fn":partial(Block, attn_class=MemEffAttention)},
            'vit_large': {"patch_size": 16, "embed_dim": 1024, "depth": 24,
                  "num_heads": 16, "block_fn":partial(Block, attn_class=MemEffAttention)},
            'vit_giant': {"patch_size": 16, "embed_dim": 1536, "depth": 32,
                   "num_heads": 24, "block_fn":partial(Block, attn_class=MemEffAttention)}
        }
        self.vit = DinoVisionTransformer(**self.model_config[vit_type]) # type: ignore[arg-type]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward function for backbone.
        
        Args:
            inputs (torch.Tensor): Inputs to the model.

        Returns:
            torch.Tensor: Class token or patch tokens of the model output.
        """
        model_output = self.vit(inputs)
        if self.patch_tokens:
            only_patch_tokens = model_output["x_norm_patchtokens"]
            only_cls_tokens = model_output["x_norm_clstoken"].unsqueeze(1)
            return only_patch_tokens + only_cls_tokens
        return model_output["x_norm_clstoken"]
