# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import nn

from .position_encoding import build_position_encoding
from .transformer import Transformer, build_transformer
from .util.misc import nested_tensor_from_tensor_list


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, num_channels: int, transformer: Transformer, position_embedding: nn.Module,
                 num_classes: int, num_queries: int, aux_loss: bool = False, out_channels: int = 4):
        """ Initializes the model.
        Parameters:
            num_channels: int representing the number of channels from the backbone
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, out_channels, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.num_channels = num_channels
        self.aux_loss = aux_loss
        self.position_embedding = position_embedding

    def forward(self, backbone_outputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        backbone_outputs = backbone_outputs.permute(0, 2, 1)
        # we want the tokens to be that
        features = backbone_outputs.reshape(-1, self.num_channels, 32, 104)
        nested_features = nested_tensor_from_tensor_list(features)
        if self.position_embedding is not None:
            pos = self.position_embedding(nested_features).to(nested_features.tensors.dtype)
        else:
            pos = None
        src, mask = nested_features.decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, # instead of the src it was self.input_proj(src)
                              self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(num_objects: int, num_classes: int, args: ListConfig | DictConfig) -> DETR:
    """
    Function to build a DETR head
    Args:
        num_objects: the number of objects (queries) to return
        num_classes: the number of classes (without that no_object class)
        args: a dictconfig with all the params for building the transformer and position encoding
    Returns:
        a DETR head configured as specified
    """
    transformer = build_transformer(args.transformer)

    position_encoding = build_position_encoding(
        transformer.d_model, args.position_encoding)

    model = DETR(
        num_channels=args.num_channels,
        transformer=transformer,
        position_embedding=position_encoding,
        num_classes=num_classes,
        num_queries=num_objects,
        aux_loss=args.aux_loss,
        out_channels=args.out_channels
    )
    return model
