"""------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------ """
from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.util import inverse_sigmoid, get_activation_fn, get_clones

from models.ops.modules import MSDeformAttn

class DeformableTransformer(nn.Module):
    """Deformable Transformer module for object detection.


    Args:
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        return_intermediate_dec (bool): Whether to return intermediate decoder layers.
        num_feature_levels (int): Number of feature levels.
        dec_n_points (int): Number of sampling points in decoder.
        enc_n_points (int): Number of sampling points in encoder.
        two_stage (bool): Whether to use two-stage approach.
        two_stage_num_proposals (int): Number of proposals in two-stage approach.
    """
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, *, num_decoder_layers=6,  dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                            dropout, activation, n_levels=num_feature_levels,
                            n_heads=nhead, n_points=enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                            dropout, activation, n_levels=num_feature_levels,
                            n_heads=nhead, n_points=dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        """Resets the model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):  # pylint: disable=W1116
                m._reset_parameters() # pylint: disable=protected-access, W1116
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    @staticmethod
    def get_proposal_pos_embed(proposals):
        """Generates positional embeddings for proposals.

        Args:
            proposals (Tensor): Input proposals.

        Returns:
            Tensor: Positional embeddings.
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """Generates encoder output proposals.

        Args:
            memory (Tensor): Encoder memory output.
            memory_padding_mask (Tensor): Mask for padding.
            spatial_shapes (Tensor): Spatial shapes of features.

        Returns:
            Tuple[Tensor, Tensor]: Output memory and output proposals.
        """
        n, _, _ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (h, w) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + h * w)].view(n, h, w, 1)
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, w - 1, w, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(n, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(n, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(n, -1, 4)
            proposals.append(proposal)
            _cur += (h * w)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """Computes valid ratios for input mask.

        Args:
            mask (Tensor): Input mask.

        Returns:
            Tensor: Valid ratio.
        """
        _, h, w= mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """Forward pass for Deformable Transformer.

        Args:
            srcs (List[Tensor]): Source features.
            masks (List[Tensor]): Padding masks.
            pos_embeds (List[Tensor]): Positional embeddings.
            query_embed (Optional[Tensor]): Query embedding.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]: Outputs from transformer layers.
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes,
                              level_start_index, valid_ratios,
                              pos=lvl_pos_embed_flatten, padding_mask=mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # code from original code, but pylint gives (unsubscriptable-object)
            # enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
            #                                                                        + output_proposals

            enc_outputs_class, enc_outputs_coord_unact = None, None

            # hack implementation for two-stage Deformable DETR
            if self.decoder.class_embed is not None:
                class_embed_layer = self.decoder.class_embed[self.decoder.num_layers]
                enc_outputs_class = class_embed_layer(output_memory)
            if self.decoder.bbox_embed is not None:
                bbox_embed_layer = self.decoder.bbox_embed[self.decoder.num_layers]
                enc_outputs_coord_unact = bbox_embed_layer(output_memory) + output_proposals


            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(
                    DeformableTransformer.get_proposal_pos_embed(
                        topk_coords_unact
                    )
                )
            )
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes,
                                            src_level_start_index=level_start_index,
                                            src_valid_ratios=valid_ratios, query_pos=query_embed,
                                            src_padding_mask=mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    """Encoder layer for Deformable Transformer.

    Args:
        d_model (int): Dimension of model.
        d_ffn (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points.
    """
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", *,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points) # pylint: disable=E1102
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Adds positional encoding to the input tensor.

        Args:
            tensor (Tensor): Input tensor of shape `(batch_size, num_queries, d_model)`.
            pos (Optional[Tensor]): Positional encoding tensor of the same shape as `tensor`, or `None`.

        Returns:
            Tensor: Tensor with positional encoding added, or the original tensor if `pos` is `None`.
        """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        """Forward pass for the feedforward network (FFN) in the transformer layer.

        Args:
            src (Tensor): Input tensor of shape `(batch_size, num_queries, d_model)`.

        Returns:
            Tensor: Output tensor after applying the feedforward network.
        """
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, *, level_start_index, padding_mask=None):
        """Forward pass for encoder layer.

        Args:
            src (Tensor): Input source.
            pos (Tensor): Positional encoding.
            reference_points (Tensor): Reference points for deformable attention.
            spatial_shapes (Tensor): Spatial shapes of input.
            level_start_index (Tensor): Start index for each feature level.
            padding_mask (Optional[Tensor]): Mask for padding.

        Returns:
            Tensor: Output feature maps.
        """
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    """Encoder module for Deformable Transformer.

    Args:
        encoder_layer (nn.Module): Encoder layer module.
        num_layers (int): Number of encoder layers.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Generates reference points for deformable attention.

        Args:
            spatial_shapes (Tensor): Spatial shapes of input feature maps.
            valid_ratios (Tensor): Valid ratios for feature maps.
            device (torch.device): Device where tensors are allocated.

        Returns:
            Tensor: Reference points for deformable attention.
        """
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, *, pos=None, padding_mask=None):
        """Forward pass of Deformable Transformer Encoder.

        Args:
            src (Tensor): Input tensor.
            spatial_shapes (Tensor): Shapes of input feature maps.
            level_start_index (Tensor): Start indices for each level.
            valid_ratios (Tensor): Valid ratios for feature maps.
            pos (Optional[Tensor]): Positional embeddings.
            padding_mask (Optional[Tensor]): Padding mask.

        Returns:
            Tensor: Output feature map after encoding.
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index=level_start_index, padding_mask=padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Decoder layer for Deformable Transformer.

    Args:
        d_model (int): Dimension of model.
        d_ffn (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points.
    """
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", *,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points) # pylint: disable=E1102
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Adds positional encoding to the input tensor.

        Args:
            tensor (Tensor): Input tensor of shape `(batch_size, num_queries, d_model)`.
            pos (Optional[Tensor]): Positional encoding tensor of the same shape as `tensor`, or `None`.

        Returns:
            Tensor: Tensor with positional encoding added, or the original tensor if `pos` is `None`.
        """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Forward pass for the feedforward network (FFN) in the transformer decoder layer.

        Args:
            tgt (Tensor): Target tensor of shape `(batch_size, num_queries, d_model)`.

        Returns:
            Tensor: Output tensor after applying the feedforward network.
        """
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, *,
                src_spatial_shapes, level_start_index, src_padding_mask=None):
        """Forward pass for decoder layer.

        Args:
            tgt (Tensor): Target sequence tensor.
            query_pos (Tensor): Query positional encoding.
            reference_points (Tensor): Reference points for deformable attention.
            src (Tensor): Input source features.
            src_spatial_shapes (Tensor): Spatial shapes of the source.
            level_start_index (Tensor): Start index for each feature level.
            src_padding_mask (Optional[Tensor]): Mask for padding.
        
        Returns:
            Tensor: Output tensor after decoding.
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    """Decoder module for Deformable Transformer.

    Args:
        decoder_layer (nn.Module): Decoder layer module.
        num_layers (int): Number of decoder layers.
        return_intermediate (bool): Whether to return intermediate outputs.
    """
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, *,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """Forward pass for Deformable Transformer Decoder.

        Args:
            tgt (Tensor): Target sequence tensor.
            reference_points (Tensor): Reference points for attention.
            src (Tensor): Input source features.
            src_spatial_shapes (Tensor): Spatial shapes of source.
            src_level_start_index (Tensor): Start index for each feature level.
            src_valid_ratios (Tensor): Valid ratios for input features.
            query_pos (Optional[Tensor]): Positional embeddings for queries.
            src_padding_mask (Optional[Tensor]): Mask for padding.
        
        Returns:
            Tuple[Tensor, Tensor]: Decoded output and updated reference points.
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src,
                           src_spatial_shapes=src_spatial_shapes,
                           level_start_index=src_level_start_index, src_padding_mask=src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
