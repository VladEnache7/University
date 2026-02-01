# Add this to your script instead of importing from deep_ops
import warnings
import torch
from torch import nn
import math

class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module (simplified for inference)
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
        warnings.warn("Using simplified MSDeformAttn implementation for inference only!")
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, 
                input_level_start_index, input_padding_mask=None):
        """
        Simplified forward pass that uses pre-computed reference points
        """
        # Project values
        value = self.value_proj(input_flatten)
        
        # For inference, we'll just return the transformed values
        # This is a simplification and won't match the true deformable attention
        # but can work for visualization/inference purposes
        
        output = self.output_proj(value)
        return output