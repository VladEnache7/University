"""Utility functions for mathematical transformations used in the Deformable DETR model.

This module contains functions such as `inverse_sigmoid`, which is used to compute
the inverse of the sigmoid function, ensuring numerical stability.
"""
import torch

def inverse_sigmoid(x, eps=1e-5):
    """ Computes the inverse sigmoid function.

    Args:
        x (Tensor): Input tensor.
        eps (float): Small value to avoid numerical instability.

    Returns:
        Tensor: Inverse sigmoid values.
    """
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))
