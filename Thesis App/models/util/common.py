"""Neural network module utilities.  

Functions:  
    _get_clones: Create N identical copies of a neural network module  
    _get_activation_fn: Map activation names to their functional implementations   
"""
import copy
import torch.nn.functional as F
from torch import nn

def get_clones(module, n): # pylint: disable=missing-type-doc
    """Create N identical copies of a neural network module.  
    
    Typically used in transformer architectures to create multiple layers  
    with identical structure but separate parameters.  
    
    Args:  
        module: The base module template to be cloned  
        n: Number of duplicate modules to create  
        
    Returns:  
        ModuleList containing the cloned modules  
        
    Raises:  
        ValueError: If n is not a positive integer
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def get_activation_fn(activation):
    """Return an activation function given a string.

    Args:
        activation (str): Name of the activation function ('relu', 'gelu', or 'glu')

    Returns:
        callable: The corresponding PyTorch activation function

    Raises:
        RuntimeError: If activation is not one of the supported types
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
