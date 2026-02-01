
"""Model loading and initialization utilities."""

from __future__ import annotations

import torch
from typing import Dict, Set, Optional

from models.deformable_detr import DeformableDETR, ResnetBackbone
from models.deformable_transformer import DeformableTransformer
from utils import CLASSES
from config import MODEL_CONFIG, BACKBONE_CONFIG, DETR_CONFIG, DEFAULT_CHECKPOINT_PATH, get_device


def load_weights(model: torch.nn.Module, loaded_weights: Dict[str, torch.Tensor]) -> Set[str]:
    """Loads weights which are not necessarily complete into a model.

    Args:
        model: Target model for the weights
        loaded_weights: Loaded state dict

    Returns:
        Set[str]: Set of uninitialized weight names
    """
    model_weights = model.state_dict()

    weights_not_loaded = set()
    initialized_weights = set()
    bad_shaped_weights = set()
    
    for name, weight in loaded_weights.items():
        if name in model_weights:
            if model_weights[name].shape == weight.shape:
                model_weights[name] = weight
                initialized_weights.add(name)
            else:
                bad_shaped_weights.add(name)
        else:
            weights_not_loaded.add(name)

    uninitialized_weights = set(model_weights).difference(initialized_weights)

    model.load_state_dict(model_weights)
    
    # Print loading statistics
    print(f"Target model has {len(model_weights)} weights, received state dict has {len(loaded_weights)} weights.")
    print(f"Present in state dict but not in model {len(weights_not_loaded)} weights: {weights_not_loaded}")
    print(f"Uninitialized {len(uninitialized_weights)} weights: {uninitialized_weights}")
    print(f"Incorrect shape {len(bad_shaped_weights)} weights: {bad_shaped_weights}")

    return uninitialized_weights


def create_transformer() -> DeformableTransformer:
    """Create the DeformableTransformer with configured parameters.
    
    Returns:
        DeformableTransformer: Initialized transformer
    """
    return DeformableTransformer(**MODEL_CONFIG)


def create_backbone() -> ResnetBackbone:
    """Create the ResNet backbone with configured parameters.
    
    Returns:
        ResnetBackbone: Initialized backbone
    """
    return ResnetBackbone(**BACKBONE_CONFIG)


def create_model() -> DeformableDETR:
    """Create the DeformableDETR model with configured parameters.
    
    Returns:
        DeformableDETR: Initialized DeformableDETR model
    """
    # Create the transformer
    transformer = create_transformer()

    # Create the backbone
    resnet_backbone = create_backbone()

    # Create the model
    model = DeformableDETR(
        backbone=resnet_backbone,
        transformer=transformer,
        num_classes=len(CLASSES) + 1,  # +1 for background class
        **DETR_CONFIG
    )
    
    return model


def load_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load model checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dict[str, torch.Tensor]: Loaded checkpoint weights
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded successfully from: {checkpoint_path}")
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


def initialize_model(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH) -> DeformableDETR:
    """Initialize model and load weights from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        DeformableDETR: Initialized and loaded model ready for inference
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model initialization fails
    """
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Create model
        model = create_model()
        
        # Load weights
        uninitialized_weights = load_weights(model, checkpoint)
        
        if uninitialized_weights:
            print(f"Warning: {len(uninitialized_weights)} weights were not initialized from checkpoint")
        
        print("The model was loaded successfully")
        
        # Move model to appropriate device
        device = get_device()
        model = model.to(device)
        model.eval()
        
        print(f"Model moved to device: {device}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")


def get_model_info(model: torch.nn.Module) -> Dict[str, any]:
    """Get information about the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict[str, any]: Dictionary containing model information
    """
    device = next(model.parameters()).device
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'device': str(device),
        'total_parameters': num_params,
        'trainable_parameters': num_trainable_params,
        'model_type': type(model).__name__,
        'training_mode': model.training
    }


def set_model_device(model: torch.nn.Module, device: Optional[str] = None) -> torch.nn.Module:
    """Move model to specified device.
    
    Args:
        model: Model to move
        device: Target device ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        torch.nn.Module: Model on the specified device
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    print(f"Model moved to device: {device}")
    return model


def prepare_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for inference (set to eval mode and move to appropriate device).
    
    Args:
        model: Model to prepare
        
    Returns:
        torch.nn.Module: Model ready for inference
    """
    model.eval()
    model = set_model_device(model)
    return model


def save_model_checkpoint(
    model: torch.nn.Module, 
    filepath: str, 
    additional_info: Optional[Dict] = None
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        filepath: Path to save the checkpoint
        additional_info: Additional information to save with the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': get_model_info(model)
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to: {filepath}")


def validate_model_architecture(model: torch.nn.Module) -> bool:
    """Validate that the model architecture is correct.
    
    Args:
        model: Model to validate
        
    Returns:
        bool: True if architecture is valid, False otherwise
    """
    try:
        # Check if model has required components
        required_components = ['backbone', 'transformer', 'class_embed', 'bbox_embed']
        
        for component in required_components:
            if not hasattr(model, component):
                print(f"Missing component: {component}")
                return False
        
        # Check if model can be put in eval mode
        model.eval()
        
        # Basic forward pass test with dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            try:
                output = model(dummy_input)
                # Check if output has expected keys
                expected_keys = ['pred_logits', 'pred_boxes']
                if not all(key in output for key in expected_keys):
                    print(f"Output missing expected keys: {expected_keys}")
                    return False
            except Exception as e:
                print(f"Forward pass failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False
