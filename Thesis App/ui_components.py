
"""UI component utilities and helper functions."""

import os
from typing import List
from app_config import CLASS_OPTIONS, COMMON_CLASSES, EXAMPLES_CONFIG


def select_all_classes() -> List[str]:
    """Return all available classes for selection.
    
    Returns:
        List of all class options
    """
    return CLASS_OPTIONS


def select_common_classes() -> List[str]:
    """Return commonly used classes for selection.
    
    Returns:
        List of common class options
    """
    return COMMON_CLASSES


def clear_selection() -> List[str]:
    """Return empty list to clear class selection.
    
    Returns:
        Empty list
    """
    return []


def get_example_files() -> List[str]:
    """Get list of example image files if available.
    
    Returns:
        List of example file paths
    """
    example_dir = EXAMPLES_CONFIG['directory']
    if not os.path.exists(example_dir):
        return []
    
    example_files = []
    for f in os.listdir(example_dir):
        if f.lower().endswith(EXAMPLES_CONFIG['supported_extensions']):
            example_files.append(os.path.join(example_dir, f))
    
    return example_files


def validate_image_input(image) -> bool:
    """Validate if the provided image input is valid.
    
    Args:
        image: Image input to validate
        
    Returns:
        True if valid, False otherwise
    """
    return image is not None


def format_confidence_threshold(threshold: float) -> str:
    """Format confidence threshold for display.
    
    Args:
        threshold: Confidence threshold value
        
    Returns:
        Formatted threshold string
    """
    return f"{threshold:.2f}"
