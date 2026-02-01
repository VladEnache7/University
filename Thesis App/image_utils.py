
"""Image processing and transformation utilities."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torchvision.transforms import transforms
from typing import Tuple, Union

from config import MEAN, STD


def transform_image(img: np.ndarray) -> torch.Tensor:
    """Applies the standard transformations to an image.

    Args:
        img: The image as a NumPy array.

    Returns:
        The transformed image as a PyTorch tensor.
    """
    # Step 1: Convert to PIL Image
    pil_img = transforms.ToPILImage()(img)
    
    # Step 2: Convert to tensor (scales to [0,1] and converts to float32)
    tensor_img = transforms.ToTensor()(pil_img)
    
    # Step 3: Normalize using mean and std
    normalized_img = transforms.Normalize(mean=MEAN, std=STD)(tensor_img)
    
    # This ensures we're explicitly returning a tensor
    return normalized_img


def get_image_rgb(img_path: str) -> ndarray:
    """Loads an image in RGB format.

    Args:
        img_path: Path to the image file.

    Returns:
        ndarray: A numpy array representing the image in RGB format.
        
    Raises:
        FileNotFoundError: If the image file cannot be found or opened.
    """
    # Read image in BGR format (OpenCV default)
    bgr_img = cv2.imread(img_path)
    
    if bgr_img is None:
        raise FileNotFoundError(f"Could not open or find the image: {img_path}")
    
    # Convert from BGR to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    return rgb_img


def load_image(image_path: str) -> Tuple[Image.Image, torch.Tensor]:
    """Load an image and convert to tensor using the transform function.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Tuple[Image.Image, torch.Tensor]: (original_img, img_tensor) - PIL Image and transformed tensor
        
    Raises:
        FileNotFoundError: If the image file cannot be found or opened.
    """
    # Load image as RGB numpy array
    img_np = get_image_rgb(image_path)
    
    # Keep a copy of the original image for visualization
    original_img = Image.fromarray(img_np)
    
    # Apply the transformation
    img_tensor = transform_image(img_np)
    
    return original_img, img_tensor


def prepare_image_for_inference(
    image_input: Union[str, Image.Image, np.ndarray]
) -> Tuple[Image.Image, torch.Tensor, str]:
    """Prepare different types of image inputs for inference.
    
    Args:
        image_input: Image input (file path, PIL Image, or numpy array)
        
    Returns:
        Tuple[Image.Image, torch.Tensor, str]: (original_img, img_tensor, basename)
        
    Raises:
        ValueError: If the image input type is not supported.
    """
    import os
    
    if isinstance(image_input, str):
        # File path input
        original_img, img_tensor = load_image(image_input)
        image_basename = os.path.splitext(os.path.basename(image_input))[0]
    elif isinstance(image_input, Image.Image):
        # PIL Image input from Gradio
        original_img = image_input
        # Convert PIL to numpy array for transform_image
        img_np = np.array(original_img)
        img_tensor = transform_image(img_np)
        image_basename = "gradio_input"
    elif isinstance(image_input, np.ndarray):
        # Numpy array input from Gradio
        img_np = image_input
        original_img = Image.fromarray(img_np)
        img_tensor = transform_image(img_np)
        image_basename = "gradio_input"
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    return original_img, img_tensor, image_basename


def convert_tensor_to_pil(output_img: torch.Tensor) -> Image.Image:
    """Convert a tensor image to PIL Image for display.
    
    Args:
        output_img: Image tensor (can be in various formats)
        
    Returns:
        Image.Image: PIL Image ready for display
    """
    # Convert tensor to numpy
    if isinstance(output_img, torch.Tensor):
        output_img = output_img.cpu()
        
        # If channel-first format (C,H,W), rearrange to (H,W,C)
        if len(output_img.shape) == 3 and output_img.shape[0] == 3:
            output_img = output_img.permute(1, 2, 0)
        
        # Convert to numpy
        output_np = output_img.numpy()
        
        # Ensure values are in [0, 255] range for PIL
        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
        else:
            output_np = output_np.astype(np.uint8)
    else:
        # If it's already a PIL Image or numpy array
        output_np = np.array(output_img)
        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(output_np)


def validate_image_format(image: Union[str, Image.Image, np.ndarray]) -> bool:
    """Validate if the image format is supported.
    
    Args:
        image: Image to validate
        
    Returns:
        bool: True if the format is supported, False otherwise
    """
    if isinstance(image, str):
        # Check if file exists and has valid extension
        import os
        if not os.path.exists(image):
            return False
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        return image.lower().endswith(valid_extensions)
    elif isinstance(image, (Image.Image, np.ndarray)):
        return True
    else:
        return False


def get_image_info(image: Union[str, Image.Image, np.ndarray]) -> dict:
    """Get information about an image.
    
    Args:
        image: Image to get info from
        
    Returns:
        dict: Dictionary containing image information
    """
    info = {}
    
    if isinstance(image, str):
        # File path
        import os
        info['source'] = 'file'
        info['path'] = image
        info['filename'] = os.path.basename(image)
        info['size'] = os.path.getsize(image) if os.path.exists(image) else 0
        
        # Try to get image dimensions
        try:
            with Image.open(image) as img:
                info['dimensions'] = img.size
                info['mode'] = img.mode
                info['format'] = img.format
        except Exception:
            info['dimensions'] = 'Unknown'
            info['mode'] = 'Unknown'
            info['format'] = 'Unknown'
            
    elif isinstance(image, Image.Image):
        info['source'] = 'pil'
        info['dimensions'] = image.size
        info['mode'] = image.mode
        info['format'] = image.format
        
    elif isinstance(image, np.ndarray):
        info['source'] = 'numpy'
        info['dimensions'] = (image.shape[1], image.shape[0]) if len(image.shape) >= 2 else image.shape
        info['shape'] = image.shape
        info['dtype'] = str(image.dtype)
        
    return info
