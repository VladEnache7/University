
"""Updated inference module using the modular structure."""

from __future__ import annotations

import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Union, List

# Import from our modular structure
from image_utils import load_image, transform_image
from model_utils import initialize_model
from utils import draw_image_bbox


def inference_gradio(
    model: torch.nn.Module, 
    image_input: Union[str, Image.Image, np.ndarray], 
    confidence_threshold: float = 0.5
) -> Tuple[Image.Image, List[str], List[torch.Tensor]]:
    """Modified inference function that handles different input types from Gradio.
    
    Args:
        model: Trained model
        image_input: Image input (file path, PIL Image, or numpy array)
        confidence_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (output_image, box_labels, processed_boxes)
    """
    # Get device
    device = next(model.parameters()).device
    
    # Handle different input types
    if isinstance(image_input, str):
        # File path input (original behavior)
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
    
    print(f"Image tensor size: {img_tensor.size()}")
    
    # Add batch dimension and move to device
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        result = model(img_batch)
        
        outputs_class = result["pred_logits"]
        outputs_coord = result["pred_boxes"]
        
        # Process outputs (last layer from decoder)
        probas = outputs_class[0].softmax(-1)
        keep = probas.max(-1).values > confidence_threshold
        
        # Convert to CPU for visualization
        probas = probas.cpu()
        outputs_coord = outputs_coord[0].cpu()
        
        # Get predicted class indices and scores
        labels = torch.argmax(probas, dim=1)
        scores = probas.max(dim=1).values
        
        # Filter by confidence threshold
        keep_indices = scores > confidence_threshold
        labels = labels[keep_indices]
        scores = scores[keep_indices]
        boxes = outputs_coord[keep_indices]
        
        # Convert relative coordinates to absolute image coordinates
        h, w = original_img.height, original_img.width
        boxes = boxes * torch.tensor([w, h, w, h])
        
        # Visualize the predictions
        output_img, box_labels, processed_boxes = draw_image_bbox(
            image=original_img, 
            boxes=boxes, 
            class_ids=labels, 
            width=w, 
            height=h, 
            scores=scores
        )
        
        # Convert tensor to numpy for Gradio display
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
        
        # Convert back to PIL Image for Gradio
        output_pil = Image.fromarray(output_np)
        
        return output_pil, box_labels, processed_boxes


# Re-export the functions from the modular structure for compatibility
from model_utils import initialize_model
from inference import inference, process_multiple_images
