from __future__ import annotations
import torch
from torch import Tensor
from .classes import CLASSES_LIST


import warnings
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[list[str]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Draws bounding boxes on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].

    """
    import torchvision.transforms.functional as F  # noqa

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[list[str], list[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    # <-------------------------------------------> Font <------------------------------------------->

    # Use load_default() instead of truetype without a font path
    txt_font = ImageFont.load_default()
    
    # Attempt to get a larger font if available on the system
    try:
        # Try to use a system font if available
        system_fonts = ['DejaVuSans.ttf', 'Arial.ttf', 'Verdana.ttf']
        for font_name in system_fonts:
            try:
                txt_font = ImageFont.truetype(font_name, size=font_size or 18)
                break  # Use the first available font
            except (IOError, OSError):
                continue
    except Exception:
        # If any error occurs, fall back to default
        pass

    img_to_draw = F.to_pil_image(image)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, label in zip(img_boxes, labels):  # type: ignore[arg-type]
    
        draw.rectangle(bbox, width=width, outline=(255, 0, 0))

        if label is not None:
            # Center text above the box
            text_x = bbox[0]
            text_y = bbox[1] - 20  # 5 pixels above the box
            
            # Make sure text doesn't go outside the image boundaries
            text_y = max(0, text_y)
            draw.text((text_x, text_y), label, fill=(255, 0, 0), font=txt_font)  # type: ignore[arg-type]

    out = F.to_tensor(img_to_draw)
    # F.to_tensor in v1 returns values in range [0, 1], convert to uint8 if needed
    if out.max() <= 1.0:
        out = (out * 255).to(torch.uint8)
    return out

def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes

def draw_image_bbox(
    image: torch.Tensor | Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    width: int,
    height: int,
    threshold: float = 0.5,
    no_object_label_index: int = 10
): # -> torch.Tensor:
    """
    Draws bounding boxes on an image after filtering them based on scores and class labels.
    Args:
        image: The initial image as a torch.Tensor (e.g., in (C, H, W) format, dtype uint8).
        boxes: Tensor of shape [N, 4] in normalized cxcywh format 
               (center_x, center_y, width, height).
        scores: Tensor of shape [N] with confidence scores for each box.
        class_ids: Tensor of shape [N] with class indices for each box.
        width: The width of the image, used for denormalizing box coordinates.
        height: The height of the image, used for denormalizing box coordinates.
        threshold: Confidence threshold for filtering predictions. Predictions with
                   scores below this threshold will be ignored.
        no_object_label_index: Class index representing "no-object" or background,
                               which will be filtered out.
    Returns:
        A torch.Tensor representing the image with the filtered and drawn bounding boxes.
    """
    # Convert PIL Image to tensor if needed
    if isinstance(image, Image.Image):
        # Using v1 transforms.functional
        import torchvision.transforms.functional as F
        image = F.to_tensor(image)
        # F.to_tensor in v1 returns values in range [0, 1], convert to uint8 if needed
        if image.max() <= 1.0:
            image = (image * 255).to(torch.uint8)


    # 1. Filter by score threshold
    keep_by_score = scores > threshold
    
    filtered_boxes_stage1 = boxes[keep_by_score]
    filtered_class_ids_stage1 = class_ids[keep_by_score]
    filtered_scores_stage1 = scores[keep_by_score]

    # If no boxes remain after score filtering, return the original image
    if filtered_boxes_stage1.numel() == 0:
        return image.cpu() # Or simply image, if it's already on CPU / appropriate device

    # 2. Filter out "no-objects"
    keep_objects = filtered_class_ids_stage1 != no_object_label_index
    
    final_boxes = filtered_boxes_stage1[keep_objects]
    final_class_ids = filtered_class_ids_stage1[keep_objects]
    final_scores = filtered_scores_stage1[keep_objects]

    # 3. Handle empty boxes after all filtering
    if final_boxes.numel() == 0 or final_boxes.shape[0] == 0:
        return image.cpu() # Or simply image

    # 4. Rescale and convert box coordinates
    # Clone to avoid in-place modification of the input tensor if it's used elsewhere
    processed_boxes = final_boxes.clone()
    
    # Convert from [center_x, center_y, width, height] to [x_min, y_min, x_max, y_max]
    # This typically uses a function like torchvision.ops.box_convert
    processed_boxes = _box_cxcywh_to_xyxy(processed_boxes)

    # 5. Create labels for the boxes
    # Uses formatting similar to visualize_predictions (score to .2f)
    box_labels = [
        f"{CLASSES_LIST[int(class_id.item())]}: {score.item():.2f}"
        for class_id, score in zip(final_class_ids, final_scores)
    ]

    # 6. Draw bounding boxes on the image
    # draw_bounding_boxes usually expects image as (C, H, W) tensor of type uint8
    return draw_bounding_boxes(image, processed_boxes, box_labels), box_labels, processed_boxes
