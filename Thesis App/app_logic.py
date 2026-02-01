
"""Application business logic and processing functions."""

import logging
from datetime import datetime
from typing import Tuple, List, Optional, Union
import pandas as pd
from PIL import Image
import numpy as np

from data_processing import (
    filter_detections_by_class, 
    format_detection_dataframe, 
    get_summary_text, 
    export_results, 
    create_export_message
)
from app_config import create_empty_results
from ui_components import validate_image_input


# Global variable to store last detection results for export
last_detection_results = create_empty_results()


def process_image(
    image: Union[Image.Image, np.ndarray, None], 
    threshold: float, 
    selected_classes: List[str],
    model,
    inference_function
) -> Tuple[Optional[Image.Image], pd.DataFrame, str]:
    """Main processing function with result storage for export and class filtering.
    
    Args:
        image: Input image
        threshold: Confidence threshold
        selected_classes: List of selected class names
        model: Trained model
        inference_function: Inference function to use
        
    Returns:
        Tuple of (processed_image, detection_dataframe, summary_text)
    """
    global last_detection_results
    
    logging.info("Entered process_image function")
    
    if not validate_image_input(image):
        logging.info("Image is None")
        empty_df = pd.DataFrame(columns=["#", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"])
        return None, empty_df, "No image provided"
    
    try:
        logging.info(f"Image type: {type(image)}, shape: {getattr(image, 'shape', 'No shape')}")
        logging.info(f"Threshold: {threshold}")
        logging.info(f"Selected classes: {selected_classes}")
        
        # Call the inference function
        result, box_labels, processed_boxes = inference_function(
            model, image, confidence_threshold=threshold
        )
        
        logging.info(f"Inference completed: {len(box_labels)} objects detected before filtering")

        # Filter detections by selected classes
        filtered_labels, filtered_boxes = filter_detections_by_class(
            box_labels, processed_boxes, selected_classes
        )
        
        logging.info(f"After class filtering: {len(filtered_labels)} objects remain")

        # Store results for export (filtered results)
        last_detection_results = {
            'box_labels': filtered_labels,
            'processed_boxes': filtered_boxes,
            'image_info': {
                'size': image.size if hasattr(image, 'size') else 'Unknown',
                'format': image.format if hasattr(image, 'format') else 'Unknown'
            },
            'timestamp': datetime.now().isoformat(),
            'threshold': threshold,
            'selected_classes': selected_classes
        }

        # Create DataFrame for detection results
        detection_df = format_detection_dataframe(filtered_labels, filtered_boxes)
        
        # Create summary text
        summary_text = get_summary_text(filtered_labels, filtered_boxes, selected_classes)
        
        # Add original detection count if filtering was applied
        if selected_classes and len(selected_classes) < len(list(selected_classes)):
            summary_text += f"\nTotal detected (all classes): {len(box_labels)}"
            
        return result, detection_df, summary_text
        
    except Exception as e:
        logging.exception("Error in process_image function")
        empty_df = pd.DataFrame(columns=["#", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"])
        return None, empty_df, f"Error during inference: {str(e)}"


def handle_export(format_type: str) -> Tuple[Optional[str], str]:
    """Handle export button click.
    
    Args:
        format_type: Export format ("json" or "csv")
        
    Returns:
        Tuple of (filename, status_message)
    """
    global last_detection_results
    
    if not last_detection_results['box_labels']:
        return None, "❌ No detection results to export. Please run detection first."
    
    try:
        filename = export_results(
            last_detection_results['box_labels'], 
            last_detection_results['processed_boxes'], 
            format_type
        )
        
        export_message = create_export_message(
            filename, 
            len(last_detection_results['box_labels'])
        )
        
        return filename, export_message
        
    except Exception as e:
        logging.exception("Error during export")
        return None, f"❌ Export failed: {str(e)}"


def get_detection_statistics(box_labels: List[str]) -> dict:
    """Get statistics from detection results.
    
    Args:
        box_labels: List of detection labels
        
    Returns:
        Dictionary with detection statistics
    """
    if not box_labels:
        return {'total': 0, 'classes': {}}
    
    class_counts = {}
    for label in box_labels:
        class_name = label.split(':')[0] if ':' in label else label
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        'total': len(box_labels),
        'classes': class_counts,
        'unique_classes': len(class_counts)
    }
