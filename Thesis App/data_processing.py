
"""Data processing utilities for detection results."""

import json
import csv
import torch
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Union
from app_config import EXPORT_CONFIG, DATAFRAME_COLUMNS, CLASS_OPTIONS


def filter_detections_by_class(
    box_labels: List[str], 
    processed_boxes: List[torch.Tensor], 
    selected_classes: List[str]
) -> Tuple[List[str], List[torch.Tensor]]:
    """Filter detection results based on selected classes.
    
    Args:
        box_labels: List of detection labels
        processed_boxes: List of detection boxes
        selected_classes: List of selected class names
        
    Returns:
        Tuple of filtered labels and boxes
    """
    if not selected_classes:  # If no classes selected, return all
        return box_labels, processed_boxes
    
    filtered_labels = []
    filtered_boxes = []
    
    for label, box in zip(box_labels, processed_boxes):
        # Extract class name from label (format: "class_name: confidence")
        class_name = label.split(':')[0] if ':' in label else label
        
        # Check if this class is in the selected classes
        if class_name in selected_classes:
            filtered_labels.append(label)
            filtered_boxes.append(box)
    
    return filtered_labels, filtered_boxes


def format_detection_dataframe(
    box_labels: List[str], 
    processed_boxes: List[torch.Tensor]
) -> pd.DataFrame:
    """Create a pandas DataFrame for detection results.
    
    Args:
        box_labels: List of detection labels
        processed_boxes: List of detection boxes
        
    Returns:
        Formatted DataFrame with detection results
    """
    if len(box_labels) == 0:
        # Return empty DataFrame with column headers
        return pd.DataFrame(columns=DATAFRAME_COLUMNS['headers'])
    
    data = []
    
    for i, (label, box) in enumerate(zip(box_labels, processed_boxes)):
        if isinstance(box, torch.Tensor):
            x1, y1, x2, y2 = box.tolist()
        else:
            x1, y1, x2, y2 = box
        
        # Extract class name and confidence from label
        if ':' in label:
            class_name, conf_str = label.split(': ')
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.0
        else:
            class_name = label
            confidence = 0.0
        
        width = x2 - x1
        height = y2 - y1
        
        data.append([
            i + 1,                    # #
            class_name,               # Class
            f"{confidence:.3f}",      # Confidence
            f"{x1:.0f}",             # X1
            f"{y1:.0f}",             # Y1
            f"{x2:.0f}",             # X2
            f"{y2:.0f}",             # Y2
            f"{width:.0f}",          # Width
            f"{height:.0f}"          # Height
        ])
    
    df = pd.DataFrame(data, columns=DATAFRAME_COLUMNS['headers'])
    return df


def get_summary_text(
    box_labels: List[str], 
    processed_boxes: List[torch.Tensor], 
    selected_classes: List[str]
) -> str:
    """Generate summary text for filtering information.
    
    Args:
        box_labels: List of detection labels
        processed_boxes: List of detection boxes
        selected_classes: List of selected class names
        
    Returns:
        Summary text string
    """
    summary = f"📊 Total detections: {len(box_labels)}"
    
    if selected_classes and len(selected_classes) < len(CLASS_OPTIONS):
        summary += f"\n🔍 Filter applied - Showing only: {', '.join(selected_classes)}"
    
    return summary


def export_results(
    box_labels: List[str], 
    processed_boxes: List[torch.Tensor], 
    format_type: str = "json"
) -> str:
    """Export detection results to JSON or CSV format.
    
    Args:
        box_labels: List of detection labels
        processed_boxes: List of detection boxes
        format_type: Export format ("json" or "csv")
        
    Returns:
        Filename of the exported file
    """
    timestamp = datetime.now().strftime(EXPORT_CONFIG['timestamp_format'])
    
    if format_type.lower() == "json":
        data = {
            "timestamp": timestamp,
            "total_detections": len(box_labels),
            "detections": []
        }
        
        for i, (label, box) in enumerate(zip(box_labels, processed_boxes)):
            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = box
                
            data["detections"].append({
                "id": i + 1,
                "class": label.split(':')[0] if ':' in label else label,
                "confidence": float(label.split(':')[1]) if ':' in label else 0.0,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "width": x2 - x1,
                "height": y2 - y1
            })
        
        filename = f"detections_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    elif format_type.lower() == "csv":
        filename = f"detections_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height'])
            
            for i, (label, box) in enumerate(zip(box_labels, processed_boxes)):
                if isinstance(box, torch.Tensor):
                    x1, y1, x2, y2 = box.tolist()
                else:
                    x1, y1, x2, y2 = box
                    
                class_name = label.split(':')[0] if ':' in label else label
                confidence = float(label.split(':')[1]) if ':' in label else 0.0
                
                writer.writerow([i+1, class_name, confidence, x1, y1, x2, y2, x2-x1, y2-y1])
        
        return filename
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def create_export_message(filename: str, total_detections: int) -> str:
    """Create export success message.
    
    Args:
        filename: Name of the exported file
        total_detections: Number of detections exported
        
    Returns:
        Formatted success message
    """
    timestamp = datetime.now().strftime(EXPORT_CONFIG['display_timestamp_format'])
    return (
        f"✅ Results exported successfully!\n"
        f"📁 File: {filename}\n"
        f"🕒 Time: {timestamp}\n"
        f"📊 Total detections: {total_detections}"
    )
