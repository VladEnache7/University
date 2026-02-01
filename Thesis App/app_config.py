
"""Application configuration and constants."""

import logging
from datetime import datetime

# Set up logging configuration
def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        filename='app.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Define the classes for road arrow markings
CLASSES = {
    "marking_arrow_180_deg": "180",
    "marking_arrow_left": "left",
    "marking_arrow_left_right": "left-right",
    "marking_arrow_right": "right",
    "marking_arrow_slight_left": "s-left",
    "marking_arrow_slight_right": "s-right",
    "marking_arrow_straight_left_right": "slr",
    "marking_arrow_straight": "str",
    "marking_arrow_straight_left": "str-left",
    "marking_arrow_straight_right": "str-right",
}

# Create class options for the UI (using the short names for better display)
CLASS_OPTIONS = list(CLASSES.values())

# UI Configuration
UI_CONFIG = {
    'title': "ArrowVision: Precise Road Arrow Marking Detection with Deformable DETR",
    'description': "Upload an image to detect road markings and export results.",
    'confidence_min': 0.1,
    'confidence_max': 0.9,
    'confidence_default': 0.5,
    'confidence_step': 0.05,
    'input_image_height': 300,
    'output_image_height': 300,
    'max_dataframe_rows': 20,
    'summary_lines': 3,
    'export_status_lines': 4
}

# Export configuration
EXPORT_CONFIG = {
    'formats': ["JSON", "CSV"],
    'default_format': "JSON",
    'timestamp_format': "%Y%m%d_%H%M%S",
    'display_timestamp_format': "%Y-%m-%d %H:%M:%S"
}

# Example images configuration
EXAMPLES_CONFIG = {
    'directory': "examples",
    'supported_extensions': ('.png', '.jpg', '.jpeg')
}

# DataFrame column configuration
DATAFRAME_COLUMNS = {
    'headers': ["#", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"],
    'datatypes': ["number", "str", "str", "str", "str", "str", "str", "str", "str"]
}

# Common class selections
COMMON_CLASSES = ["str", "left", "right", "str-left", "str-right"]

# Global variable structure for storing results
def create_empty_results():
    """Create empty results structure."""
    return {
        'box_labels': [],
        'processed_boxes': [],
        'image_info': {},
        'timestamp': None,
        'threshold': None,
        'selected_classes': []
    }
