import numpy as np

# Define the class names
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

# Create a list of class names in order
CLASSES_LIST = list(CLASSES.values())

# Assign a color to each class
COLORS = {}
for i, class_name in enumerate(CLASSES):
    COLORS[class_name] = tuple([int(c) for c in np.random.randint(0, 255, 3)])