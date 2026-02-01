from data import ObjectDetectionDataset, object_detection_general_markings_dataset


TASK_MAP_DS = {
    'object_detection': ObjectDetectionDataset,
    'object_detection_no_arrows': object_detection_general_markings_dataset
}
