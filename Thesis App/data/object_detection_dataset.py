# External imports
import json
import os
import sys
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

sys.path.append("/home/eni2clj/repos/fm-perception/src")


class ObjectDetectionDataset(Dataset):
    """
    Dataset class for keypoint detection of painted sign arrows
    """

    def __init__(self, annotation_file: str,
                 root_path_jsons: str,
                 linux_paths: bool,
                 data_type: str,
                 transform=None,
                 target_transform=None
                 ):
        """
        Args:
            annotation_file: full path to the .csv file where are all the json files
            root_path_jsons: full path to the directory containing all the json files
            linux_paths: if true the paths to the images are changed to match the ones from linux to abtst
            data_type: value from ('train', 'val' or 'test') for the type of the dataset
        """
        all_data = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform
        self.root_path_jsons = root_path_jsons
        self.data = all_data[all_data["type"] ==
                             data_type].reset_index()
        self.linux = linux_paths

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict[str, Union[List[int], str]]]]:
        """
        Function to a sample from an specific index from the csv
        Args:
            idx: the index from the .csv
        Returns:
            A Dict with the following keys:
                image -> a np.ndarray representing the image
                labels -> A list of Dicts(one Dict for each object).                               
                        "bbox": [x_top_left, y_top_left, width, height]
                        "class": The class name of the object 
        """
        json_path = self.data.loc[idx].at["json_files"]
        full_path_to_json = os.path.join(self.root_path_jsons, json_path)
        return self._get_information_from_json(full_path_to_json)

    def _get_bounding_box_objects(self, data: Dict[str, str]) -> List[Dict[str, Union[List[int], str]]]:
        """
        Implementation to get information about all the polygons objects from the json content
        Args:
            data: a Dict containing all the information from the json file
        Returns: 
            A list of Dicts (one Dict for each object).                               
                "bbox": [x_top_left, y_top_left, width, height]
                "class": The class name of the object

        """
        all_objects = data["content"]["openlabel"]["objects"]
        bbox_list = []
        content = data["content"]["openlabel"]
        frame = content["frames"]["0"]
        frame_objects = frame["objects"]
        for object_id, object_contents in frame_objects.items():
            object_data = object_contents["object_data"]
            if "poly2d" in object_data:
                points = object_data["poly2d"][0]["val"]
                points = [(points[i], points[i + 1])
                          for i in range(0, len(points) - 1, 2)]
                object_type = "None"
                for info in all_objects[object_id]["object_data"]["text"]:
                    if "Type" in info["name"]:
                        object_type = info["val"]

                lowest_x_coord = min(points, key=lambda coord: coord[0])[0]
                lowest_y_coord = min(points, key=lambda coord: coord[1])[1]

                greatest_x_coord = max(points, key=lambda coord: coord[0])[0]
                greatest_y_coord = max(points, key=lambda coord: coord[1])[1]
                center_x = (lowest_x_coord + greatest_x_coord) / 2
                center_y = (lowest_y_coord + greatest_y_coord) / 2
                width = greatest_x_coord - lowest_x_coord
                height = greatest_y_coord - lowest_y_coord

                bbox_list.append({
                    "bbox": [center_x, center_y, width, height],
                    "class": object_type
                })

        return bbox_list

    def _get_image(self, json_content: Dict[str, str]) -> np.ndarray:
        """
        Function to get the image that it's labeled in the json file
        Args:
            json_content: a Dict containing all the information from the json file
        Returns:
            A np.ndarray representing the image
        """
        windows_path_to_image = json_content["content"]["openlabel"][
            "frames"]["0"]["frame_properties"]["streams"]["front"]["uri"]
        final_path = windows_path_to_image
        if self.linux:
            final_path = "/shares/CC_v_Val_FV_Gen3_all" + \
                final_path.split("Val_FV_Gen3\\01_all")[1].replace("\\", "/")
        image = cv2.imread(final_path, -1)
        return image

    def _get_information_from_json(self, file_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Implements the parsing of the json file received
        Args:
            file_path: the path to the json file where is the information stored
        Returns:
            A Tuple with (image, labels)
                image -> a np.array representing the image
                labels -> a list of Dicts (one Dict for each object).                               
                            "bbox": [x_top_left, y_top_left, width, height]
                            "class": The class name
        """
        with open(file_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)
            image_contained = self._get_image(json_content)

            # for the case when there are no objects in the json file
            bounding_box_objects = None
            try:
                bounding_box_objects = self._get_bounding_box_objects(
                    json_content)
            except KeyError:
                bounding_box_objects = []

            return image_contained, bounding_box_objects

    def __len__(self) -> int:
        """
        Returns 
            The number of elements in the dataset
        """
        return len(self.data)
