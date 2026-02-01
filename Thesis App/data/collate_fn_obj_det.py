from typing import Tuple

import torch

from utils.classes_to_predict_object_detection import CLASS_TO_IDX_NOT_ARROWS


class ObjectDetectionCollateFn:
    """
    Collate function for creating dataloader from the ObjectDetecionDataset
    """

    def __init__(self, max_objects, device) -> None:
        """
        Args:
            max_objects: number of maximum object detected by the model for padding
        """
        self.max_objects = max_objects
        self.device = device

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        the collate_function that takes all the information and returns the image and labels
        Args:
            batch: the batch of data from the dataset
        Returns:
            a tuple containing:
                the image 
                a list of max_objects length with the class index, x_coord and y_coord for each point
        """
        return self._custom_collate_fn(batch)

    def _custom_collate_fn(self, batch):
        images, final_labels = [], []

        for sample in batch:
            image = torch.tensor(sample[0], dtype=torch.float).permute(2, 0, 1)
            images.append(image)
            height, width = image.shape[1:]

            labels = torch.tensor([CLASS_TO_IDX_NOT_ARROWS[obj["class"]]
                                   for obj in sample[1]]).to(device=self.device)

            boxes = torch.tensor([[obj["bbox"][0]/width, obj["bbox"][1]/height,
                                   obj["bbox"][2]/width, obj["bbox"][3]/height]
                                  for obj in sample[1]]).to(device=self.device)

            final_labels.append({"labels": labels, "boxes": boxes})

        final_images = torch.stack(images, dim=0).to(device=self.device)
        return final_images, final_labels
