from typing import Dict, Tuple

import torch
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision


class ObjectDetectionAccuracy(Metric):
    def __init__(self, matcher: torch.nn.Module) -> None:
        """
        computes the mean average precision for object detection task
        Args:
            matcher: for matching the outputs with the targets
        """
        super().__init__()
        self.map = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh", iou_thresholds=torch.linspace(
            0.1, 0.95, round((0.95 - 0.1) / 0.05) + 1).tolist())
        # self.map = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh")
        self.matcher = matcher

    def update(self, outputs: Dict[str, torch.tensor], targets: Dict[str, torch.tensor]) -> None:
        """ 
        This performs the loss computation.
        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted point coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target point coordinates

        """
        indices = self.matcher(outputs, targets)

        src_logits = outputs['pred_logits']
        src_boxes = outputs['pred_boxes']
        final_preds = []
        final_targets = []
        for i in range(src_logits.shape[0]):
            scores, indexes = src_logits[i][indices[i][0]].softmax(
                dim=1).max(dim=1)
            # print(indexes.to(dtype=torch.int) ==
            #       targets[i]["labels"][indices[i][1]].to(dtype=torch.int))
            final_pred_boxes = src_boxes[i][indices[i][0]]
            final_preds.append({
                "labels": indexes.to(dtype=torch.int),
                "boxes": final_pred_boxes,
                "scores": scores
            })
            final_targets_boxes = targets[i]["boxes"][indices[i][1]]
            final_targets.append({
                "labels": targets[i]["labels"][indices[i][1]].to(dtype=torch.int),
                "boxes": final_targets_boxes,
            })
        self.map.update(preds=final_preds, target=final_targets)

    def compute(self, more: bool = False) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
        """
        Computes and returns the map
        Args:
            more:   if set to true, returns the entire dict
                    else, returns just a tensor with a single value representing the map
        Returns:
            dict of tensors with one element for each map or just one tensor
        """
        if more:
            return self.map.compute()
        return self.map.compute()["map"]

    def reset(self):
        self.map.reset()
