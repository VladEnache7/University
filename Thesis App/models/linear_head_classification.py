from __future__ import annotations

import torch
from omegaconf import DictConfig, ListConfig


class LiniarHeadClassification(torch.nn.Module):
    """
    A simple linear head classification with just a linear layer.
    """

    num_classes: int
    max_objects: int
    received_embed_dim: int
    out_channels: int
    classifier: torch.nn.Module
    bbox: torch.nn.Module

    def __init__(self, received_embed_dim: int, num_objects: int, num_classes: int, out_channels: int) -> None:
        """
        A simple implementation of a classification head.

        This class is intended to be used as a head for a neural network that receives
        embedded features and outputs logits for classification tasks.
          It extends the
        `HeadInterface`.

        Args:
            received_embed_dim (int): The dimension of the input embeddings/features that the
                head will receive from the backbone of the model (e.g., output of a transformer).
            num_objects (int): The maximum number of objects (or tokens) the model is expected to predict.
            num_classes (int): The number of classes in the classification task. This defines
                the size of the output layer, where each output corresponds to a class.
            out_channels (int): The number of output channels for the final linear layer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_objects = num_objects
        self.received_embed_dim = received_embed_dim
        self.out_channels = out_channels

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=received_embed_dim,
                out_features=num_objects * (num_classes + 1)
            )
        )
        self.bbox = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=received_embed_dim,
                out_features=num_objects * self.out_channels
            )
        )


    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        forward function for a simple head classifier
        Args:
            inputs: the outputs of the backbone
        Returns:
            the outputs of the head - with the dimension: num_classes * num_objects
        """
        pred_logits = self.classifier(inputs)

        reshaped_logits = pred_logits.reshape(-1,
                                              self.max_objects, (self.num_classes + 1))

        reshaped_boxes = self.bbox(inputs).sigmoid().reshape(
            -1, self.max_objects, self.out_channels)

        return {
            "pred_logits": reshaped_logits,
            "pred_boxes": reshaped_boxes
        }

def build_linear_head(num_objects: int, num_classes: int, args: DictConfig | ListConfig) -> torch.nn.Module:
    model = LiniarHeadClassification(
        num_classes=num_classes,
        num_objects=num_objects,
        received_embed_dim=args.num_channels,
        out_channels=args.out_channels
    )

    return model
