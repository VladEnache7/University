from __future__ import annotations
import abc

import torch
from omegaconf import DictConfig, ListConfig

from .models_constants import TASK_DEC


class HeadFactory(metaclass=abc.ABCMeta):
    """
    Factory to create a decoder 
    """
    @staticmethod
    def get_head(config: DictConfig | ListConfig) -> torch.nn.Module:
        """
        Creation of a head model based on the config info provided
        Args:
            config: the config information from the config file
        Returns:
            an implementation of HeadInteface
        """
        head = TASK_DEC[config.model.head.name](
            num_objects=config.model.num_objects,
            num_classes=config.classes.num_classes,
            args=config.model.head,
        )
        return head
