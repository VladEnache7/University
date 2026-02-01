from __future__ import annotations
import abc

import torch
from omegaconf import DictConfig, ListConfig

# from .backbone_vit import BackboneViT


class BackboneFactory(metaclass=abc.ABCMeta):
    """
    Factory to create a backbone implementation 
    """
    @staticmethod
    def get_backbone(config: DictConfig | ListConfig, given_device: torch.device) -> torch.nn.Module:
        """
        based on the configuration file, it returns the specified backbone model
        Args:
            config: the configuration file
            device: on which device to load the model and possibly the weights
        Returns:
            an implementation for a BackboneInteface
        # """

        # encoder = BackboneViT(config.model.backbone.name)
        # encoder.to(given_device)
        # weigts_path = config.model.backbone.weights
        # if weigts_path is not None:
        #     loaded_state_dict = torch.load(weigts_path, map_location=given_device)
        #     encoder.vit.load_state_dict(loaded_state_dict)
        # return encoder
