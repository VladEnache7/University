import torch
from omegaconf import OmegaConf

from .losses_constants import TASK_LOSS, TASK_MATCHER


class LossFactory:
    """
    The Factory for Losses
    """
    def __new__(cls):
        raise TypeError(
            f"Cannot instantiate {cls.__name__} as it is a static class.")

    @staticmethod
    def get_loss(config: OmegaConf) -> torch.nn.Module:
        """
        Creates a loss function according to the config info
        Args:
            config: the config information from the config file
        Returns:
            a loss function implementation
        """
        loss = TASK_LOSS[config.loss.name](
            num_classes=config.classes.num_classes,
            matcher=TASK_MATCHER[config.matcher.name](**config.matcher.params),
            **config.loss.params,
        )
        return loss
