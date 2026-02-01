import torch
from omegaconf import DictConfig, ListConfig

from .optimizers_constants import TASK_OPT, TASK_SCD


class OptimizerFactory:

    def __new__(cls):
        raise TypeError(
            f"Cannot instantiate {cls.__name__} as it is a static class.")

    @staticmethod
    def get_optimizer_and_scheduler(
        config: DictConfig | ListConfig,
        parameters
        ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Creates an optimizer according to the config info
        Args:
            config: the config information from the config file
            parameters: the model's parameters
        Returns:
            a torch optimizer
        """
        optimizer = TASK_OPT[config.optimizer.name](
            parameters,
            **config.optimizer.params,
        )
        scheduler = TASK_SCD[config.scheduler.name](
            optimizer,
            **config.scheduler.params,
        )
        return optimizer, scheduler
