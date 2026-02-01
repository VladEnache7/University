import torch
from omegaconf import OmegaConf

from data.collate_fn_obj_det import ObjectDetectionCollateFn
from data.data_constants import TASK_MAP_DS


class DataloaderFactory:
    """
    The factory for creating a dataloader
    """
    def __new__(cls):
        raise TypeError(
            f"Cannot instantiate {cls.__name__} as it is a static class.")

    @staticmethod
    def get_dataloader(config: OmegaConf, dataloader_type: str, device: torch.device) -> torch.utils.data.DataLoader:
        """
        the creationg of a dataloader based on the config info and the type
        Args:
            config: the config information from the config file
            dataloader_type: a value from 'train', 'val' and 'test'
        Returns:
            A dataloader
        """
        dataset = TASK_MAP_DS[config.data_loader.task](
            **config.data_loader.params,
            data_type=dataloader_type
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data_loader.batch_size,
            collate_fn=ObjectDetectionCollateFn(
                max_objects=config.model.num_objects, device=device)
        )
