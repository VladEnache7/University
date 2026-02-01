import torch
from omegaconf import OmegaConf

from losses import HungarianMatcher
from metrics.metrics_constants import TASK_METRIC
from torchmetrics import Metric


class MetricFactory:
    """
    The Factory for Metrics
    """
    def __new__(cls):
        raise TypeError(
            f"Cannot instantiate {cls.__name__} as it is a static class.")

    @staticmethod
    def get_metric(config: OmegaConf) -> Metric:
        """
        Creates a loss function according to the config info
        Args:
            config: the config information from the config file
        Returns:
            a metric function implementation
        """
        metric = TASK_METRIC[config.metric.name](
            matcher=HungarianMatcher(**config.matcher.params)
        )
        return metric
