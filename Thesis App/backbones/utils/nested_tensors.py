"""Nested Tensors implementation
Contains convert function from a list of tensors to a NestedTensor
"""
from __future__ import annotations

import torch


class NestedTensor:
    """Nested Tensor for tensors with varying shapes"""

    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor | None = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device, non_blocking: bool = False) -> NestedTensor:
        """Moved the nestedTensor to a device
        Args:
            device: destination device
            non_blocking: type of transfer (async or sync)
        Returns:
            NestedTensor with the tensors and masks moved to other device
        """
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs) -> None:
        """Calls the record_stream on both the tensor and the mask
        Args:
            *args: args to be passed to the function applied on tensor and mask
            **kwargs: kwargs to be passed to the function applied on tensor and mask
        """
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decompose the elements of the tensor
        Returns:
            the tensor and the mask
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        """Representaition function
        Returns:
            the tensor converted to string
        """
        return str(self.tensors)


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: list[torch.Tensor]) -> NestedTensor:
    """Convers a list of tensors to a NestedTensor
    Args:
        tensor_list: the list of tensors to be converted
    Returns:
        Nested Tensors containing the input tensors and the masks for them
    Raises:
        ValueError: not supported
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
