# pylint: disable=W0621,R0801
import os
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import distributed as dist

from data import DataloaderFactory, build_object_detection_dataset


@pytest.fixture(scope="module")
def root_dir() -> Path:
    return Path(__file__).parents[2]


@pytest.fixture(scope="module")
def path_to_config(root_dir: Path) -> Path:
    return root_dir / "src" / "test" / "configs" / "test_config_obj_det.yml"


@pytest.fixture(scope="module")
def config_file(path_to_config: Path, root_dir: Path) -> DictConfig | ListConfig:
    read_file = OmegaConf.load(path_to_config)
    # modify the root_dir in the config file
    read_file.data_loader.params.annotation_file = str(
        root_dir) + "/" + read_file.data_loader.params.annotation_file
    read_file.data_loader.params.root_path_jsons = str(
        root_dir) + "/" + read_file.data_loader.params.root_path_jsons
    read_file.data_loader.params.current_root_path_images = str(
        root_dir) + "/" + read_file.data_loader.params.current_root_path_images
    return read_file


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module")
def image_dim() -> torch.Size:
    return torch.Size([3, 512, 1664])


@pytest.fixture(scope="module")
def class_number() -> int:
    return 12


def initialize_for_distributed():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl")


def destroy_for_distributed():
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "dataloader_type, dataset_size",
    [
        ("train", 4),
        ("val", 1),
    ],
)
def test_dataset(config_file: DictConfig | ListConfig,
                 dataloader_type: str,
                 image_dim: torch.Size,
                 dataset_size: int) -> None:
    dataset = build_object_detection_dataset(config_file, dataloader_type)
    assert len(dataset) == dataset_size, "The dataset should be this big"
    for _, (input_img, label) in enumerate(dataset):
        assert isinstance(input_img, torch.Tensor)
        assert input_img.shape == image_dim, "the image in the correct format"
        assert len(label) >= 1, "At least one object in the image"
        assert isinstance(label, list)
        for sign in label:
            assert "bbox" in sign, "we have the bbox of the sign"
            assert "class" in sign, "the class are in the sign"
            assert isinstance(sign["bbox"], list)
            assert isinstance(sign["class"], str)


@pytest.mark.parametrize(
    "dataloader_type, dataset_size, batch_size, distributed",
    [
        ("train", 4, 2, False),
        ("val", 1, 1, False),
        ("train", 4, 2, True),
        ("val", 1, 1, True),
    ],
)
def test_dataloader(config_file: DictConfig | ListConfig, dataloader_type: str,
                    image_dim: torch.Size, dataset_size: int,
                    batch_size: int, device: torch.device,
                    class_number: int, distributed: bool) -> None:
    if distributed:
        initialize_for_distributed()
    dataloader = DataloaderFactory.get_dataloader(
        config_file, dataloader_type, device, distributed)
    for batch, (x_data, y_data) in enumerate(dataloader):
        assert isinstance(
            x_data, torch.Tensor), "The images should be a Tensor"
        assert x_data.shape[0] == batch_size, "the number of images should be equal to the batch_size"
        assert x_data.shape[1:] == image_dim, "the shape of the images should be the same"
        assert x_data.device == device, "The device should be correct"
        assert batch < dataset_size // batch_size, "The number of batches should not be greater than the calculated one"
        assert isinstance(y_data, list)
        assert len(
            y_data) == batch_size, "The number of labels should be equal to the batch_size"
        for label in y_data:
            assert isinstance(label, dict), "The labels should be a dict"
            assert "labels" in label, "The labels should be present - the classes"
            assert "boxes" in label, "the boxes should be present"
            classes = label["labels"]
            boxes = label["boxes"]
            assert isinstance(
                classes, torch.Tensor), "The classes should be a Tensor"
            assert isinstance(
                boxes, torch.Tensor), "The coords should be a Tensor"
            assert classes.device == device, "The classes should be on the correct device"
            assert boxes.device == device, "The boxes should be on the correct device"
            assert classes.max().item(
            ) < class_number, "The classes indexes should be smaller than the number of them"
            assert classes.min().item() >= 0, "The classes indexes should be greater or equal to 0"
            assert boxes.max().item() <= 1, "The coords should be normalized -> smaller than 1"
            assert boxes.min().item() >= 0, "The coords should be normalized -> greater than 0"
    if distributed:
        destroy_for_distributed()


def test_wrong_dataloader(config_file: DictConfig | ListConfig, device: torch.device) -> None:
    try:
        _ = DataloaderFactory.get_dataloader(
            config_file, "wrong_type", device, False)
        assert False
    except ValueError as ve:
        assert str(ve) == "Dataset type is not one of train, val, test!"
