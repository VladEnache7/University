# pylint: disable=W0621
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from models import BackboneFactory, HeadFactory, SingleTaskModel


@pytest.fixture(scope="module")
def root_dir() -> Path:
    return Path(__file__).parents[2]


@pytest.fixture(scope="module")
def path_to_config(root_dir: Path) -> Path:
    return root_dir / "src" / "test" / "configs" / "test_model_config_obj_det.yaml"


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module")
def config_file(path_to_config: Path) -> DictConfig | ListConfig:
    read_file = OmegaConf.load(path_to_config)
    return read_file


@pytest.mark.parametrize(
    "backbone, channels",
    [
        ("vit_small", 384),
        ("vit_base", 768),
        ("vit_large", 1024),
        ("vit_giant", 1536),
    ],
)
def test_inference(config_file: DictConfig | ListConfig, device: torch.device,
                   backbone: str, channels: int):
    num_objects = config_file.model.num_objects
    num_classes = config_file.classes.num_classes + 1
    config_file.model.backbone.name = backbone
    config_file.model.head.num_channels = channels
    backbone_part = BackboneFactory.get_backbone(config_file, device)
    head_part = HeadFactory.get_head(config_file)
    model = SingleTaskModel(backbone_part, head_part).to(device)
    input_data = torch.rand(1, 3, 512, 1664).to(device)
    model_output = model(input_data)
    assert isinstance(model_output, dict)
    assert 'pred_logits' in model_output
    assert 'pred_boxes' in model_output
    logits = model_output['pred_logits']
    boxes = model_output['pred_boxes']
    assert logits.shape == torch.Size([1, num_objects, num_classes])
    assert boxes.shape == torch.Size([1, num_objects, 4])
    assert 0 <= boxes.all() <= 1, "These should be normalized"
