import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import distributed as dist

from training import Trainer

# pylint: disable=W0621


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Derives the root directory of the FM-PERCEPTION git repository, independently on the working directory."""
    return Path(__file__).parents[2]


@pytest.fixture(scope="module")
def config(repo_root: Path):
    """Load configuration for the dataloader from the YAML file."""
    file_path = repo_root / "src" / "test" / "configs" / "test_config_obj_det.yml"
    return OmegaConf.load(file_=file_path)


@pytest.fixture(scope="module")
def device():
    """Fixture to determine the device for the tests."""
    return torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module")
def temp_output_dir():
    """Creates a temporary directory for saving outputs, models, and logs."""
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="module")
def trainer(config: DictConfig | ListConfig, device: torch.device, temp_output_dir) -> Trainer:
    """Initializes the Trainer with the provided configuration and device."""
    return Trainer(
        config=config,
        device=device,
        load_state_path=None,
        freeze_backbone=True,
        distributed=False,
        output_dir=str(temp_output_dir),
        save_steps=1,
        logs=True,
        log_dir=str(temp_output_dir)
    )


def test_model_saving(trainer: Trainer, temp_output_dir):
    """Tests that the model is saved correctly at the end of training."""
    trainer.train(epochs=1)
    saved_files = list(temp_output_dir.glob('*.pth'))
    assert len(
        saved_files) > 0, "Model checkpoint was not saved. No .pth files found in the output directory."


def initialize_for_distributed():
    # Set environment variables for PyTorch distributed training
    os.environ['RANK'] = '0'  # Rank of the current process
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['MASTER_ADDR'] = 'localhost'  # Address of the master node
    os.environ['MASTER_PORT'] = '12355'  # Port used for communication
    dist.init_process_group(backend="nccl")


@pytest.mark.parametrize(
    "distributed",
    [
        (True),
        (False)
    ]
)
def test_train(config: DictConfig | ListConfig, device: torch.device, distributed, temp_output_dir):
    """Tests that the model is training having data distributed."""
    if distributed is True:
        initialize_for_distributed()
    trainer = Trainer(
        config=config,
        device=device,
        load_state_path=None,
        freeze_backbone=True,
        distributed=distributed,
        output_dir=str(temp_output_dir),
        save_steps=1
    )
    results = trainer.train(epochs=2)

    assert 'train_loss' in results and len(
        results['train_loss']) == 2, "Expected 'train_loss' in results and length of 2"
    assert 'val_loss' in results and len(
        results['val_loss']) == 2, "Expected 'val_loss' in results and length of 2"
    assert 'train_mAP' in results and len(
        results['train_mAP']) == 2, "Expected 'train_mAP' in results and length of 2"
    assert 'val_mAP' in results and len(
        results['val_mAP']) == 2, "Expected 'val_mAP' in results and length of 2"

    assert all(0 <= val <= 1 for val in results['train_mAP']
               ), f"Train mAP values out of expected range: {results['train_mAP']}"
    assert all(0 <= val <= 1 for val in results['val_mAP']
               ), f"Validation mAP values out of expected range: {results['val_mAP']}"
