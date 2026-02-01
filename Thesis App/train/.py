import os
from pprint import pprint

import torch
from omegaconf import OmegaConf
from torch import distributed as dist
from torchinfo import summary

from training import TrainerFactory


def main():
    dist.init_process_group(backend="nccl")
    device = torch.device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device=device)
    dist.barrier()

    print(f'Current device: {torch.cuda.current_device()}')

    path_to_config = "src/configs/train_config_obj_det.yml"
    omega_conf = OmegaConf.load(path_to_config)

    trainer = TrainerFactory.get_trainer(omega_conf, device=device)

    if device.index == 0:
        print(trainer.model)
        summary(trainer.model)

    results = trainer.train(5000)

    if device.index == 0:
        pprint(results)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


# command: torchrun --standalone --nproc-per-node=gpu <filename>
