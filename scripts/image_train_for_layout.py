"""
Train a diffusion model on images.
"""
print("Hi")
import argparse

import torch.distributed as dist
from omegaconf import OmegaConf

from layout_diffusion import dist_util, logger
from layout_diffusion.train_util import TrainLoopWithTensorboard
from layout_diffusion.util import loopy
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.respace import build_diffusion
#from torch.summary import summary

import torch


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)
    print(OmegaConf.to_yaml(cfg))
    
    dist_util.setup_dist(local_rank=cfg.local_rank)
    logger.configure(dir=cfg.train.log_dir)
    logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))

    logger.log("creating model...")
    model = build_model(cfg)
    model.to(dist_util.dev())
    #print(model)
    #summary(model, (1,3, 256, 256))
    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg)

    logger.log("creating schedule sampler...")
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    logger.log("creating data loader...")
    train_loader = build_loaders(cfg, mode='train')

    logger.log("training...")
    # Measure memory usage before each iteration
    start_allocated = torch.cuda.memory_allocated()
    start_cached = torch.cuda.memory_cached()
    print('Memory allocated (MB) after building model :', (start_allocated) / (1024 * 1024))
    print('Memory cached (MB) after building model:', (start_cached) / (1024 * 1024))
    trainer = TrainLoopWithTensorboard(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(train_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        tensorboard_logdir="./tensorboard_logs/no_freezing",  # Specify the actual tensorboard log directory
        **cfg.train
    )

    start_allocated = torch.cuda.memory_allocated()
    start_cached = torch.cuda.memory_cached()
    print('Memory allocated (MB) after building trainer :', (start_allocated) / (1024 * 1024))
    print('Memory cached (MB) after building trainer:', (start_cached) / (1024 * 1024))
    
    trainer.run_loop()



if __name__ == "__main__":
    main()