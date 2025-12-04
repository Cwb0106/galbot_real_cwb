import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from mani_skill.utils import common

# MSHAB imports
from mshab.agents.dp import Agent
from mshab.agents.dp.utils import IterationBasedBatchSampler, worker_init_fn
from mshab.envs.make import make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger
from mshab.utils.time import NonOverlappingTimeProfiler

# Local imports (refactored files)
from config import get_mshab_train_cfg, TrainConfig
from dataset import DPDataset

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_config.yml>")
        sys.exit(1)

    PASSED_CONFIG_PATH = sys.argv[1]
    cfg: TrainConfig = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))

    print("cfg:", cfg, flush=True)

    # 1. Setup Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    # Check Horizon Logic
    assert cfg.algo.obs_horizon + cfg.algo.act_horizon - 1 <= cfg.algo.pred_horizon
    assert (
        cfg.algo.obs_horizon >= 1
        and cfg.algo.act_horizon >= 1
        and cfg.algo.pred_horizon >= 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 3. Setup Agent & Optimizer
    # -------------------------------------------------------------------------
    print("making agent and logger...", flush=True)

    agent = Agent(
        single_observation_space=cfg.algo.state_dim,
        single_action_space=cfg.algo.action_dim,
        obs_horizon=cfg.algo.obs_horizon,
        act_horizon=cfg.algo.act_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        diffusion_step_embed_dim=cfg.algo.diffusion_step_embed_dim,
        unet_dims=cfg.algo.unet_dims,
        n_groups=cfg.algo.n_groups,
        device=device,
    ).to(device)

    optimizer = optim.AdamW(
        params=agent.parameters(),
        lr=cfg.algo.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=cfg.algo.num_iterations,
    )

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(
        single_observation_space=cfg.algo.state_dim,
        single_action_space=cfg.algo.action_dim,
        obs_horizon=cfg.algo.obs_horizon,
        act_horizon=cfg.algo.act_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        diffusion_step_embed_dim=cfg.algo.diffusion_step_embed_dim,
        unet_dims=cfg.algo.unet_dims,
        n_groups=cfg.algo.n_groups,
        device=device,
    ).to(device)

    # -------------------------------------------------------------------------
    # 4. Setup Logger & Checkpointing
    # -------------------------------------------------------------------------
    def save(save_path):
        ema.copy_to(ema_agent.parameters())
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent"])
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    print("agent and logger made!", flush=True)

    # -------------------------------------------------------------------------
    # 5. Load Dataset
    # -------------------------------------------------------------------------
    print("loading dataset...", flush=True)

    dataset = DPDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=cfg.algo.obs_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        control_mode=None,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=cfg.algo.truncate_trajectories_at_success,
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(
        sampler, batch_size=cfg.algo.batch_size, drop_last=True
    )
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.algo.num_iterations)
    
    train_dataloader = ClosableDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.algo.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=cfg.seed),
        pin_memory=True,
        persistent_workers=(cfg.algo.num_dataload_workers > 0),
    )

    print("dataset loaded!", flush=True)

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    iteration = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    def check_freq(freq):
        return iteration % freq == 0


    agent.train()
    timer = NonOverlappingTimeProfiler()

    for iteration, data_batch in tqdm(
        enumerate(train_dataloader),
        initial=logger_start_log_step,
        total=cfg.algo.num_iterations,
    ):
        data_batch = to_tensor(data_batch, device=device, dtype=torch.float)
        if iteration + logger_start_log_step > cfg.algo.num_iterations:
            break

        # Data transfer to GPU
        obs_batch_dict = data_batch["observations"]
        obs_batch_dict = {
            k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()
        }
        act_batch = data_batch["actions"].cuda(non_blocking=True)

        # Forward & Loss
        total_loss = agent.compute_loss(
            obs_seq=obs_batch_dict, 
            action_seq=act_batch, 
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        ema.step(agent.parameters())

        logger.store("losses", loss=total_loss.item())
        logger.store("charts", learning_rate=optimizer.param_groups[0]["lr"])
        timer.end(key="train")

        # Logging
        if check_freq(cfg.algo.log_freq):
            if iteration > 0:
                logger.store("time", **timer.get_time_logs(iteration))
            logger.log(logger_start_log_step + iteration)
            timer.end("log")


        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{iteration}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    # Clean up
    train_dataloader.close()
    logger.close()

if __name__ == "__main__":
    main()