import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from dacite import from_dict
from omegaconf import OmegaConf

from mani_skill import ASSET_DIR
from mshab.utils.config import parse_cfg
from mshab.utils.dataclasses import default_field
from mshab.utils.logger import LoggerConfig
from mshab.envs.make import EnvConfig

@dataclass
class DPConfig:
    name: str = "diffusion_policy"

    # Diffusion Policy Parameters
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    obs_horizon: int = 2  
    act_horizon: int = 4 
    pred_horizon: int = 8 
    state_dim: int = 11
    action_dim: int = 11
    diffusion_step_embed_dim: int = 64 
    unet_dims: List[int] = default_field([64, 128, 256])
    n_groups: int = 8 
    encoded_image_feature_size: int = 1024

    # Dataset Parameters
    data_dir_fp: str = str(ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange-dataset/tidy_hosue/pick")
    trajs_per_obj: Union[Literal["all"], int] = "all"
    truncate_trajectories_at_success: bool = False
    max_image_cache_size: Union[Literal["all"], int] = 0
    num_dataload_workers: int = 0

    # Experiment Parameters
    num_iterations: int = 1_000_000
    eval_episodes: Optional[int] = None
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: int = 5000
    torch_deterministic: bool = True
    save_backup_ckpts: bool = False


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: DPConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        if self.algo.eval_episodes is None:
            self.algo.eval_episodes = self.eval_env.num_envs
        self.algo.eval_episodes = max(self.algo.eval_episodes, self.eval_env.num_envs)
        assert self.algo.eval_episodes % self.eval_env.num_envs == 0

        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        # Handle resume logic
        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            # Logic to check if we are resuming from the same path or a new one
            # Note: Assuming PASSED_CONFIG_PATH handling is done outside or passed explicitly if needed
            if old_config_path.exists():
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.logger.exp_cfg = asdict(self)
        # Clean up config for logging
        if "exp_cfg" in self.logger.exp_cfg["logger"]:
            del self.logger.exp_cfg["logger"]["exp_cfg"]
        if "resume_logdir" in self.logger.exp_cfg:
            del self.logger.exp_cfg["resume_logdir"]
        if "model_ckpt" in self.logger.exp_cfg:
            del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg_dict) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg_dict) if isinstance(cfg_dict, (OmegaConf, dict)) else cfg_dict)