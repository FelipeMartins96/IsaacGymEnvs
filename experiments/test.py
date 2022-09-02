import hydra
import gym
import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.learning.replay_buffer import ReplayBuffer

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg : DictConfig) -> None:
    envs = isaacgym_task_map[cfg.task.name](
            cfg=omegaconf_to_dict(cfg.task),
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=cfg.virtual_screen_capture,
            force_render=True,
        )

    writer = SummaryWriter()

    while True:
        envs.step(torch.tensor(envs.action_space.sample(), device=cfg.rl_device))


if __name__ == "__main__":
    train()
