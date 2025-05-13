#!/usr/bin/env python3
# Adapted heavily from:
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
#   Docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from PIL import Image
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from decoder_vis import draw_histogram_with_axes
from nes_ai.last_and_skip_wrapper import LastAndSkipEnv
from super_mario_env import SuperMarioEnv



from gymnasium.envs.registration import register

register(
    id="SuperMarioBros-mame-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=60 * 60 * 5,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]


USE_VAE = False

class VisionModel(Enum):
    CONV_GRAYSCALE = 'conv_grayscale'
    CONV_GRAYSCALE_224 = 'conv_grayscale_224'
    CONV_GRAYSCALE_224_extra_reduction = 'conv_grayscale_224_extra_reduction'
    CONV_GRAYSCALE_224_small_stride = 'conv_grayscale_224_small_stride'
    CONV_GRAYSCALE_224_with_linear = 'conv_grayscale_224_with_linear'
    RESNET_GRAYSCALE_224 = 'resnet_grayscale_224'
    VAE_GRAYSCALE_224 = 'vae_grayscale_224'
    PRETRAINED = 'pretrained'


@dataclass
class Args:
    r"""
    Run example:
        > WANDB_API_KEY=<key> python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_130130-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30

    Resume example:
        > WANDB_API_KEY=<key> WANDB_RUN_ID=SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30 WANDB_RESUME=must python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_133317-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
    --> wandb: Resuming run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        ...
    --> resumed at update 9
        ...
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    wandb_project_name: str = "MarioRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_run_id: str | None = None
    """the id of a wandb run to resume"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    train_agent: bool = True
    """enable or disable training of the agent"""

    # Visualization
    visualize_decoder: bool = True

    # Specific experiments
    dump_trajectories: bool = False
    reset_to_save_state: bool = False

    # Vision model
    vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_with_linear
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_with_linear
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_extra_reduction
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_small_stride
    #vision_model: VisionModel = VisionModel.RESNET_GRAYSCALE_224
    #vision_model: VisionModel = VisionModel.VAE_GRAYSCALE_224

    # Algorithm specific arguments
    env_id: str = "SuperMarioBros-mame-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    #learning_rate: float = 2.5e-4
    learning_rate: float = 2.5e-3

    """the learning rate of the decoder optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""


    num_minibatches: int = 4
    """the number of mini-batches"""
    #update_epochs: int = 100
    update_epochs: int = 20
    """the K epochs to update the policy"""

    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, vision_model: VisionModel):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            env = gym.make(env_id, render_mode="human", reset_to_save_state=False, screen_rc=(4,4))

        print(f"RENDER MODE: {env.render_mode}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = LastAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)

        if vision_model == VisionModel.CONV_GRAYSCALE:
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
        elif vision_model in (
            VisionModel.CONV_GRAYSCALE_224,
            VisionModel.CONV_GRAYSCALE_224_extra_reduction,
            VisionModel.CONV_GRAYSCALE_224_small_stride,
            VisionModel.CONV_GRAYSCALE_224_with_linear,
            VisionModel.RESNET_GRAYSCALE_224,
            VisionModel.VAE_GRAYSCALE_224,
        ):
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.ResizeObservation(env, (224, 224))
        elif vision_model == VisionModel.PRETRAINED:
            env = gym.wrappers.ResizeObservation(env, (224, 224))
        else:
            raise AssertionError(f"Unexpected vision model type: {vision_model}")

        env = gym.wrappers.FrameStackObservation(env, 4)

        return env

    return thunk


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0, weight_const=None):
    if weight_const is not None:
        torch.nn.init.constant_(layer.weight, weight_const)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Smallest mobilenet
IMAGE_MODEL_NAME = "mobilenetv3_small_050.lamb_in1k"


class ConvTrunkGrayscale224(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),     # -> (B, 32, 55, 55)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    # -> (B, 64, 26, 26)
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),   # -> (B, 128, 24, 24)
            #nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)

class ConvTrunkGrayscale224Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),       # 24 → 26
            #nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),        # 26 → 54
            #nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4),
            nn.Sigmoid(),

            nn.Upsample(size=(224, 224), mode='nearest'),
        )

    def forward(self, x):
        #return (self.deconv(x) + 1) / 2.0   # -> (B, 4, 224, 224)
        return self.deconv(x)


class ConvTrunkGrayscale224_small_stride(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3),     # (B, 32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),    # (B, 64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),   # (B, 128, 224, 224)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)  # -> (B, 128, 224, 224)


class ConvTrunkGrayscale224Decoder_small_stride(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=2),  # -> (64, 224, 224)
            #nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),   # -> (32, 224, 224)
            #nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 4, kernel_size=7, stride=1, padding=3),    # -> (4, 224, 224)
            nn.Tanh()
        )

    def forward(self, x):
        return (self.deconv(x) + 1) / 2.0  # Scale output to [0, 1]


class ConvTrunkGrayscale224_extra_reduction(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=16, stride=4),     # -> (B, 32, 55, 55)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=2),    # -> (B, 64, 26, 26)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),   # -> (B, 128, 24, 24)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)  # -> (B, 128, 224, 224)


class ConvTrunkGrayscale224Decoder_extra_reduction(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),       # 24 → 26
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2),        # 26 → 54
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 4, kernel_size=16, stride=4),
            nn.Tanh(),

            nn.Upsample(size=(224, 224), mode='nearest'),
        )

    def forward(self, x):
        return (self.deconv(x) + 1) / 2.0  # Scale output to [0, 1]


class ConvTrunkGrayscale224_with_linear(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),     # -> (B, 32, 55, 55)
            #nn.BatchNorm2d(32),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),    # -> (B, 64, 26, 26)
            #nn.BatchNorm2d(64),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),   # -> (B, 64, 24, 24)
            #nn.BatchNorm2d(64),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Flatten(),
            nn.LayerNorm(64 * 24 * 24),
            layer_init(nn.Linear(64 * 24 * 24, 512)),        # (B, 36864) → (B, 512)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)  # -> (B, 128, 224, 224)


class ConvTrunkGrayscale224Decoder_with_linear(nn.Module):
    def __init__(self):
        super().__init__()

        if False:
            self.deconv = nn.Sequential(
                layer_init(nn.Linear(512, 64 * 24 * 24)),
                nn.ReLU(),

                nn.Unflatten(1, (64, 24, 24)),  # (B, 512) → (B, 64, 7, 7)
                layer_init(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)),      # 7 → 9
                #nn.LayerNorm([64, 26, 26]),
                #nn.BatchNorm2d(64),
                nn.GroupNorm(8, 64),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),      # 9 → 20
                #nn.BatchNorm2d(32),
                nn.GroupNorm(4, 32),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4)),       # 20 → 84
                #nn.Tanh(),
                nn.Sigmoid(),

                nn.Upsample(size=(224, 224), mode='nearest')             # 84 → 224
            )

        self.deconv = nn.Sequential(
            layer_init(nn.Linear(512, 32 * 12 * 12)),   # Fewer channels, smaller spatial size
            nn.ReLU(),

            nn.Unflatten(1, (32, 12, 12)),

            layer_init(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)),   # 12 → 24
            nn.GroupNorm(4, 16),
            nn.ReLU(),

            layer_init(nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)),    # 24 → 48
            nn.Sigmoid(),

            nn.Upsample(size=(224, 224), mode='nearest')  # Final upscale to full image
        )

    def forward(self, x):
        return self.deconv(x)   # Scale output to [0, 1]
        #return (self.deconv(x) + 1) / 2.0  # Scale output to [0, 1]




class VaeGrayscale224(nn.Module):
    def __init__(self):
        super().__init__()

        # (E, 4, 224, 224) -> (1, 36864)
        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),     # (B, 32, 55, 55)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),    # (B, 64, 26, 26)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Conv2d(64, 128, 3, stride=2),   # (B, 128, 12, 12)
            nn.Conv2d(64, 128, 3, stride=1),   # (B, 128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 256, 3, stride=2),  # (B, 256, 5, 5)
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
        )

        # For VAE
        self.mu = nn.Conv2d(128, 128, kernel_size=1)
        self.logvar = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x):
        h = self.trunk(x)           # (B, 256, 5, 5)
        mu = self.mu(h)             # (B, 256, 5, 5)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std          # Reparameterization trick
        return z, mu, logvar


class VaeGrayscale224Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),   # -> (B, 64, 23, 23)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),    # -> (B, 32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4),     # -> (B, 4, 224, 224)
            nn.BatchNorm2d(4),
            nn.Upsample(size=(224, 224), mode='nearest'),
            nn.Sigmoid(),

            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),  # -> (B, 128, 11, 11)
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),   # -> (B, 64, 23, 23)
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),    # -> (B, 32, 48, 48)
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, output_padding=0),     # -> (B, 4, 224, 224)
            # nn.BatchNorm2d(4),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.deconv(x)   # -> (B, 4, 224, 224)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transposed=False):
        super().__init__()

        # Choose Conv type
        Conv = nn.ConvTranspose2d if transposed else nn.Conv2d

        # First conv kwargs
        conv1_kwargs = {'stride': stride, 'padding': 1}
        if transposed and stride > 1:
            conv1_kwargs['output_padding'] = 1

        # Shortcut kwargs
        shortcut_kwargs = {'stride': stride}
        if transposed and stride > 1:
            shortcut_kwargs['output_padding'] = 1

        # Residual path
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, **conv1_kwargs),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels)
        )

        # Shortcut (identity or matching conv)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, **shortcut_kwargs),
                # nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        return self.relu(out + residual)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),  # → (B, 64, 112, 112)
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # → (B, 64, 56, 56)
        )

        self.layers = nn.Sequential(
            ResidualBlock(64, 128, stride=2),   # → (B, 128, 28, 28)
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),  # → (B, 256, 14, 14)
            # ResidualBlock(256, 256, stride=2),  # → (B, 256, 7, 7)
        )

    def forward(self, x):
        x = self.initial(x)
        return self.layers(x)


class ResNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # ResidualBlock(256, 256, stride=2, transposed=True),  # → (B, 256, 14, 14)
            ResidualBlock(256, 128, stride=2, transposed=True),  # → (B, 128, 28, 28)
            ResidualBlock(128, 64, stride=2, transposed=True),   # → (B, 64, 56, 56)
            ResidualBlock(64, 64, transposed=True),              # → (B, 64, 56, 56)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=4),  # → (B, 4, 224, 224)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return self.final(x)


class Agent(nn.Module):
    def __init__(self, envs, vision_model: VisionModel):
        super().__init__()

        print(f"USING VISION MODEL IN AGENT: {vision_model}")
        if vision_model == VisionModel.CONV_GRAYSCALE_224:
            self.trunk = ConvTrunkGrayscale224()
            self.decoder = ConvTrunkGrayscale224Decoder()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224_small_stride:
            self.trunk = ConvTrunkGrayscale224_small_stride()
            self.decoder = ConvTrunkGrayscale224Decoder_small_stride()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224_extra_reduction:
            self.trunk = ConvTrunkGrayscale224_extra_reduction()
            self.decoder = ConvTrunkGrayscale224Decoder_extra_reduction()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224_with_linear:
            self.trunk = ConvTrunkGrayscale224_with_linear()
            self.decoder = ConvTrunkGrayscale224Decoder_with_linear()
        elif vision_model == VisionModel.RESNET_GRAYSCALE_224:
            self.trunk = ResNetEncoder()
            self.decoder = ResNetDecoder()
        elif vision_model == VisionModel.VAE_GRAYSCALE_224:
            self.trunk = VaeGrayscale224()
            self.decoder = VaeGrayscale224Decoder()
        else:
            raise AssertionError(f"Unexpected vision model: {vision_model}")

        self.action_dim = envs.single_action_space.n

    def get_action_and_value(self, x, action=None):
        trunk_output = self.trunk(x / 255.0)

        action = torch.randint(low=0, high=self.action_dim, size=(1,))
        return action, trunk_output


def _draw_obs(obs_np, screen: Any, screen_index: int):
    # assert obs_np.shape == (224, 224), f"Unexpected observation shape: {obs_np.shape} != (224, 224)"
    assert obs_np.max() <= 1.0, f"Unexpected observation values: min={obs_np.min()} max={obs_np.max()}"

    obs_grayscale = (obs_np * 255).astype(np.uint8)
    img_gray = Image.fromarray(obs_grayscale.T, mode='L')
    img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')

    screen.blit_image(img_rgb_240, screen_index=screen_index)


def contrastive_loss(z1, z2, temperature=0.1):
    """
    Contrastive loss for 4D input tensors (B, C, H, W).
    z1, z2: two augmented views of shape (B, 4, 224, 224)
    """
    B = z1.size(0)

    # Flatten spatial dimensions: (B, 4, 224, 224) -> (B, 200704)
    z1 = z1.view(B, -1)
    z2 = z2.view(B, -1)

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Combine for all pairs: (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # Compute cosine similarity matrix: (2B, 2B)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B, 2B)

    # Create labels: positives are at (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])  # shape (2B,)

    # Mask out self-similarity
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -9e15)

    # Scale by temperature
    sim = sim / temperature

    # Cross entropy loss, where each example should match its positive pair
    loss = F.cross_entropy(sim, labels)
    return loss


def contractive_loss(x, x_hat, z, lam=1e-4):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')

    # Approximate the contractive penalty
    z_flat = z.view(z.size(0), -1)         # (B, D)
    ones = torch.ones_like(z_flat)
    z_flat.backward(ones, retain_graph=True)
    # x.grad will be populated
    frob_norm = torch.norm(x.grad.view(x.size(0), -1), p='fro') ** 2
    return recon_loss + lam * frob_norm


def generate_activation_map(encoder, device: str) -> NdArrayUint8:
    # x = observation

    # Target latent feature index
    i = 0  # Choose the feature dimension you want to visualize

    # Initialize input image (e.g., 4-channel NES frame)
    x = torch.randn(1, 4, 224, 224, requires_grad=True, device=device)

    # Use Adam optimizer on the image tensor
    optimizer = torch.optim.Adam([x], lr=0.05)

    # Main optimization loop
    for step in range(100):
        optimizer.zero_grad()

        # Optionally normalize input to [0, 1] range for stability
        #x_clamped = torch.clamp(x, 0, 1)

        # Forward pass through encoder
        #z = encoder(x_clamped)
        z = encoder(x)

        print(f"SHAPE OF X: {x.shape}")
        print(f"SHAPE OF Z: {z.shape}")

        # Maximize the i-th feature (can use abs() if you want activation)
        loss = -z[0, -1].norm()

        print(f"SHAPE OF LOSS: {loss.shape}")

        loss.backward()

        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Feature {i} | Activation: {z[0, -1].norm():.4f}")

    # Visualize the last optimized image (one channel or more)
    with torch.no_grad():
        result_obs = torch.clamp(x[0], 0, 1).cpu()

    return result_obs


def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))


def l2_loss(x):
    return torch.mean(x ** 2)


def get_activation_hook(container):
    def hook(module, input, output):
        container['activation'] = output.detach()
    return hook


from PIL import Image, ImageFilter
import torchvision.transforms as T

def blur_stack(input_tensor, radius=5):
    to_pil = T.ToPILImage()
    to_tensor = T.ToTensor()

    blurred_frames = []

    for i in range(4):
        img_pil = to_pil(input_tensor[0, i])  # (224, 224)

        print(f"IMG PIL Shape: {img_pil.size}")

        img_blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        img_tensor = to_tensor(img_blurred).squeeze(0)  # (224, 224)

        print(f"FRAME img_tensor: {img_tensor.shape}")
        blurred_frames.append(img_tensor)

    blurred_stack = torch.stack(blurred_frames, dim=0).unsqueeze(0)  # (1, 4, 224, 224)
    return blurred_stack


def check_grad(model, device: str):
    x = torch.rand(1, 4, 224, 224, requires_grad=True, device=device)
    model.eval()
    act_container = {}

    print(f"x.is_leaf in check_grad(): {x.is_leaf}")

    def get_activation_hook(container):
        def hook(module, input, output):
            container['activation'] = output
        return hook

    for name, module in model.named_modules():
        print(f"CHECKING ACTIVATION FOR NAME: {name}")
        if not isinstance(module, torch.nn.Conv2d):
            continue

        handle = module.register_forward_hook(get_activation_hook(act_container))
        break

    out = model(x)
    activation = act_container['activation']
    loss = -activation.mean()
    loss.backward()

    print(f"x.grad mean: {x.grad.abs().mean()} activation mean: {activation.abs().mean()} activation.mean(): {activation.mean()}")  # Should be > 0


def maximize_each_layer(
    model,
    observation,
    input_shape=(1, 4, 224, 224),
    max_steps=1000,
    lr=0.05,
    tv_weight=1e-3,
    l2_weight=1e-3,
    #l2_weight=0,
    device='cpu',
    screen=None):

    check_grad(model, device=device)

    results = {}

    act_container = {}

    model.eval()

    obs_blur = blur_stack(observation.detach()).to(device)

    print(f"OBS BLUR: {obs_blur.shape}")

    layer_i = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue

        # Get real activation from input observation
        if False:
            with torch.no_grad():
                handle = module.register_forward_hook(get_activation_hook(act_container))
                obs_tensor = observation.clone().detach().float().div(255).to(device)  # (1, C, H, W)
                _ = model(obs_tensor)
                real_act = act_container['activation'].detach()

        print(f"Maximizing activation for: {name}")
        act_container = {}

        # Register a temporary hook
        handle = module.register_forward_hook(get_activation_hook(act_container))

        # Create a leaf input with grad
        if False:
            x = torch.rand(input_shape, device=device, requires_grad=True)
        else:
            x = obs_blur.clone().detach().requires_grad_()

        # print(f"x.is_leaf in maximize_each_layer(): {x.is_leaf}")
        optimizer = torch.optim.Adam([x], lr=lr)

        patience = 20

        best_loss = float('inf')
        steps_without_improvement = 0
        for step in range(max_steps):
            optimizer.zero_grad()

            # print(f"X is_leaf: {x.is_leaf}")
            _x_encoded = model(x)
            activation = act_container['activation']

            if False:
                act_score = F.cosine_similarity(activation.view(1, -1), real_act.view(1, -1)).mean()
            else:
                # E.g. torch.Size([1, 128, 24, 24])
                # print(f"ACTIVATION SHAPE: {activation.shape}")

                # Optimize specific neuron.  This looks like something, but is it just averaging noise?
                #act_score = activation[0, -1, -1, -1]
                act_score = activation[0, 16, 12, 12]
                print(f"act_score: {act_score=}")

                # Optimize whole sequence of frames.
                #act_score = activation[0].mean()
                #act_score = activation[0, 16].mean()


                # act_score = activation.norm()

            # Regularization
            if True:
                tv = total_variation_loss(x)
                l2 = l2_loss(x)
            else:
                tv = torch.tensor([0], device=x.device)
                l2 = torch.tensor([0], device=x.device)

            # Weighted sum
            loss = -act_score + tv_weight * tv + l2_weight * l2
            #loss = -act_score

            if x.grad:
                #print(f"X MEAN BEFORE BACKWARD: {x.grad.abs().mean()}")
                pass

            loss.backward()

            # print(f"X MEAN AFTER BACKWARD: {x.grad.abs().mean()}")
            # print(f"activation.mean().item() AFTER BACKWARD: {activation.mean()}")

            #print("BEFORE STEP:", x[0,0,0,0].item())
            optimizer.step()
            #print("AFTER STEP:", x[0,0,0,0].item())

            #print(f"X MEAN AFTER BACKWARD STEP: {x.grad.abs().mean()}")


            if step % 10 == 0 or step == max_steps - 1:
                # print(f"[{step:03}] cosine_sim={sim.item():.4f} | TV={tv.item():.5f} | L2={l2.item():.5f}")
                print(f"Step {step:03} | Loss: {loss} | Activation: {act_score.item():.5f} | TV: {tv.item():.5f} | L2: {l2.item():.5f}")
                #print(f"Step {step:03} | cosine_sim: {act_score.item():.5f} | TV: {tv.item():.5f} | L2: {l2.item():.5f}")

            if screen is not None:
                # print(f"SHOWING ON LAYER: {layer_i}")
                screen_index = 4 + layer_i

                activation_map_f = x[0].mean(dim=0).detach().cpu().numpy()
                activation_map_grayscale = (activation_map_f * 255).astype(np.uint8)

                img_gray = Image.fromarray(activation_map_grayscale.T, mode='L')
                img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
                screen.blit_image(img_rgb_240, screen_index=screen_index)

                if True:
                    # Show all 4 frames of the image.
                    for i in range(4):
                        activation_map_f = x[0,i].detach().cpu().numpy()
                        activation_map_grayscale = (activation_map_f * 255).astype(np.uint8)

                        img_gray = Image.fromarray(activation_map_grayscale.T, mode='L')
                        img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
                        screen.blit_image(img_rgb_240, screen_index=8 + i)

                screen.show()



            # Early stopping check
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    print(f"Activation: Early stopping at step {step+1}")
                    break

        # Save result image
        optimized_image = x.clone().detach().clamp(0, 1).cpu()[0]
        results[name] = optimized_image


        print(f"OPTIMIZED IMAGE SIZE: {optimized_image.shape}")

        handle.remove()  # Remove hook after each layer

        layer_i += 1

    model.train()

    return results


def find_highest_activated_neuron(model, input_image, device='cpu', max_layers=10):
    model.eval()
    input_image = input_image.clone().detach().float().div(255).to(device)

    top_neuron = {'layer': None, 'channel': None, 'h': None, 'w': None, 'activation': -float('inf')}
    container = {}

    # Register forward hooks to capture activations
    def get_hook(name):
        def hook_fn(module, input, output):
            container[name] = output.detach().cpu()
        return hook_fn

    handles = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            filters = module.weight
            print(f"LAYER NAME: {name}: weights_shape={filters.shape}")
            handles.append(module.register_forward_hook(get_hook(name)))
            if len(handles) >= max_layers:
                break

    with torch.no_grad():
        _ = model(input_image)

    # Search for highest activation
    for name, act in container.items():
        val, idx = act.view(act.shape[0], act.shape[1], -1).mean(dim=2).max(dim=1)  # mean across spatial, max across channels
        c = idx.item()
        spatial = act[0, c]
        max_val = spatial.max().item()
        h, w = (spatial == spatial.max()).nonzero(as_tuple=True)
        h = h[0].item()
        w = w[0].item()

        if max_val > top_neuron['activation']:
            top_neuron.update({
                'layer': name,
                'channel': c,
                'h': h,
                'w': w,
                'activation': max_val
            })

    for handle in handles:
        handle.remove()

    return top_neuron


def visualize_conv_filters(screen, model):
    for i, (name, module) in enumerate(model.named_modules()):
        if not isinstance(module, torch.nn.Conv2d):
            continue

        filters = module.weight
        print(f"LAYER NAME: {name}: weight id={id(module.weight)} weights_shape={filters.shape}")

        expected_shape = (32, 4, 8, 8)
        assert filters.shape == expected_shape, f"Unexpected shape: {filters.shape} != {expected_shape}"

        # 32 filters of size 8x8
        filters_per_row = 240 // 8
        filters_per_col = 224 // 8

        filters_np = filters.detach().cpu().numpy()

        # -> (C, B, H, W)
        filters_cbhw = filters_np.transpose(1, 0, 2, 3)

        print(f"VISUALIZING FILTERS cbhw: {filters_cbhw.shape}")

        for channel, filters_bhw in enumerate(filters_cbhw):
            # Each channel goes in a different screen.
            screen_index_i = channel

            img_f = np.zeros((224, 240), dtype=np.float32)

            for f, filter in enumerate(filters_bhw):
                # (4, H, W)
                row = f // filters_per_row
                col = f % filters_per_row

                y = row * 8
                x = col * 8

                #print(f"FILTER SIZE: {filter.shape} f={f} row={row} col={col} x={x} y={y}")
                img_f[y:y+8, x:x+8] = filter

            img_gray = Image.fromarray((img_f * 255).astype(np.uint8), mode='L')
            img_rgb_240 = img_gray.convert('RGB')
            screen.blit_image(img_rgb_240, screen_index=12 + screen_index_i)

        # First layer only.
        break

def visualize_receptive_field(screen, model, input_image, target_layer_name='layer3.2.conv2', target_channel=16, target_h=12, target_w=12, device='cpu'):
    model = model.to(device).eval()
    input_image = input_image.clone().detach().float().div(255).to(device).requires_grad_()  # (1, 4, 224, 224)

    # === Register Hook ===
    act_container = {}
    def hook_fn(module, input, output):
        act_container['activation'] = output
    hook = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    if hook is None:
        raise ValueError(f"Layer {target_layer_name} not found")

    # === Forward & Saliency ===
    model.zero_grad()
    _ = model(input_image)
    activation = act_container['activation']
    target_neuron = activation[0, target_channel, target_h, target_w]
    target_neuron.backward()

    saliency = input_image.grad.abs().squeeze().cpu().numpy()  # (4, 224, 224)

    # === Activation Maximization with Spatial Mask ===
    x = torch.zeros_like(input_image, requires_grad=True)

    # Define spatial mask (e.g., 64×64 box centered in 224x224)
    mask = torch.zeros_like(x)
    #mask[:, :, 80:144, 80:144] = 1.0  # You can adjust this
    mask[:, :, :, :] = 1.0  # You can adjust this

    optimizer = torch.optim.Adam([x], lr=0.05)
    for _ in range(100):
        optimizer.zero_grad()
        _ = model(x * mask + (1 - mask) * x.detach())  # only optimize masked region
        act = act_container['activation'][0, target_channel, target_h, target_w]
        loss = -act + 1e-4 * (x ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.data.clamp_(0, 1)

    maxed_image = x.detach().squeeze(0).cpu().numpy()  # (4, 224, 224)

    # === Visualization ===
    for i in range(4):
        saliency_f = saliency[i]
        saliency_grayscale = (saliency_f * 255).astype(np.uint8)

        img_gray = Image.fromarray(saliency_grayscale.T, mode='L')
        img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
        screen.blit_image(img_rgb_240, screen_index=8 + i)

        screen.show()

    for i in range(4):
        maxed_image_f = maxed_image[i]
        max_image_grayscale = (maxed_image_f * 255).astype(np.uint8)

        img_gray = Image.fromarray(max_image_grayscale.T, mode='L')
        img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
        screen.blit_image(img_rgb_240, screen_index=12 + i)

        screen.show()

    hook.remove()

    model.train()


def visualize_full_saliency(screen, model, decoder, input_image, target_layer_name, target_channel, device='cpu'):
    model.eval()
    input_image = input_image.clone().detach().float().div(255).to(device).requires_grad_()  # (1, 4, 224, 224)

    # Hook to extract activation
    act_container = {}
    def hook_fn(module, input, output):
        act_container['activation'] = output

    # Register hook
    hook = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    if hook is None:
        raise ValueError(f"Layer {target_layer_name} not found")

    # Forward + backward
    model.zero_grad()
    encoded = model(input_image)
    # decoded = decoder(encoded)
    activation = act_container['activation']  # (1, C, H, W)

    # Sum over all spatial locations for this channel
    act_score = activation[0, target_channel].sum()
    act_score.backward()

    # Gradients w.r.t input
    saliency = input_image.grad.abs().squeeze(0).cpu().numpy()  # shape (4, 224, 224)

    print("input_image.grad.mean():", input_image.grad.abs().mean())
    print("activation.mean():", activation[0, target_channel].mean().item())


    for i in range(4):
        saliency_f = saliency[i]
        print("Saliency stats:", saliency_f.min(), saliency_f.max(), saliency_f.mean())

        saliency_norm = (saliency_f - saliency_f.min()) / (saliency_f.max() - saliency_f.min() + 1e-8)

        print(f"SALIENCY NORM RANGE: {saliency_norm.min()} {saliency_norm.max()}")

        saliency_grayscale = (saliency_norm * 255).astype(np.uint8)

        img_gray = Image.fromarray(saliency_grayscale.T, mode='L')
        img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
        screen.blit_image(img_rgb_240, screen_index=8 + i)

        screen.show()

    hook.remove()

    model.train()

def main():
    args = tyro.cli(Args)

    # Derived args.
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # NOTE: Run name should be descriptive, but not unique.
    # In particular, we don't include the date because the date does not affect the results.
    # Date prefixes are handled by wandb automatically.

    if not args.wandb_run_id:
        run_prefix = f"{args.env_id}__{args.exp_name}__{args.seed}"
        run_name = f"{run_prefix}__{date_str}"
        args.wandb_run_id = run_name

    run_name = args.wandb_run_id

    run_dir = f"runs/{run_name}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if True:
        if device == torch.device("cpu"):
            # Try mps
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                print("No GPU available, using CPU.")
    else:
        device = 'cpu'

    global USE_VAE
    if args.vision_model == VisionModel.VAE_GRAYSCALE_224:
        USE_VAE = True

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.vision_model) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    first_env = envs.envs[0].unwrapped
    screen = first_env.screen
    nes = first_env.unwrapped.nes
    action_dim = envs.single_action_space.n

    font = pygame.font.SysFont("Arial", 14)

    print(f"ACTION DIM: {action_dim}")

    # ActorCritic
    agent = Agent(envs, args.vision_model).to(device)
    decoder = agent.decoder

    optimizer = torch.optim.Adam(
        list(agent.trunk.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)

    next_obs = torch.Tensor(next_obs).to(device)

    starting_iter = 1

    for iteration in range(starting_iter, args.num_iterations + 1):
        print(f"Iter: {iteration}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        steps_start = time.time()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if USE_VAE:
                    action, (next_encoded_obs, mu, logvar) = agent.get_action_and_value(next_obs)
                else:
                    action, next_encoded_obs = agent.get_action_and_value(next_obs)

            next_obs_np = next_obs.cpu().numpy()

            # Check that the encoding matches the trunk, like we think.
            if True:
                if USE_VAE:
                    encoded_output2, mu2, logvar2 = agent.trunk(next_obs / 255)
                else:
                    encoded_output2 = agent.trunk(next_obs / 255)

                    #print(f"SHAPE OF ENCODED2: {len(encoded_output2)} NEXT={len(next_encoded_obs)}")
                    #print(f"SHAPE OF ENCODED2: {encoded_output2.shape} NEXT={next_encoded_obs.shape}")
                    if (encoded_output2 != next_encoded_obs).all():
                        print(f"ENCODED {encoded_output2=} OBS {next_obs=}")
                        raise AssertionError("NOT MATCHING")

            # Visualize latest observation.
            if True:
                _draw_obs(next_obs_np[0, -1] / 255, screen, 1)

            if args.visualize_decoder:
                with torch.no_grad():
                    # (1, 4, 224, 224)
                    decoded_obs_np = decoder(next_encoded_obs).cpu().numpy()

                # Use random image.
                if False:
                    decoded_grayscale = np.random.random((224, 240))
                else:
                    decoded_grayscale_f = decoded_obs_np[0, -1]

                _draw_obs(decoded_grayscale_f, screen, 2)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # NOTE: Silent conversion to float32 for Tensor.
            next_obs = torch.Tensor(next_obs).to(device)

            if pygame.K_v in nes.keys_pressed:
                start_vis = time.time()
                print("Generating activation maps...")
                #activation_map_rgb = generate_activation_map(agent.trunk, device=device)

                if True:
                    print(f"MODEL ID OF agent.trunk: {id(agent.trunk)}")
                    visualize_conv_filters(screen, agent.trunk)

                if False:
                    activation_map_results = maximize_each_layer(agent.trunk, next_obs, device=device, screen=screen)

                if False:
                    model = agent.trunk
                    decoder = agent.decoder

                    # Get your real frame stack: shape (1, 4, 224, 224)
                    neuron = find_highest_activated_neuron(model, next_obs, device=device)
                    print(f"Best neuron found: {neuron}")

                    if False:
                        visualize_receptive_field(
                            screen,
                            model,
                            input_image=next_obs,
                            target_layer_name=neuron['layer'],
                            target_channel=neuron['channel'],
                            target_h=neuron['h'],
                            target_w=neuron['w'],
                            device=device,
                        )

                    visualize_full_saliency(
                        screen,
                        model,
                        decoder,
                        input_image=next_obs,
                        target_layer_name=neuron['layer'],
                        target_channel=neuron['channel'],
                        device=device,
                    )

                if False:
                    for i, (name, result) in enumerate(activation_map_results.items()):
                        activation_map_f = activation_map_results[name][-1].cpu().numpy()

                        print(f"GRAYSCALE_F RANGE: {activation_map_f.min()}, {activation_map_f.max()}, dtype={activation_map_f.dtype}")

                        activation_map_grayscale = (activation_map_f * 255).astype(np.uint8)

                        print(f"GRAYSCALE RANGE: {activation_map_grayscale.min()}, {activation_map_grayscale.max()}, dtype={activation_map_grayscale.dtype}")

                        img_gray = Image.fromarray(activation_map_grayscale.T, mode='L')
                        img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')
                        screen.blit_image(img_rgb_240, screen_index=4 + i)

                        if font is None:
                            font = pygame.font.SysFont("Arial", 14)
                        antialias = True
                        color = (255, 0, 0)
                        layer_text = font.render(name, antialias, color)
                        screen.surfs[4 + i].blit(layer_text, (0, 224 - 14))

                print(f"Generated activation maps: {time.time()-start_vis:.4f}s")

            if nes.keys_pressed:
                nes.keys_pressed = []

        steps_end = time.time()

        if args.dump_trajectories:
            traj_dir = Path(run_dir) / "traj"
            traj_dir.mkdir(parents=True, exist_ok=True)

            traj_filename = traj_dir / f'iter_{iteration}.npz'

            np.savez_compressed(
                traj_filename,
                obs=obs.cpu().numpy(),
                actions=np.array(),
                logprobs=np.array(),
                rewards=np.array(),
                dones=np.array(),
                values=np.array(),
            )

        if args.train_agent:
            optimize_networks_start = time.time()

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

            b_obs_tensor = b_obs / 255

            executed_epochs = 0
            epochs_start = time.time()

            # Select 80% as training set, remainder as test set.
            train_end_index = int(args.batch_size * 0.8)
            b_inds = np.arange(train_end_index)
            b_val_inds = train_end_index + np.arange(args.batch_size - train_end_index)

            b_val_obs = b_obs_tensor[b_val_inds]

            # Display 4 random observations
            print(f"INIT BATCH INDICES: {b_inds}")
            print(f"INIT VAL INDICES: {b_val_inds}")

            # Validation
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            patience = 20  # Stop if no improvement for this many epochs

            while executed_epochs < args.update_epochs or args.update_epochs <= 0:
                executed_epochs += 1
                epoch = executed_epochs

                # Shuffle all observations.
                np.random.shuffle(b_inds)

                # OK: These look random
                # print(f"RANDOM BATCH INDICES: {b_inds}")

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    obs_tensor = b_obs_tensor[mb_inds]

                    obs_tensor.requires_grad = True


                    if USE_VAE:
                        x = obs_tensor
                        z, mu, logvar = agent.trunk(obs_tensor)
                        encoded_tensor = z
                    else:
                        encoded_tensor = agent.trunk(obs_tensor)
                        z = encoded_tensor

                    if False:
                        print(f"Encoded std across batch (shape={z.shape}): {z.std(dim=0).mean()}")  # low std across batch => collapse

                    if USE_VAE:
                        # Agent image decoder
                        x_hat = decoder(z)

                        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

                        # KL divergence per pixel
                        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

                        loss = recon_loss + kl

                        decoded_tensor = x_hat
                    else:
                        decoded_tensor = decoder(encoded_tensor)

                        # print(f"DECODED SHAPE: {decoded_tensor.shape} obs={obs_tensor.shape}")

                        # loss = F.mse_loss(decoded_tensor, obs_tensor)
                        # loss = contrastive_loss(decoded_tensor, obs_tensor)
                        x = obs_tensor
                        z = encoded_tensor
                        x_hat = decoded_tensor
                        loss = contractive_loss(x, x_hat, z)

                    # Total loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if False:
                    # TODO(millman): Activation maximization seems like the next thing to try
                    with torch.no_grad():
                        from sklearn.manifold import TSNE

                        # Encode the entire
                        z = next_encoded_obs.cpu().numpy()

                        print(f"SIZE OF Z: {z.shape}")

                        z_2d = TSNE().fit_transform(z.swapaxes(0, 1))

                        img = Image.asarray(z_2d).resize((240, 224))

                        screen.surfs[3].set_image(img)


                # --- Visualization ---
                #with torch.no_grad():
                if False:
                    # Draw the first 4 random items
                    b_obs_vis = b_obs_tensor[b_inds[:1]].detach()
                    if USE_VAE:
                        b_encoded_obs, mu, logvar = agent.trunk(b_obs_vis)
                    else:
                        b_encoded_obs = agent.trunk(b_obs_vis)

                    b_decoded_obs = decoder(b_encoded_obs)

                    for i in range(4):
                        _draw_obs(b_obs_vis[0, i].cpu().numpy(), screen, 4 + i)
                        _draw_obs(b_decoded_obs[0, i].detach().cpu().numpy(), screen, 8 + i)

                        surf = screen.surfs[12 + i]

                        surf.fill((25, 25, 25))

                        x, y = 0, 0
                        hist_width = 240
                        spacing = 2

                        if False:
                            named_params = list(agent.trunk.named_parameters())
                            num_params = len(named_params)

                            # Total height is: hist_height*N + spacing*(N-1) + 18 = 224
                            hist_height = (224 - 18 - 10 - spacing * (num_params-1)) / num_params

                            # print(f"NUM_PARAMS: {num_params} HIST HEIGHT: {hist_height} y={y}")

                            for name, param in named_params:
                                # E.g.:
                                #  agent.trunk param: trunk.0.weight
                                #  agent.trunk param: trunk.0.bias
                                #  agent.trunk param: trunk.2.weight
                                #  agent.trunk param: trunk.2.bias

                                # print(f"agent.trunk param: {name}")
                                if param.grad is None:
                                    continue
                                grad_data = param.grad.detach().cpu().flatten().numpy()
                                rect = (x, y, hist_width, hist_height)
                                draw_histogram_with_axes(surf, grad_data, rect, label=name, font=font)
                                y += hist_height + spacing

                                # print(f"Y={y} += {hist_height} + {spacing}")

                        # Show loss
                        if font is None:
                            font = pygame.font.SysFont("Arial", 14)
                        loss_text = font.render(f"Epoch {epoch+1} | Loss: {loss.item():.4f}", True, (255, 255, 255))
                        surf.blit(loss_text, (x, y + 10))


                # ----- Validation -----
                with torch.no_grad():
                    if USE_VAE:
                        z, mu, logvar = agent.trunk(b_val_obs)
                        x = b_val_obs
                        x_hat = decoder(z)

                        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

                        # KL divergence per pixel
                        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

                        val_loss = recon_loss + kl

                    else:
                        val_encoded = agent.trunk(b_val_obs)

                        val_recon = decoder(val_encoded)

                        val_loss = F.mse_loss(val_recon, b_val_obs)

                print(f"Epoch {epoch+1}: train_loss={loss:.4f} val_loss={val_loss:.4f}")

                # ----- Early stopping logic -----
                if val_loss < best_val_loss - 1e-4:  # Small threshold to avoid noise
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}, no improvement in {patience} epochs.")
                        break

            epochs_end = time.time()
            epoch_dt = epochs_end - epochs_start

            optimize_networks_end = time.time()

            num_samples = executed_epochs * args.batch_size
            per_sample_dt = epoch_dt / num_samples

            steps_dt = steps_end - steps_start
            optimize_networks_dt = optimize_networks_end - optimize_networks_start

            print(f"Time steps: (num_steps={args.num_steps}): {steps_dt:.4f}")
            print(f"Time optimize: (epochs={args.update_epochs} batch_size={args.batch_size} minibatch_size={args.minibatch_size}) per-sample: {per_sample_dt:.4f} optimize_networks: {optimize_networks_dt:.4f}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()