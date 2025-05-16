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
from scipy.ndimage import zoom
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


class VisionModel(Enum):
    CONV_GRAYSCALE_224 = 'conv_grayscale_224'
    CONV_GRAYSCALE_224_big = 'conv_grayscale_224_big'
    CONV_GRAYSCALE_224_with_linear = 'conv_grayscale_224_with_linear'


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
    continuous_vis: bool = True
    visualize_decoder: bool = True

    # Specific experiments
    dump_trajectories: bool = False
    reset_to_save_state: bool = False

    # Vision model
    vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_big
    #vision_model: VisionModel = VisionModel.CONV_GRAYSCALE_224_with_linear

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
    num_steps: int = 2
    """the number of steps to run in each environment per policy rollout"""

    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = -1
    #update_epochs: int = 20
    """the K epochs to update the policy"""

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
            env = gym.make(env_id, render_mode="human", reset_to_save_state=False, screen_rc=(5,8))

        print(f"RENDER MODE: {env.render_mode}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = LastAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)

        if vision_model in (
            VisionModel.CONV_GRAYSCALE_224,
            VisionModel.CONV_GRAYSCALE_224_big,
            VisionModel.CONV_GRAYSCALE_224_with_linear,
        ):
            env = gym.wrappers.GrayscaleObservation(env)
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


class ConvTrunkGrayscale224_big(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=9, stride=4),         # 224 → 55
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),        # 55 → 26
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),       # 26 → 11
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=1),      # 11 → 10
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)

class ConvTrunkGrayscale224Decoder_big(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1),         # 10 → 11
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),          # 11 → 25
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),           # 25 → 53
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=9, stride=4),            # 53 → 224

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.deconv(x)


class ConvTrunkGrayscale224_with_linear(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),     # -> (B, 32, 55, 55)
            #nn.BatchNorm2d(32),
            #nn.GroupNorm(4, 32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),    # -> (B, 64, 26, 26)
            #nn.BatchNorm2d(64),
            #nn.GroupNorm(8, 64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),   # -> (B, 64, 24, 24)
            #nn.BatchNorm2d(64),
            #nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Flatten(),
            #nn.LayerNorm(64 * 24 * 24),
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
                #nn.GroupNorm(8, 64),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),      # 9 → 20
                #nn.BatchNorm2d(32),
                #nn.GroupNorm(4, 32),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4)),       # 20 → 84
                #nn.Tanh(),
                nn.Sigmoid(),

                nn.Upsample(size=(224, 224), mode='nearest')             # 84 → 224
            )

        if True:
            self.deconv = nn.Sequential(
                layer_init(nn.Linear(512, 64 * 24 * 24)),
                nn.ReLU(),

                nn.Unflatten(1, (64, 24, 24)),  # (B, 512) → (B, 64, 7, 7)
                layer_init(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)),      # 7 → 9
                #nn.LayerNorm([64, 26, 26]),
                #nn.BatchNorm2d(64),
                #nn.GroupNorm(8, 64),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),      # 9 → 20
                #nn.BatchNorm2d(32),
                #nn.GroupNorm(4, 32),

                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4)),       # 20 → 84
                #nn.Tanh(),
                nn.Sigmoid(),

                nn.Upsample(size=(224, 224), mode='nearest')             # 84 → 224
            )


        if False:
            self.deconv = nn.Sequential(
                layer_init(nn.Linear(512, 32 * 12 * 12)),   # Fewer channels, smaller spatial size
                nn.ReLU(),

                nn.Unflatten(1, (32, 12, 12)),

                layer_init(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)),   # 12 → 24
                #nn.GroupNorm(4, 16),
                nn.ReLU(),

                layer_init(nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)),    # 24 → 48
                nn.Sigmoid(),

                nn.Upsample(size=(224, 224), mode='nearest')  # Final upscale to full image
        )

    def forward(self, x):
        return self.deconv(x)   # Scale output to [0, 1]
        #return (self.deconv(x) + 1) / 2.0  # Scale output to [0, 1]


class Agent(nn.Module):
    def __init__(self, envs, vision_model: VisionModel):
        super().__init__()

        print(f"USING VISION MODEL IN AGENT: {vision_model}")
        if vision_model == VisionModel.CONV_GRAYSCALE_224:
            self.trunk = ConvTrunkGrayscale224()
            self.decoder = ConvTrunkGrayscale224Decoder()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224_big:
            self.trunk = ConvTrunkGrayscale224_big()
            self.decoder = ConvTrunkGrayscale224Decoder_big()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224_with_linear:
            self.trunk = ConvTrunkGrayscale224_with_linear()
            self.decoder = ConvTrunkGrayscale224Decoder_with_linear()
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


def visualize_conv_filters(screen, model):
    scale = 3
    row_spacing = 2
    col_spacing = 2

    max_screen_index = len(screen.surfs)

    # Start on the second row.
    screen_index = screen.cols

    def blit_current_image(img_f):
        img_gray = Image.fromarray((img_f * 255).astype(np.uint8), mode='L')
        img_rgb_240 = img_gray.convert('RGB')
        screen.blit_image(img_rgb_240, screen_index=screen_index)

    for i, (name, module) in enumerate(model.named_modules()):
        if not isinstance(module, torch.nn.Conv2d):
            continue

        # Normalize filters weights across layer.
        filters = module.weight.detach().cpu().numpy()
        filters_f = (filters - filters.min()) / (filters.max() - filters.min())

        raw_fh, raw_fw = filters.shape[2:]
        fh, fw = int(raw_fh * scale), int(raw_fw * scale)

        filters_per_row = 240 // (fw + col_spacing)
        filters_per_col = 224 // (fh + row_spacing)

        img_f = np.zeros((224, 240), dtype=np.float32)
        row = 0
        col = 0

        for channel, filters_bhw in enumerate(filters_f):
            for f, filter in enumerate(filters_bhw):
                if screen_index >= max_screen_index:
                    break

                scaled_filter = zoom(filter, zoom=scale, order=0)
                y = row * (fh + row_spacing)
                x = col * (fw + col_spacing)
                img_f[y:y+fh, x:x+fw] = scaled_filter

                col += 1

                if col >= filters_per_row:
                    col = 0
                    row += 1

                    if row >= filters_per_col:
                        blit_current_image(img_f)
                        screen_index += 1
                        img_f[:] = 0
                        row = 0
                        col = 0

        if screen_index >= max_screen_index:
            break

        # Blit any remaining filters on this layer's image
        blit_current_image(img_f)
        screen_index += 1


def contractive_loss(x, x_hat, z, lam=1e-4):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')

    # Approximate the contractive penalty
    z_flat = z.view(z.size(0), -1)         # (B, D)
    ones = torch.ones_like(z_flat)
    z_flat.backward(ones, retain_graph=True)
    # x.grad will be populated
    frob_norm = torch.norm(x.grad.view(x.size(0), -1), p='fro') ** 2
    return recon_loss + lam * frob_norm


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

        steps_start = time.time()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, next_encoded_obs = agent.get_action_and_value(next_obs)

            next_obs_np = next_obs.cpu().numpy()

            # Check that the encoding matches the trunk, like we think.
            if False:
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

            if False: #pygame.K_v in nes.keys_pressed or (args.continuous_vis and iteration > 1):
                start_vis = time.time()
                print("Visualizing filters...")

                if True:
                    # print(f"MODEL ID OF agent.trunk: {id(agent.trunk)}")
                    visualize_conv_filters(screen, agent.trunk)

                print(f"Visualizing filters: {time.time()-start_vis:.4f}s")

            if nes.keys_pressed:
                nes.keys_pressed = []

        steps_end = time.time()

        if args.train_agent:
            optimize_networks_start = time.time()

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

            b_obs_tensor = b_obs / 255

            executed_epochs = 0
            epochs_start = time.time()

            # Select 80% as training set, remainder as test set.
            train_end_index = max(1, int(args.batch_size * 0.8))
            b_inds = np.arange(train_end_index)
            b_val_inds = train_end_index + np.arange(args.batch_size - train_end_index)

            b_val_obs = b_obs_tensor[b_val_inds]

            # Display 4 random observations
            if False:
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

                    if len(mb_inds) == 0:
                        continue

                    obs_tensor = b_obs_tensor[mb_inds]

                    # print(f"OBS TENSOR: {obs_tensor.shape} mb_inds={mb_inds} start={start} end={end}")

                    # NOTE: grad is required for contractive_loss()
                    obs_tensor.requires_grad = True

                    encoded_tensor = agent.trunk(obs_tensor)
                    z = encoded_tensor

                    if False:
                        print(f"Encoded std across batch (shape={z.shape}): {z.std(dim=0).mean()}")  # low std across batch => collapse

                    decoded_tensor = decoder(encoded_tensor)

                    # print(f"DECODED SHAPE: {decoded_tensor.shape} obs={obs_tensor.shape}")

                    loss = F.mse_loss(decoded_tensor, obs_tensor)

                    # TODO(millman): try a different loss function?  This seems to be blurry.  Or, use another conv layer?
                    # loss = F.smooth_l1_loss(decoded_tensor, obs_tensor)

                    # x = obs_tensor
                    # z = encoded_tensor
                    # x_hat = decoded_tensor
                    # loss = contractive_loss(x, x_hat, z)

                    # Total loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # ---- Visualization ----
                if args.continuous_vis:
                    if True:
                        start_vis = time.time()
                        # print("Visualizing filters...")

                        with torch.no_grad():
                            visualize_conv_filters(screen, agent.trunk)

                        # print(f"Visualizing filters: {time.time()-start_vis:.4f}s")

                    with torch.no_grad():
                        # (1, 4, 224, 224)
                        encoded_tensor = agent.trunk(b_obs_tensor[-1:])
                        decoded_obs_np = decoder(encoded_tensor).cpu().numpy()
                        decoded_grayscale_f = decoded_obs_np[0, -1]
                        _draw_obs(decoded_grayscale_f, screen, 2)

                    screen.show()

                # ----- Validation -----
                if len(b_val_obs) > 0:
                    with torch.no_grad():
                        val_encoded = agent.trunk(b_val_obs)

                        val_recon = decoder(val_encoded)

                        val_loss = F.mse_loss(val_recon, b_val_obs)
                else:
                    val_loss = float('nan')

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

            # TODO(millman): turn this into "time since last print", and aggregate stats across seconds
            if False: #
                print(f"Time steps: (num_steps={args.num_steps}): {steps_dt:.4f}")
                print(f"Time optimize: (epochs={args.update_epochs} batch_size={args.batch_size} minibatch_size={args.minibatch_size}) per-sample: {per_sample_dt:.4f} optimize_networks: {optimize_networks_dt:.4f}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()