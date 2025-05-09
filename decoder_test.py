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
    CONV_GRAYSCALE = 'conv_grayscale'
    CONV_GRAYSCALE_224 = 'conv_grayscale_224'
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

    # Algorithm specific arguments
    env_id: str = "SuperMarioBros-mame-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4

    """the learning rate of the decoder optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""


    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 40
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
            env = gym.make(env_id, render_mode="human", reset_to_save_state=False)

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
        elif vision_model == VisionModel.CONV_GRAYSCALE_224:
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


class ConvTrunkGrayscale(nn.Module):
    def __init__(self):
        super().__init__()

        # (E, 4, 84, 84) -> (1, 3136)
        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),

            # Input size for a grayscale observation of size: 84x84
            # (1, 3136) -> (1, 512)
            layer_init(nn.Linear(64 * 7 * 7, 512)),
        )

    def forward(self, x):
        return self.trunk(x)


class ConvTrunkGrayscaleDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.Linear(512, 64 * 7 * 7),  # Match encoder's flatten
            nn.ReLU(),

            nn.Unflatten(1, (64, 7, 7)),  # inverse of flatten
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # -> (64, 26, 26)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),   # -> (32, 54, 54)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, output_padding=4),    # -> (4, 224, 224)
            nn.Sigmoid()  # assume input in [0, 1]
        )

    def forward(self, x):
        return self.deconv(x)   # -> (B, 4, 224, 224)


class ConvTrunkGrayscale224(nn.Module):
    def __init__(self):
        super().__init__()

        # (E, 4, 224, 224) -> (1, 36864)
        self.trunk = nn.Sequential(
            # nn.Conv2d(4, 32, 8, stride=4),     # (B, 32, 55, 55)
            # nn.ReLU(),
            # nn.Conv2d(32, 64, 4, stride=2),    # (B, 64, 26, 26)
            # nn.ReLU(),
            # nn.Conv2d(64, 128, 3, stride=2),   # (B, 128, 12, 12)
            # nn.ReLU(),
            # nn.Conv2d(128, 256, 3, stride=2),  # (B, 256, 5, 5)
            # nn.ReLU(),

            # nn.Flatten(),                      # (B, 6400)

            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),

            # Input size for a grayscale observation of size: 224x224
            # (1, 36864) -> (1, 512)
            #layer_init(nn.Linear(64 * 24 * 24, 512)),
        )

    def forward(self, x):
        return self.trunk(x)


class ConvTrunkGrayscale224Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv = nn.Sequential(
            #nn.Linear(512, 64 * 24 * 24),  # Match encoder's flatten
            #nn.ReLU(),

            nn.Unflatten(1, (64, 24, 24)),  # inverse of flatten
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # -> (64, 26, 26)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),   # -> (32, 54, 54)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, output_padding=4),    # -> (4, 224, 224)
            nn.Sigmoid()  # assume input in [0, 1]


            # nn.Linear(512 * 10, 64 * 24 * 24),  # Match encoder's flatten
            # nn.ReLU(),

            # nn.Unflatten(1, (256, 5, 5)),                         # (B, 256, 5, 5)
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),  # -> (B, 128, 11, 11)
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),   # -> (B, 64, 23, 23)
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),    # -> (B, 32, 48, 48)
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4),     # -> (B, 4, 200, 200)
            # # nn.ReLU(),
            # # nn.Conv2d(4, 4, kernel_size=5, padding=2),              # -> (B, 4, 224, 224)
            # nn.Sigmoid()  # Assumes pixel values normalized to [0, 1]

            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),  # -> (B, 128, 11, 11)
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),   # -> (B, 64, 23, 23)
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),    # -> (B, 32, 48, 48)
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, output_padding=0),     # -> (B, 4, 224, 224)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.deconv(x)   # -> (B, 4, 224, 224)



class Agent(nn.Module):
    def __init__(self, envs, vision_model: VisionModel):
        super().__init__()

        if vision_model == VisionModel.CONV_GRAYSCALE:
            self.trunk = ConvTrunkGrayscale()
            self.decoder = ConvTrunkGrayscaleDecoder()
        elif vision_model == VisionModel.CONV_GRAYSCALE_224:
            self.trunk = ConvTrunkGrayscale224()
            self.decoder = ConvTrunkGrayscale224Decoder()
        else:
            raise AssertionError(f"Unexpected vision model: {vision_model}")

        self.action_dim = envs.single_action_space.n

    def get_action_and_value(self, x, action=None):
        trunk_output = self.trunk(x / 255.0)

        action = torch.randint(low=0, high=self.action_dim, size=(1,))
        return action, trunk_output


def _draw_obs(obs_np, screen: Any, screen_index: int):
    assert obs_np.shape == (224, 224), f"Unexpected observation shape: {obs_np.shape} != (224, 224)"
    assert obs_np.max() < 1.0, f"Unexpected observation values: min={obs_np.min()} max={obs_np.max()}"

    obs_grayscale = (obs_np * 255).astype(np.uint8)
    img_gray = Image.fromarray(obs_grayscale.T, mode='L')
    img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST).convert('RGB')

    screen.blit_image(img_rgb_240, screen_index=screen_index)


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

    if device == torch.device("cpu"):
        # Try mps
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("No GPU available, using CPU.")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.vision_model) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    first_env = envs.envs[0].unwrapped
    screen = first_env.screen
    nes = first_env.unwrapped.nes
    action_dim = envs.single_action_space.n

    print(f"ACTION DIM: {action_dim}")

    # ActorCritic
    agent = Agent(envs, args.vision_model).to(device)
    decoder = agent.decoder

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, eps=1e-5)

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
                action, next_encoded_obs = agent.get_action_and_value(next_obs)

            next_obs_np = next_obs.cpu().numpy()

            # Check that the encoding matches the trunk, like we think.
            if True:
                encoded_output2 = agent.trunk(next_obs / 255)
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
            patience = 5  # Stop if no improvement for this many epochs

            for epoch in range(args.update_epochs):
                executed_epochs += 1

                # Shuffle all observations.
                np.random.shuffle(b_inds)

                # OK: These look random
                # print(f"RANDOM BATCH INDICES: {b_inds}")

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    obs_tensor = b_obs_tensor[mb_inds]
                    encoded_tensor = agent.trunk(obs_tensor)

                    if start < 4:
                        _draw_obs(obs_tensor[0, -1].cpu().numpy(), screen, 4)
                        _draw_obs(obs_tensor[1, -1].cpu().numpy(), screen, 5)
                        _draw_obs(obs_tensor[2, -1].cpu().numpy(), screen, 6)
                        _draw_obs(obs_tensor[3, -1].cpu().numpy(), screen, 7)

                    # Agent image decoder
                    decoded_tensor = decoder(encoded_tensor.detach())
                    decoder_loss = F.mse_loss(decoded_tensor, obs_tensor)

                    # print(f"obs_tensor.shape={obs_tensor.shape} encoded_tensor.shape={encoded_tensor.shape} decoded_tensor.shape={decoded_tensor.shape}")

                    if start < 4:
                        _draw_obs(decoded_tensor[0, -1].detach().cpu().numpy(), screen, 8)
                        _draw_obs(decoded_tensor[1, -1].detach().cpu().numpy(), screen, 9)
                        _draw_obs(decoded_tensor[2, -1].detach().cpu().numpy(), screen, 10)
                        _draw_obs(decoded_tensor[3, -1].detach().cpu().numpy(), screen, 11)

                    # Total loss
                    optimizer.zero_grad()
                    decoder_loss.backward()
                    optimizer.step()

                # ----- Validation -----
                with torch.no_grad():
                    val_encoded = agent.trunk(b_val_obs)
                    val_recon = decoder(val_encoded)

                    val_loss = F.mse_loss(val_recon, b_val_obs)

                print(f"Epoch {epoch+1}: train_loss={decoder_loss:.4f} val_loss={val_loss:.4f}")

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