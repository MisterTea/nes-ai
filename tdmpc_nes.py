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
from typing import Any, List, Optional, Union

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
from tensordict.tensordict import TensorDict
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)



from termcolor import colored
from tdmpc2.envs.wrappers.tensor import TensorWrapper
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.common.buffer import Buffer
from tdmpc2.common.logger import Logger
from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed

from nes_ai.last_and_skip_wrapper import LastAndSkipEnv
from super_mario_env import SuperMarioEnv
from super_mario_env_viz import render_mario_pos_policy_value_sweep

from gymnasium.envs.registration import register

register(
    id="smb-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=60 * 60 * 5,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MarioRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_run_id: str | None = None
    """the id of a wandb run to resume"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_frequency: float = 30
    """create a checkpoint every N seconds"""
    train_agent: bool = True
    """enable or disable training of the agent"""

    # Visualization
    value_sweep_frequency: int | None = 0

    visualize_decoder: bool = True

    """create a value sweep visualization every N updates"""
    visualize_reward: bool = True
    visualize_actions: bool = True
    visualize_intrinsic_decoder: bool = True
    visualize_intrinsic_reward: bool = True

    # Specific experiments
    dump_trajectories: bool = False
    reset_to_save_state: bool = False

    # Algorithm specific arguments
    env_id: str = "smb-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    learning_rate_decoder: float = 2.5e-4
    """the learning rate of the decoder optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    intrinsic_reward_coef: float = 1e-3
    """coefficient of the intrinsic reward when combining with extrinsic reward"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


@dataclass
class Config:
    # Defaults
    # defaults: List[dict] = field(default_factory=lambda: [{"override hydra/launcher": "submitit_local"}])

    # Environment
    task: str = "smb"
    obs: str = "rgb"
    episodic: bool = True

    # Evaluation
    checkpoint: Optional[str] = None
    eval_episodes: int = 10
    eval_freq: int = 50000

    # Training
    steps: int = 10_000_000
    batch_size: int = 256
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1
    consistency_coef: float = 20
    rho: float = 0.5
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20
    tau: float = 0.01
    discount_denom: int = 5
    discount_min: float = 0.95
    discount_max: float = 0.995
    buffer_size: int = 250_000
    exp_name: str = "default"
    data_dir: Optional[str] = None

    # Planning
    mpc: bool = True
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    horizon: int = 3
    min_std: float = 0.05
    max_std: float = 2
    temperature: float = 0.5

    # Actor
    log_std_min: float = -10
    log_std_max: float = 2
    entropy_coef: float = 1e-4

    # Critic
    num_bins: int = 101
    vmin: float = -10
    vmax: float = 10

    # Architecture
    model_size: Optional[str] = None
    num_enc_layers: int = 2
    enc_dim: int = 256
    num_channels: int = 32
    mlp_dim: int = 512
    latent_dim: int = 512
    task_dim: int = 96
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8

    # Logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_silent: bool = False
    enable_wandb: bool = False
    save_csv: bool = False

    # Misc
    compile: bool = False
    save_video: bool = False
    save_agent: bool = False
    seed: int = 1

    # Convenience
    work_dir: Optional[str] = None
    task_title: Optional[str] = None
    multitask: Optional[Union[str, bool]] = False
    tasks: Optional[Any] = None
    obs_shape: Optional[Any] = None
    action_dim: Optional[Any] = None
    episode_length: Optional[Any] = None
    obs_shapes: Optional[Any] = None
    action_dims: Optional[Any] = None
    episode_lengths: Optional[Any] = None
    seed_steps: Optional[Any] = None
    bin_size: Optional[Any] = None
    num_discrete_actions: Optional[Any] = None


from gymnasium.wrappers import TransformObservation
from gymnasium.core import ActType, ObsType, WrapperObsType

class PermuteWHCtoCHWObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Reshapes Array based observations to a specified shape.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> reshape_env = ReshapeObservation(env, (24, 4, 96, 1, 3))
        >>> reshape_env.observation_space.shape
        (24, 4, 96, 1, 3)

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        # assert isinstance(env.observation_space, spaces.Box)
        # assert np.prod(shape) == np.prod(env.observation_space.shape)

        # assert isinstance(shape, tuple)
        # assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        # assert all(x > 0 or x == -1 for x in shape)

        # new_observation_space = spaces.Box(
        #     low=np.reshape(np.ravel(env.observation_space.low), shape),
        #     high=np.reshape(np.ravel(env.observation_space.high), shape),
        #     shape=shape,
        #     dtype=env.observation_space.dtype,
        # )
        # self.shape = shape
        new_observation_space = env.observation_space

        gym.utils.RecordConstructorArgs.__init__(self)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: obs.transpose(2, 1, 0),
            observation_space=new_observation_space,
        )


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            env = gym.make(env_id, render_mode="human", reset_to_save_state=False, screen_rc=(2,3))

        print(f"RENDER MODE: {env.render_mode}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = LastAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)

        # Specific size required by TDMPC for rgb.
        env = gym.wrappers.ResizeObservation(env, (64, 64))

        # env = PermuteWHCtoCHWObservation(env)

        # env = gym.wrappers.FrameStackObservation(env, 4)

        # Required for TDMPC2
        env = TensorWrapper(env)

        return env

    return thunk


def _loglike_scale(x: float, lower: float, upper: float, s: float = 1.0) -> int:
    # s = 0.1 -> linear from ~–50 to +50
    # s = 1 -> linear from ~–5 to +5
    # s = 10 -> linear from ~–0.5 to +0.5

    half_dim = (upper - lower) / 2.0
    center = lower + (upper - lower) / 2.0
    return center + (half_dim * (2 / np.pi) * np.atan(x))


def _sigmoid(x: float, k: float = 1.0):
    return 1 / (1 + np.exp(-k * x))


def _sigmoid_scale(x: float, lower: float, upper: float, k: float = 1.0):
    scale = upper - lower
    sigmoid = 1 / (1 + np.exp(-k * x))
    return lower + scale * sigmoid


def _draw_reward(surf: pygame.Surface, at: int, reward: float, rmin: float, rmax: float, screen_size: tuple[float, float], block_size: int = 10, log_scale: bool = False):
    assert type(reward) in [int, float], f"Unexpected type: {reward=}"
    if not log_scale:
        max_abs_reward = max(abs(reward), max(abs(rmin), abs(rmax)))

        # Use a linear scale from [rmin, 0] and [0, rmax].  Positive and negative rewards each get
        # their own space of 128 values.
        if reward >= 0:
            reward_vis_value = 128 + abs(reward / max_abs_reward) * 128
        else:
            reward_vis_value = 128 - abs(reward / max_abs_reward) * 128
        reward_vis_value = int(max(0, min(255, reward_vis_value)))
    else:
        # reward_normalized = (reward - rmin) / rmax
        # log_reward = np.log(reward_normalized)
        # log_reward_max = np.log(rmax)
        # reward_vis_value = log_reward / log_reward_max * 255
        # reward_vis_value = int(max(0, min(255, reward_vis_value)))

        # reward_vis_value = int(_loglike_scale(reward, lower=0, upper=255))
        # reward_vis_value = int(_sigmoid_scale(reward, lower=0, upper=255))

        reward_normalized = (reward - rmin / rmax)
        reward_vis_value = int(_sigmoid(reward_normalized) * 255)

        # reward_vis_value = np.random.randint(0, 255)

    w, h = screen_size
    bw = w // block_size
    bh = h // block_size

    if _DRAW_CURRENT := True:
        # Draw vertical columns before moving on to the next column.  This feels natural since time
        # flows left-to-right, like the visualization.
        bx = (at // bw) % bh
        by = at % bw

        x = bx * block_size
        y = by * block_size

        if False:
            # Directly set the pixel values.
            surf_np = pygame.surfarray.pixels3d(surf)
            surf_np[x:x + block_size, y:y+block_size] = reward_vis_value
        else:
            # Draw a rectangle.
            reward_vis_rgb = pygame.Color(reward_vis_value, reward_vis_value, reward_vis_value)
            pygame.draw.rect(surface=surf, color=reward_vis_rgb, rect=(x, y, block_size, block_size))

    if _DRAW_NEXT := True:
        bx_next = ((at+1) // bw) % bh
        by_next = (at+1) % bw

        x_next = bx_next * block_size
        y_next = by_next * block_size

        # Draw the next block as green, just to follow where it's drawing.
        reward_vis_rgb = pygame.Color(0, 255, 0)
        pygame.draw.rect(surface=surf, color=reward_vis_rgb, rect=(x_next, y_next, block_size, block_size))


def _draw_action_probs(surf: pygame.Surface, at: int, action_probs: np.ndarray, screen_size: tuple[float, float], block_width: int = 10, block_height: int = 10):
    """Draw multiple columns of action probabilities.  Sweep each horizontal block top-to-bottom."""

    num_actions = len(action_probs)

    # Each "cell" is made up of N blocks across, and 1 block down.  Add an extra block as a spacer.
    cell_width = block_width * (num_actions + 1)
    cell_height = block_height * 1

    w, h = screen_size

    num_cell_rows = h // cell_height
    num_cell_cols = w // cell_width

    if _DRAW_CURRENT := True:
        # Calculate cell x,y position.
        cell_x = (at // num_cell_rows) % num_cell_cols
        cell_y = at % num_cell_rows

        x = cell_x * cell_width
        y = cell_y * cell_height

        # Draw rectangles across.
        for i, prob in enumerate(action_probs):
            prob_vis_gray = int(prob * 255)
            prob_vis_rgb = pygame.Color(prob_vis_gray, prob_vis_gray, prob_vis_gray)
            pygame.draw.rect(surface=surf, color=prob_vis_rgb, rect=(x + i*block_width, y, block_width, block_height))

    if _DRAW_NEXT := True:
        cell_x_next = ((at+1) // num_cell_rows) % num_cell_cols
        cell_y_next = (at+1) % num_cell_rows

        x_next = cell_x_next * cell_width
        y_next = cell_y_next * cell_height

        # Draw green rectangles across, one block below the current value.
        for i in range(len(action_probs)):
            prob_vis_rgb = pygame.Color(0, 255, 0)
            pygame.draw.rect(surface=surf, color=prob_vis_rgb, rect=(x_next + i*block_width, y_next, block_width, block_height))


def _draw_action_probs2(surf: pygame.Surface, at: int, action_probs: np.ndarray, action_index: int, screen_size: tuple[float, float], block_width: int = 4):
    """Draw a single column of action probabilities.  Sweep left-to-right."""

    num_actions = len(action_probs)

    w, h = screen_size

    block_height = h // num_actions

    # Each "cell" is made up of 1 block across, and N blocks down.
    cell_width = block_width
    cell_height = block_height * num_actions

    num_cell_rows = h // cell_height
    num_cell_cols = w // cell_width

    if _DRAW_CURRENT := True:
        # Calculate cell x,y position.
        cell_x = at % num_cell_cols
        cell_y = (at // num_cell_cols) % num_cell_rows

        x = cell_x * cell_width
        y = cell_y * cell_height

        # Draw rectangles across.
        for i, prob in enumerate(action_probs):
            prob_vis_gray = int(prob * 255)
            prob_vis_rgb = pygame.Color(prob_vis_gray, prob_vis_gray, prob_vis_gray)
            pygame.draw.rect(surface=surf, color=prob_vis_rgb, rect=(x, y + i*block_height, block_width, block_height))

        # Draw selected action as a small circle inside the cell.
        selected_rgb = pygame.Color(int(0.67 * 255), 0, 0)
        center = (
            int(x + block_width/2),
            int(y + block_height/2 + action_index*block_height),
        )
        radius = int(min(block_width/2, block_height/2))
        pygame.draw.circle(surface=surf, color=selected_rgb, center=center, radius=radius, width=1)

    if _DRAW_NEXT := True:
        cell_x_next = (at+1) % num_cell_cols
        cell_y_next = ((at+1) // num_cell_cols) % num_cell_rows

        x_next = cell_x_next * cell_width
        y_next = cell_y_next * cell_height

        # Draw green rectangles across, one block below the current value.
        for i in range(len(action_probs)):
            prob_vis_rgb = pygame.Color(0, 255, 0)
            pygame.draw.rect(surface=surf, color=prob_vis_rgb, rect=(x_next, y_next + i*block_height, block_width, block_height))


def to_td(env, obs, action=None, reward=None, terminated=None):
    """Creates a TensorDict for a new episode."""
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device='cpu')
    else:
        # print(f"OBSERVATION SIZE: {obs.shape}")
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        rand_act = env.rand_act()

        # print(f"SIZE OF RAND_ACT: {rand_act}")
        # raise AssertionError("STOP")

        # action = torch.full_like(rand_act, float('nan'))
        action = torch.tensor(-1)

    if reward is None:
        reward = torch.tensor(float('nan'))
    if terminated is None:
        terminated = torch.tensor(float('nan'))

    # print(f"to_td: obs={obs.shape} action={action.shape} reward={reward.shape} terminated={terminated.shape}")

    assert action.dim() == 0, f"Unexpected action shape: {action.shape}, action={action}"
    #assert action.shape == (1,), f"Unexpected action shape: {action.shape}, action={action}"

    td = TensorDict(
        obs=obs,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )

    # print(f"to_td, shape: {td.shape}")

    return td


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

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            #name=run_name,
            monitor_gym=True,
            save_code=True,
            id=run_name,
        )
        assert run.dir == f"runs/{run_name}"
        run_dir = run.dir
    else:
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    first_env = envs.envs[0].unwrapped
    screen = first_env.screen
    nes = first_env.unwrapped.nes

    action_dim = 1 # envs.single_action_space.n
    env = envs.envs[0]

    print(f"ACTION DIM: {action_dim}")

    cfg = Config()

    #try: # Dict
    #	cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    #except: # Box

    #cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.obs_shape = {
        # cfg.get('obs', 'state'): env.observation_space.shape

        # (1, 4, 64, 64, 3)
        'rgb': (3, 64, 64),
    }
    print(f"env.action_space.shape: {envs.action_space.shape=}")
    # cfg.action_dim = env.action_space.shape[0]

    # NOTE: We're saying the action dimension for the agent is 7, even though it's really 1 dimensional with 7 values.
    #   TD-MPC2 was designed for continuous action spaces, so we'll use a 1-hot-encoded version to mimic continuous space.
    cfg.action_dim = 1

    cfg.num_discrete_actions = env.action_space.n


    cfg.episode_length = 60 * 60 * 5 # env.max_episode_steps
    cfg.seed_steps = 3 # max(1000, 5*cfg.episode_length)

    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    # Agent
    agent = TDMPC2(cfg, device=device)
    buffer = Buffer(cfg, device=device)
    logger = Logger(cfg)

    # Reward visualization.
    surf_np = pygame.surfarray.pixels3d(screen.surfs[3])
    surf_np[:] = (128, 128, 128)
    vis_reward_min = 0
    vis_reward_max = 0
    vis_reward_image_i = 0

    vis_action_probs_i = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    if args.track and run.resumed:
        # Reference example (that seems out of date) from: https://docs.cleanrl.dev/advanced/resume-training/#resume-training_1
        # Updated with example from: https://wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Machine-Learning-Models-with-W-B--Vmlldzo3MDQ3Mw

        starting_iter = run.starting_step
        global_step = starting_iter * args.batch_size
        model = run.restore('files/agent.ckpt')

        agent.load_state_dict(torch.load(model.name, map_location=device))

        agent.eval()

        print(f"Resumed at update {starting_iter}")
    else:
        starting_iter = 1

    # Initialize last checkpoint time.  We don't want to checkpoint again for a bit.
    last_checkpoint_time = time.time()

    # TDMPC2 members
    _tds = []
    train_metrics, done, eval_next = {}, True, False

    for step in range(0, args.total_timesteps):
        iteration = step

        print(f"Iteration: {iteration}")
        global_step += args.num_envs

        # Evaluate agent periodically
        if step % cfg.eval_freq == 0:
            eval_next = True

        # Reset environment
        if done:
            if False: # eval_next:
                eval_metrics = eval()
                eval_metrics.update(common_metrics())
                logger.log(eval_metrics, 'eval')
                eval_next = False

            if step > 0:
                if False: #info['terminated'] and not cfg.episodic:
                    raise ValueError('Termination detected but you are not in episodic mode. ' \
                    'Set `episodic=true` to enable support for terminations.')

                if False:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in _tds[1:]]).sum(),
                        episode_success=info['success'],
                        episode_length=len(_tds),
                        episode_terminated=info['terminated'])
                    train_metrics.update(common_metrics())
                    logger.log(train_metrics, 'train')

                if False:
                    print(f"torch.cat on len: {len(_tds)}")
                    for i, td in enumerate(_tds):
                        for k, v in td.items():
                            print(f"GOT TENSOR[{i}]: {k}: {v.shape}")

                print("ADDED TO BUFFER")
                _ep_idx = buffer.add(torch.cat(_tds))

            obs, _info = env.reset()
            obs = torch.Tensor(obs).to(device)

            _tds = [to_td(env, obs)]

        # Collect experience
        if True: # step > cfg.seed_steps:
            action_index = agent.act(obs, t0=len(_tds)==1)
        else:
            action_index = env.rand_act()

        # print(f"ACTION INDEX: {action_index}")

        obs, reward, done, truncated, info = env.step(action_index)
        obs = torch.Tensor(obs).to(device)


        # print(f"OBS SIZE: {obs.shape}")

        _tds.append(to_td(env, obs, action_index, reward, info['terminated']))

        # Update agent
        if step > cfg.seed_steps and buffer.num_eps > 0:
            if step == cfg.seed_steps:
                num_updates = cfg.seed_steps
                print('Pretraining agent on seed data...')
            else:
                num_updates = 1

            for _ in range(num_updates):
                _train_metrics = agent.update(buffer)

                for key, value in _train_metrics.items():
                    print(f"_train_metrics[{key}]: {value}")

            if False:
                train_metrics.update(_train_metrics)


        if args.visualize_reward:
            vis_reward_min = min(vis_reward_min, reward)
            vis_reward_max = max(vis_reward_max, reward)

            _draw_reward(screen.surfs[3], at=vis_reward_image_i, reward=reward.item(), rmin=vis_reward_min, rmax=vis_reward_max, screen_size=screen.screen_size)

            vis_reward_image_i += 1

        if False: # args.visualize_actions:
            assert action_probs.shape == (1,7), f"Unexpected action_probs shape: {action_probs.shape}"
            action_probs_single_batch = action_probs.squeeze(0)
            # _draw_action_probs(screen.surfs[4], at=vis_action_probs_i, action_probs=action_probs_single_batch, screen_size=screen.screen_size)
            _draw_action_probs2(screen.surfs[4], at=vis_action_probs_i, action_probs=action_probs_single_batch, action_index=action, screen_size=screen.screen_size)

            vis_action_probs_i += 1

        if pygame.K_v in nes.keys_pressed:
            start_vis = time.time()
            print("Generating value sweep...")
            policy_sweep_rgb, values_sweep_rgb = render_mario_pos_policy_value_sweep(envs=envs, device=device, agent=agent)
            screen.blit_image(values_sweep_rgb, screen_index=1)
            screen.blit_image(policy_sweep_rgb, screen_index=2)
            print(f"Generated value sweep: {time.time()-start_vis:.4f}s")

        if nes.keys_pressed:
            nes.keys_pressed = []

        if False:
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # Checkpoint.
        seconds_since_last_checkpoint = time.time() - last_checkpoint_time
        if args.checkpoint_frequency > 0 and seconds_since_last_checkpoint > args.checkpoint_frequency:
            seconds_since_last_checkpoint = time.time() - last_checkpoint_time
            print(f"Checkpoint at iter: {iteration}, since last checkpoint: {seconds_since_last_checkpoint:.2f}s")
            start_checkpoint = time.time()

            # NOTE: The run.dir location includes a 'files/' suffix.
            #
            # E.g. 'agent.cpkt' will be saved to:
            #   /Users/dave/rl/nes-ai/wandb/run-20250418_130130-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30/files/agent.ckpt
            #
            agent.save(f"{run_dir}/agent.ckpt")

            if args.track:
                wandb.save(f"{run_dir}/agent.ckpt", policy="now")

            print(f"Checkpoint done: {time.time() - start_checkpoint:.4f}s")

            # Reset the checkpoint time, so we don't include the amount of time necessary to perform
            # the checkpoint itself.
            last_checkpoint_time = time.time()

        # Show value sweep.
        if args.value_sweep_frequency and iteration % args.value_sweep_frequency == 0:
            policy_sweep_rgb, values_sweep_rgb = render_mario_pos_policy_value_sweep(envs=envs, device=device, agent=agent)
            screen.blit_image(values_sweep_rgb, screen_index=1)
            screen.blit_image(policy_sweep_rgb, screen_index=2)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()