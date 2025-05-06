#!/usr/bin/env python3
# Adapted heavily from:
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
#   Docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
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
from super_mario_env_viz import render_mario_pos_policy_value_sweep

from gymnasium.envs.registration import register

register(
    id="SuperMarioBros-mame-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=60 * 60 * 5,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]



USE_OBSERVATION_224x224 = False


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
    """create a value sweep visualization every N updates"""
    visualize_reward: bool = True
    visualize_actions: bool = True

    # Algorithm specific arguments
    env_id: str = "SuperMarioBros-mame-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            env = gym.make(env_id, render_mode="human")

        print(f"RENDER MODE: {env.render_mode}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = LastAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)

        if not USE_OBSERVATION_224x224:
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
        else:
            env = gym.wrappers.ResizeObservation(env, (224, 224))

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

class ResidualMlpBlock(nn.Module):
    def __init__(self, dim, std: float = np.sqrt(2)):
        super().__init__()
        self.block = nn.Sequential(
            layer_init(nn.Linear(dim, dim), std=std),
            nn.ReLU(),
            layer_init(nn.Linear(dim, dim), std=std),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        layers = [
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        ]

        if not USE_OBSERVATION_224x224:
            # Input size for an input observation of size: 84x84
            layers.append(layer_init(nn.Linear(64 * 7 * 7, 512)))
        else:
            # Input size for an input observation of size: 224x224
            layers.append(layer_init(nn.Linear(64 * 24 * 24, 512)))

        # Shared linear layers.
        if False:
            layers += [
                layer_init(nn.Linear(512, 512), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512), std=1),
                nn.ReLU(),
            ]
        else:
            layers += [
                ResidualMlpBlock(512, std=1),
                ResidualMlpBlock(512, std=1),
                ResidualMlpBlock(512, std=1),
                ResidualMlpBlock(512, std=1),
                ResidualMlpBlock(512, std=1),
            ]

        self.network = nn.Sequential(*layers)

        if SMALL_AGENT := False:
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
        else:
            if INIT_WEIGHTS_TO_ZERO := False:
                wc = 0.0
            else:
                wc = None

            self.actor = nn.Sequential(
                layer_init(nn.Linear(512, 256), weight_const=wc, std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(256, 128), weight_const=wc, std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(128, envs.single_action_space.n), weight_const=wc, std=0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(512, 256), weight_const=wc, std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(256, 128), weight_const=wc, std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(128, 1), weight_const=wc, std=1),
            )

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), probs.probs, self.critic(hidden)

    def get_actor_policy_probs_and_critic_value(self, x):
        with torch.no_grad():
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            action_probs = probs.probs

            critic_value = self.critic(hidden)
        return action_probs, critic_value


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
    if not log_scale:
        # Use a linear scale from [rmin, 0] and [0, rmax].  Positive and negative rewards each get
        # their own space of 128 values.
        if reward >= 0:
            reward_vis_value = 128 + reward / rmax * 128
        else:
            reward_vis_value = 128 + reward / rmin * 128
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


def _draw_action_probs2(surf: pygame.Surface, at: int, action_probs: np.ndarray, screen_size: tuple[float, float], block_width: int = 4):
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

    if _DRAW_NEXT := True:
        cell_x_next = (at+1) % num_cell_cols
        cell_y_next = ((at+1) // num_cell_cols) % num_cell_rows

        x_next = cell_x_next * cell_width
        y_next = cell_y_next * cell_height

        # Draw green rectangles across, one block below the current value.
        for i in range(len(action_probs)):
            prob_vis_rgb = pygame.Color(0, 255, 0)
            pygame.draw.rect(surface=surf, color=prob_vis_rgb, rect=(x_next, y_next + i*block_height, block_width, block_height))



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

    screen = envs.envs[0].unwrapped.screen
    nes = envs.envs[0].unwrapped.nes

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Reward visualization.
    surf_np = pygame.surfarray.pixels3d(screen.surfs[3])
    surf_np[:] = (128, 128, 128)
    vis_reward_min = 0
    vis_reward_max = 1
    vis_reward_image_i = 0

    vis_action_probs_i = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _entropy, action_probs, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device, dtype=torch.float32).view(-1)

            if args.visualize_reward:
                vis_reward_min = min(vis_reward_min, reward)
                vis_reward_max = max(vis_reward_max, reward)

                _draw_reward(screen.surfs[3], at=vis_reward_image_i, reward=reward, rmin=vis_reward_min, rmax=vis_reward_max, screen_size=screen.screen_size)

                vis_reward_image_i += 1

            if args.visualize_actions:
                assert action_probs.shape == (1,7), f"Unexpected action_probs shape: {action_probs.shape}"
                action_probs_single_batch = action_probs.squeeze(0)
                # _draw_action_probs(screen.surfs[4], at=vis_action_probs_i, action_probs=action_probs_single_batch, screen_size=screen.screen_size)
                _draw_action_probs2(screen.surfs[4], at=vis_action_probs_i, action_probs=action_probs_single_batch, screen_size=screen.screen_size)

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

            # NOTE: Silent conversion to float32 for Tensor.
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        steps_end = time.time()

        if args.train_agent:
            optimize_networks_start = time.time()

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            executed_epochs = 0
            epochs_start = time.time()

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                executed_epochs += 1

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, _action_probs, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
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

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("Steps/sec:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
            torch.save(agent.state_dict(), f"{run_dir}/agent.ckpt")

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
            screen.blit_image(reward_rgb, screen_index=3)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()