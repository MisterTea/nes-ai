#!/usr/bin/env python3

import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import tyro

from torch.utils.tensorboard import SummaryWriter

from super_mario_env_search import SuperMarioEnv, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses

from gymnasium.envs.registration import register

register(
    id="smb-search-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=None,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]

@dataclass
class SaveInfo:
    save_id: int
    x: int
    y: int
    level: int
    world: int
    save_state: Any
    visited_patches: set


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

    # Specific experiments
    reset_to_save_state: bool = False

    # Algorithm specific arguments
    env_id: str = "smb-search-v0"


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            env = gym.make(env_id, render_mode="human")

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def _seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"


def _choose_save(saves: list[SaveInfo]) -> SaveInfo:
    # Uniform random
    # return random.choice(saves)

    # Hyperbolic, last saves have the highest weighting.
    N = len(saves)
    indices = np.arange(N)
    c = 1.0  # Offset to avoid divide-by-zero

    # Weights: highest at the end, slow decay toward the beginning
    # Formula: w_i ∝ 1 / (N - i + c)
    weights = 1.0 / (N - indices + c)
    weights /= weights.sum()  # Normalize to sum to 1

    sample = np.random.choice(saves, p=weights)

    return sample


def _flip_buttons(controller_presses: NdArrayUint8, flip_prob: float) -> NdArrayUint8:
    flip_mask = np.random.rand(8) < flip_prob   # True where we want to flip
    return np.where(flip_mask, 1 - controller_presses, controller_presses)


def main():
    args = tyro.cli(Args)

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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name)],
    )

    first_env = envs.envs[0].unwrapped
    nes = first_env.unwrapped.nes
    nes = first_env.nes

    # Global state.
    step = 0
    next_save_id = 0
    start_time = time.time()
    last_print_time = time.time()
    patches_histogram = Counter()

    # Per-trajectory state.  Resets after every death/level.
    action_history = []
    visited_patches = set()
    controller = _to_controller_presses([])

    # Start searching the Mario game tree.
    envs.reset()
    envs.step((controller,))

    ram = nes.ram()
    world = get_world(ram)
    level = get_level(ram)
    x = get_x_pos(ram)
    y = get_y_pos(ram)

    saves = [SaveInfo(
        save_id=next_save_id,
        x=x,
        y=y,
        level=level,
        world=world,
        save_state=nes.save(),
        visited_patches=visited_patches.copy(),
    )]
    next_save_id += 1

    while True:
        # Remember previous states.
        prev_level = level

        # Select an action, save in action history.
        controller = _flip_buttons(controller, flip_prob=0.05)

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((controller,))

        # Update state.
        #   * Add position count to histogram.
        #   * Add action to action history.
        world = get_world(ram)
        level = get_level(ram)
        x = get_x_pos(ram)
        y = get_y_pos(ram)

        action_history.append(controller)

        # If we died, reload from a gamestate based on recency heuristic.
        if termination:
            # Step again so that the environment reset happens before we load.
            envs.step((controller,))

            # Reorder saves across all trajectories by advancement through the game (x pos).
            saves = sorted(saves, key=lambda s: (s.world, s.level, s.x, s.y))

            # Choose save.
            save_info = _choose_save(saves)

            # Reload and re-initialize.
            nes.load(save_info.save_state)
            ram = nes.ram()
            controller[:] = nes.controller1.is_pressed[:]

            visited_patches = save_info.visited_patches.copy()
            action_history = []
            world = get_world(ram)
            level = get_level(ram)
            x = get_x_pos(ram)
            y = get_y_pos(ram)


            # Determine hyperbolic weighting as a multiple of the first weight.
            N = len(saves)
            indices = np.arange(N)
            c = 1.0
            hyperbolic_weights = 1.0 / (N - indices + c)
            hyperbolic_weights /= hyperbolic_weights[0]

            print(f"Loaded save: save_id={save_info.save_id} level={world}-{level} x={x} y={y}")

            # Print bottom-10 and top-10 saves.
            for s, w in zip(saves[:10], hyperbolic_weights[:10]):
                print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id} save={id(s.save_state)}")

            num_top = min(len(saves) - 10, 10)
            if num_top > 0:
                print('  ...')
                for s, w in zip(saves[-num_top:], hyperbolic_weights[-num_top:]):
                    print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id} save={id(s.save_state)}")

            prev_level = level

        # If we made progress, save state.
        #   * Assign (level, x, y) patch position to save state.
        #   * Add state to buffer, with vector-time (number of total actions taken)
        else:
            patch_id = (world, level, x // 50, y // 50)

            if patch_id not in visited_patches:
                saves.append(SaveInfo(
                    save_id=next_save_id,
                    x=x,
                    y=y,
                    level=level,
                    world=world,
                    save_state=nes.save(),
                    visited_patches=visited_patches.copy(),
                ))
                next_save_id += 1
                visited_patches.add(patch_id)

            patches_histogram[patch_id] += 1

        # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
        # Also dump state histogram.
        if level != prev_level:
            print(f"Starting level: {world}-{level}")

            saves = [SaveInfo(
                save_id=next_save_id,
                x=x,
                y=y,
                level=level,
                world=world,
                save_state=nes.save(),
                visited_patches=visited_patches.copy(),
            )]
            next_save_id += 1

            # TODO(millman): dump states

        # Print stats every second:
        #   * Current position: (x, y)
        #   * Number of states in memory.
        #   * Elapsed time since level start.
        #   * Novel states found (across all trajectories)
        #   * Novel states/sec
        now = time.time()
        if now - last_print_time > 1.0:
            print(f"{_seconds_to_hms(now-start_time)} states={len(saves)} level={world}-{level} x={x} y={y} visited={len(visited_patches)})")
            last_print_time = now

        step += 1


if __name__ == "__main__":
    main()
