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

from super_mario_env_search import SuperMarioEnv, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses, get_time_left

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


def _print_saves_list(saves: list[SaveInfo]):
    # Determine hyperbolic weighting as a multiple of the first weight.
    N = len(saves)
    indices = np.arange(N)
    c = 1.0
    hyperbolic_weights = 1.0 / (N - indices + c)
    hyperbolic_weights /= hyperbolic_weights[0]

    # Print bottom-10 and top-10 saves.
    for s, w in zip(saves[:10], hyperbolic_weights[:10]):
        print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id}")

    num_top = min(len(saves) - 10, 10)
    if num_top > 0:
        print('  ...')
        for s, w in zip(saves[-num_top:], hyperbolic_weights[-num_top:]):
            print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id}")


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


def _flip_buttons(controller_presses: NdArrayUint8, flip_prob: float, ignore_button_mask: NdArrayUint8) -> NdArrayUint8:
    flip_mask = np.random.rand(8) < flip_prob   # True where we want to flip
    flip_mask[ignore_button_mask] = 0
    result = np.where(flip_mask, 1 - controller_presses, controller_presses)
    return result


_MASK_START_AND_SELECT = _to_controller_presses(['start', 'select']).astype(bool)


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
    last_save_x = x
    force_terminate = False

    while True:
        # Remember previous states.
        prev_level = level

        # Select an action, save in action history.
        controller = _flip_buttons(controller, flip_prob=0.05, ignore_button_mask=_MASK_START_AND_SELECT)

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
        if termination or force_terminate:
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

            prev_level = level
            last_save_x = x

            if True:
                print(f"Loaded save: save_id={save_info.save_id} level={world}-{level} x={x} y={y}")
                _print_saves_list(saves)

            force_terminate = False

        # If we made progress, save state.
        #   * Assign (level, x, y) patch position to save state.
        #   * Add state to buffer, with vector-time (number of total actions taken)
        else:
            # If time left is too short, this creates a bad feedback loop because we can keep
            # dying due to timer.  That would encourage the agent to finish quick, but it may not
            # be possible to actually finish.
            #
            # We use some domain-specific knowledge here that all levels are about 3000-4000 units.
            # If we assume that we can clear 4000 units in 400 timer ticks, then that means we need
            # about 10 units/tick.  If we're too far behind this ratio, avoid saving based on time.
            #
            # For example, level 1-1 starts with 401 ticks (approx seconds) and is 3266 position
            # units long.  The minimum rate at which we can cover ground is 3266 units in 401 ticks,
            # or 3266/401 (8.1).  If we're under this ratio, we won't be able to finish the level.
            #
            # To get the ratio of our advancement, we actually want the number of ticks used, not the
            # number of ticks left.  We assume that all levels have 401 ticks total.
            #
            # Here are some sample numbers:
            #   3266 units / 401 ticks used ->  8.1 units/tick (min)
            #   3266 units / 200 ticks used -> 16.3 units/tick (twice normal rate, good)
            #    100 units /  20 ticks used ->  5.0 units/tick (bad, too slow)
            #    100 units / 100 ticks used ->  1.0 units/tick (bad, too slow)

            ticks_left = get_time_left(ram)
            ticks_used = max(1, 400 - ticks_left)

            distance = x - last_save_x
            distance_per_tick = x / ticks_used
            min_distance_per_tick = 3266 / 400 * 0.5

            patch_id = (world, level, x // 50, y // 50)

            if patch_id not in visited_patches:
                if distance_per_tick >= min_distance_per_tick:
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
                    last_save_x = x
                    visited_patches.add(patch_id)
                else:
                    print(f"Ending trajectory, traversal is too slow: x={x} ticks_left={ticks_left} distance={distance} ratio={distance_per_tick:.4f}")
                    force_terminate = True

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
            print(f"{_seconds_to_hms(now-start_time)} states={len(saves)} level={world}-{level} x={x} y={y} visited={len(visited_patches)}")
            last_print_time = now

        step += 1


if __name__ == "__main__":
    main()
