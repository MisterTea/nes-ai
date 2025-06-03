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

from super_mario_env_search import SuperMarioEnv, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses, get_time_left, life

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
    level_ticks: int
    distance_x: int
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
    headless: bool = False

    # Algorithm specific arguments
    env_id: str = "smb-search-v0"


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, headless: bool):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            render_mode = "rgb" if headless else "human"
            env = gym.make(env_id, render_mode=render_mode, world_level=(4, 4))

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def _seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"


def _print_saves_list(saves: list[SaveInfo]):
    # Determine weighting as a multiple of the first weight.
    weights = _weight_hyperbolic(len(saves))
    weights /= weights[0]

    N = 2

    # Print bottom-N and top-N saves.
    for s, w in zip(saves[:N], weights[:N]):
        print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id}")

    num_top = min(len(saves) - N, N)
    if num_top > 0:
        print('  ...')
        for s, w in zip(saves[-num_top:], weights[-num_top:]):
            print(f"  {w:.4f}x {s.world}-{s.level} x={s.x} y={s.y} save_id={s.save_id}")


def _weight_hyperbolic(N: int) -> np.array:
    # Hyperbolic, last saves have the highest weighting.
    indices = np.arange(N)
    c = 1.0  # Offset to avoid divide-by-zero

    # Weights: highest at the end, slow decay toward the beginning
    # Formula: w_i ∝ 1 / (N - i + c)
    weights = 1.0 / (N - indices + c)
    weights /= weights.sum()  # Normalize to sum to 1

    return weights


def _weight_exp(N: int, beta: float = 0.3) -> np.array:
    indices = np.arange(N)
    weights = np.exp(beta * indices)
    weights /= weights.sum()
    return weights


PATCH_SIZE = 20


def _choose_save(saves: list[SaveInfo]) -> SaveInfo:
    if False:
        # Uniform random
        # return random.choice(saves)

        weights = _weight_hyperbolic(len(saves))
        sample = np.random.choice(saves, p=weights)

    if True:
        # Cluster patches.
        saves_by_patch = {}
        for s in saves:
            patchx_id = (s.world, s.level, s.x // PATCH_SIZE)
            saves_by_patch.setdefault(patchx_id, []).append(s)

        # Select a patch location.
        patches = sorted(saves_by_patch.keys())

        # Choose across x dimension.
        weights = _weight_hyperbolic(len(patches))
        patch_indices = np.arange(len(patches))
        chosen_patch_index = np.random.choice(patch_indices, p=weights)
        chosen_patch = patches[chosen_patch_index]

        # Choose uniformly across y dimension.
        sample = random.choice(saves_by_patch[chosen_patch])

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
        [make_env(args.env_id, 0, args.capture_video, run_name, args.headless)],
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
    level_ticks = get_time_left(ram)
    distance_x = 0

    if False: # x >= 65500:
        print("SOMETHING WENT WRONG WITH CURRENT STATE")
        ticks_left = get_time_left(ram)
        print(f"level={world}-{level} x={x} y={y} ticks-left={ticks_left} states=0 visited={len(visited_patches)}")
        raise AssertionError("STOP")

    saves = [SaveInfo(
        save_id=next_save_id,
        x=x,
        y=y,
        level=level,
        world=world,
        level_ticks=level_ticks,
        distance_x=distance_x,
        save_state=nes.save(),
        visited_patches=visited_patches.copy(),
    )]
    next_save_id += 1
    force_terminate = False

    while True:
        # Remember previous states.
        prev_level = level
        prev_x = x

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

        # If we get teleported, or if the level boundary is discontinuous, the change in x position isn't meaningful.
        if abs(x - prev_x) > 50:
            print(f"Discountinuous x position: {prev_x} -> {x}")
        else:
            distance_x += x - prev_x

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
            lives = life(ram)
            distance_x = save_info.distance_x

            if True:
                print(f"Validate save state:")
                print(f"  world: {save_info.world} =? {world} -> {save_info.world == world}")
                print(f"  level: {save_info.level} =? {level} -> {save_info.level == level}")
                print(f"  x:     {save_info.x} =? {x} -> {save_info.x == x}")
                print(f"  y:     {save_info.y} =? {y} -> {save_info.y == y}")
                print(f"  lives: ??? =? {lives} -> ???")
                assert save_info.world == world, f"Mismatched save state!"
                assert save_info.level == level, f"Mismatched save state!"
                assert save_info.x == x, f"Mismatched save state!"
                assert save_info.y == y, f"Mismatched save state!"

            prev_level = level
            level_ticks = save_info.level_ticks

            if True:
                print(f"Loaded save: save_id={save_info.save_id} level={world}-{level} x={x} y={y} lives={lives}")
                _print_saves_list(saves)

            force_terminate = False

        # If we made progress, save state.
        #   * Assign (level, x, y) patch position to save state.
        #   * Add state to buffer, with vector-time (number of total actions taken)
        else:
            # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
            # Also dump state histogram.
            if level != prev_level:
                print(f"Starting level: {world}-{level}")

                lives = life(ram)
                assert lives > 0 and lives < 100, f"How did we end up with lives?: {lives}"

                visited_patches = set()
                distance_x = 0

                if False: # x >= 65500:
                    print("SOMETHING WENT WRONG WITH CURRENT STATE")
                    ticks_left = get_time_left(ram)
                    print(f"level={world}-{level} x={x} y={y} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)}")
                    raise AssertionError("STOP")

                saves = [SaveInfo(
                    save_id=next_save_id,
                    x=x,
                    y=y,
                    level=level,
                    world=world,
                    level_ticks=level_ticks,
                    distance_x=distance_x,
                    save_state=nes.save(),
                    visited_patches=visited_patches.copy(),
                )]
                next_save_id += 1
                level_ticks = get_time_left(ram)

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
            #
            # Level 1-3 is 3100 units, 300 ticks available ->  10.1 units/tick (min)
            #
            #
            # Level 4-4 (and probably 8-4) are discontinuous.
            # There is a jump from x=200 to x=665535 (max value) at some point.  If the x has
            # changed by more than some large amount (say 100 units), then it means the distance_per_tick
            # metric is invalid.  Instead of looking at pure x value, we need to measure the accumulated x
            # position.
            #
            # Here are sample numbers when using distance, comparing with ticks remaining vs ticks used:
            #   3000 distance remaining / 400 ticks remaining ->  7.5 units/tick
            #   3000 distance remaining / 300 ticks remaining -> 10.0 units/tick
            #   3000 distance remaining / 200 ticks remaining -> 15.0 units/tick
            #   3000 distance remaining / 100 ticks remaining -> 30.0 units/tick
            #
            #   3000 distance remaining /   0 ticks used -> +inf units/tick
            #   3000 distance remaining / 100 ticks used -> 30.0 units/tick
            #   3000 distance remaining / 300 ticks used -> 10.0 units/tick
            #   3000 distance remaining / 400 ticks used ->  7.5 units/tick
            #
            # A fast world might be 3000 distance in 300 ticks.  If we spend half the time waiting around, then
            # the remaining time we need finish in half the ticks:
            #   3000 distance remaining / 300 ticks used -> 10.0 units/ticks (nominal)
            #   3000 distance remaining / 150 ticks used -> 15.0 units/ticks (required)
            #

            ticks_left = get_time_left(ram)
            ticks_used = max(1, level_ticks - ticks_left)

            distance = distance_x
            distance_from_goal = 3000 - distance_x
            distance_per_tick = distance_from_goal / ticks_used

            # Use a ratio of <1.0, because we want to be able to slow down and speed up within a level.
            min_distance_per_tick = 3000 / (300 * 0.5)

            patch_id = (world, level, x // PATCH_SIZE, y // PATCH_SIZE)

            if distance_per_tick < min_distance_per_tick:
                print(f"Ending trajectory, traversal is too slow: x={x} ticks_left={ticks_left} distance={distance} ratio={distance_per_tick:.4f}")
                force_terminate = True
            elif False: # patch_id in visited_patches:
                # TODO(millman): This doesn't work right, because we always start on the same patch.
                #   Maybe need to consider transitioning patches?  But then, we'll always pick working off the frontier,
                #   which isn't right either.

                # We were already here, resample.
                print(f"Ending trajectory, revisited state: x={x} ticks_left={ticks_left} distance={distance} ratio={distance_per_tick:.4f}")
                force_terminate = True
            else:
                if patch_id not in visited_patches:
                    lives = life(ram)

                    valid_lives = lives > 0 and lives < 100
                    valid_x = x < 65500

                    # NOTE: Some levels (like 4-4) are discontinuous.  We can get x values of > 65500.
                    if False: # not valid_x:
                        # TODO(millman): how did we get into a weird x state?  Happens on 4-4.
                        print(f"RAM values: ram[0x006D]={ram[0x006D]=} * 256 + ram[0x0086]={ram[0x0086]=}")
                        print(f"Something is wrong with the x position, don't save this state: level={world}-{level} x={x} y={y} lives={lives} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)}")

                    if not valid_lives:
                        # TODO(millman): how did we get to a state where we don't have full lives?
                        print(f"Something is wrong with the lives, don't save this state: level={world}-{level} x={x} y={y} ticks-left={ticks_left} lives={lives}")

                    if valid_lives and valid_x:
                        saves.append(SaveInfo(
                            save_id=next_save_id,
                            x=x,
                            y=y,
                            level=level,
                            world=world,
                            level_ticks=level_ticks,
                            distance_x=distance_x,
                            save_state=nes.save(),
                            visited_patches=visited_patches.copy(),
                        ))
                        next_save_id += 1
                        visited_patches.add(patch_id)

                    # TODO(millman): dump states

            patches_histogram[patch_id] += 1

        # Print stats every second:
        #   * Current position: (x, y)
        #   * Number of states in memory.
        #   * Elapsed time since level start.
        #   * Novel states found (across all trajectories)
        #   * Novel states/sec
        now = time.time()
        if now - last_print_time > 1.0:
            ticks_left = get_time_left(ram)
            steps_per_sec = step / (now - start_time)

            ticks_used = max(1, level_ticks - ticks_left)

            distance = distance_x
            distance_from_goal = 3000 - distance_x
            distance_per_tick = distance_from_goal / ticks_used

            print(f"{_seconds_to_hms(now-start_time)} level={world}-{level} x={x} y={y} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)} steps/sec={steps_per_sec:.4f} ticks-used:{ticks_used} dist-from-goal:{distance_from_goal} dist-per-tick-required:{distance_per_tick:.4f}")
            last_print_time = now

        step += 1


if __name__ == "__main__":
    main()
