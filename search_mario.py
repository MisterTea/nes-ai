#!/usr/bin/env python3

import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import tyro

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from super_mario_env_search import SuperMarioEnv, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses, get_time_left, life
from super_mario_env_ram_hacks import encode_world_level

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
    visited_patches_x: set


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
    print_freq_sec: float = 1.0

    # Visualization
    visualize_save_states: bool = True
    vis_freq_sec: float = 0.1

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
            env = gym.make(env_id, render_mode=render_mode, world_level=(8, 1), screen_rc=(2,2))

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
    for i, (s, w) in enumerate(zip(saves[:N], weights[:N])):
        print(f"  [{i}] {w:.4f}x {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id} visited={len(s.visited_patches)} dist={s.distance_x}")

    num_top = min(len(saves) - N, N)
    if num_top > 0:
        print('  ...')
        for e, (s, w) in enumerate(zip(saves[-num_top:], weights[-num_top:])):
            i = len(saves) - num_top + e
            print(f"  [{i}] {w:.4f}x {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id} visited={len(s.visited_patches)} dist={s.distance_x}")


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
        return random.choice(saves)

    if False:
        weights = _weight_hyperbolic(len(saves))
        sample = np.random.choice(saves, p=weights)

    if False:
        # Cluster patches.
        saves_by_patch = {}
        for s in saves:
            patchx_id = (s.world, s.level, s.distance_x // PATCH_SIZE)
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

    if False:
        # Order by most novel, determined by patches explored, along the x dimension.
        saves_ordered_by_patches = sorted(saves, key=lambda s: len(s.visited_patches_x))

        weights = _weight_hyperbolic(len(saves_ordered_by_patches))
        sample = np.random.choice(saves_ordered_by_patches, p=weights)

    if True:
        # Cluster patches along x dimension.
        saves_by_patch = {}
        for s in saves:
            patchx_id = (s.world, s.level, len(s.visited_patches_x))
            saves_by_patch.setdefault(patchx_id, []).append(s)

        # Select a patch location.
        patches = sorted(saves_by_patch.keys())

        # Choose across x dimension.
        weights = _weight_hyperbolic(len(patches))
        patch_indices = np.arange(len(patches))
        chosen_patch_x_index = np.random.choice(patch_indices, p=weights)
        chosen_patch_x = patches[chosen_patch_x_index]

        # Choose uniformly across y dimension.
        sample = random.choice(saves_by_patch[chosen_patch_x])

    return sample


def _flip_buttons(controller_presses: NdArrayUint8, flip_prob: float, ignore_button_mask: NdArrayUint8) -> NdArrayUint8:
    flip_mask = np.random.rand(8) < flip_prob   # True where we want to flip
    flip_mask[ignore_button_mask] = 0
    result = np.where(flip_mask, 1 - controller_presses, controller_presses)
    return result


_MASK_START_AND_SELECT = _to_controller_presses(['start', 'select']).astype(bool)

def _str_level(world_ram: int, level_ram: int) -> str:
    world, level = encode_world_level(world_ram, level_ram)
    return f"{world}-{level}"


def _optimal_patch_layout(screen_width, screen_height, n_patches):
    max_patch_size = 0
    best_cols = best_rows = None

    # Try all possible number of columns from 1 to n_patches (but can't exceed screen_width)
    for n_cols in range(1, n_patches + 1):
        patch_size = screen_width // n_cols
        n_rows = int(np.ceil(n_patches / n_cols))
        total_height = patch_size * n_rows

        if total_height <= screen_height:
            # Maximize patch size
            if patch_size > max_patch_size:
                max_patch_size = patch_size
                best_cols = n_cols
                best_rows = n_rows

    return (best_rows, best_cols, max_patch_size)


class PatchReservoir:

    def __init__(self, max_saves_per_patch: int = 5):
        self.max_saves_per_patch = max_saves_per_patch
        self._saves_by_patches = defaultdict(list)
        self._patch_seen_counts = Counter()

    def add(self, save: SaveInfo):
        patch_id = (save.world, save.level, save.x // PATCH_SIZE, save.y // PATCH_SIZE)

        if self._patch_seen_counts[patch_id] < self.max_saves_per_patch:
            # Reservoir is still small, add it.
            self._saves_by_patches[patch_id].append(save)

        else:
            seen_count = self._patch_seen_counts[patch_id]

            # Random chance of selecting an item in the reservoir.
            k = random.randint(0, seen_count)

            # Kick out the existing item in reservoir.
            if k < self.max_saves_per_patch:
                self._saves_by_patches[patch_id][k] = save

        # Update count.
        self._patch_seen_counts[patch_id] += 1

    def values(self) -> list[SaveInfo]:
        return [
            save
            for saves in self._saves_by_patches.values()
            for save in saves
        ]

    def __len__(self) -> int:
        return len(self._saves_by_patches)


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
    screen = first_env.screen

    # Global state.
    step = 0
    next_save_id = 0
    start_time = time.time()
    last_print_time = time.time()
    last_vis_time = time.time()
    patches_histogram = Counter()

    # Per-trajectory state.  Resets after every death/level.
    action_history = []
    visited_patches = set()
    visited_patches_x = set()
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
    lives = life(ram)

    print(f"LIVES AT START: {life(ram)}")

    if False: # x >= 65500:
        print("SOMETHING WENT WRONG WITH CURRENT STATE")
        ticks_left = get_time_left(ram)
        print(f"level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} states=0 visited={len(visited_patches)}")
        raise AssertionError("STOP")

    saves = PatchReservoir()
    saves.add(SaveInfo(
        save_id=next_save_id,
        x=x,
        y=y,
        level=level,
        world=world,
        level_ticks=level_ticks,
        distance_x=distance_x,
        save_state=nes.save(),
        visited_patches=visited_patches.copy(),
        visited_patches_x=visited_patches_x.copy(),
    ))

    next_save_id += 1
    force_terminate = False
    steps_since_load = 0

    # Configure visualization.

    # Approximate the size of the histogram based on how many patches we need.
    max_level_dist = 6400
    max_patches_x = int(np.ceil(max_level_dist / PATCH_SIZE))
    max_patches_y = int(np.ceil(240 / PATCH_SIZE))

    # We'll wrap rows around if they hit the edge of the screen.
    # Figure out how many wraps we need by:
    #
    #   (actual_screen_w / pixel_size) * num_rows = (actual_screen_w / pixel_size) * (actual_screen_h / pixel_size)
    #
    #   patches_per_row * num_rows = patches_per_row * patches_per_col
    #
    #
    # We want the pixel size to be maximized, as long as everything fits on the screen.
    # When everything fits exactly,
    # We don't know the patches_per_row or the patch_pixel_size.
    hr, hc, pixel_size = _optimal_patch_layout(screen_width=240, screen_height=224, n_patches=max_patches_x * max_patches_y)

    # print(f"PATCH LAYOUT: max_patches_x={max_patches_x} max_patches_y={max_patches_y} pixel_size={pixel_size} hr={hr} hc={hc} screen_w={screen_w} screen_h={screen_h}")

    space_r = 0


    while True:
        # Remember previous states.
        prev_level = level
        prev_x = x
        prev_lives = lives

        # Select an action, save in action history.
        controller = _flip_buttons(controller, flip_prob=0.025, ignore_button_mask=_MASK_START_AND_SELECT)

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((controller,))

        # Read and record state.
        #   * Add position count to histogram.
        #   * Add action to action history.
        world = get_world(ram)
        level = get_level(ram)
        x = get_x_pos(ram)
        y = get_y_pos(ram)
        lives = life(ram)

        action_history.append(controller)

        # If we get teleported, or if the level boundary is discontinuous, the change in x position isn't meaningful.
        if abs(x - prev_x) > 50:
            print(f"Discountinuous x position: {prev_x} -> {x}")
        else:
            distance_x += x - prev_x

        # Calculate derived states.
        ticks_left = get_time_left(ram)
        ticks_used = max(1, level_ticks - ticks_left)

        speed = distance_x / ticks_used
        patches_per_tick = len(visited_patches) / ticks_used

        # TODO(millman): something is broken with the termination flag?
        if lives < prev_lives and not termination:
            print(f"Lost a life: x={x} ticks_left={ticks_left} distance={distance_x}")
            raise AssertionError("Missing termination flag for lost life")

        # If we died, reload from a gamestate based on recency heuristic.
        if termination or force_terminate:
            # Step again so that the environment reset happens before we load.
            envs.step((controller,))

            # Collect saves.
            saves_list = saves.values()

            # Choose save.
            save_info = _choose_save(saves_list)

            # Reload and re-initialize.
            nes.load(save_info.save_state)
            ram = nes.ram()
            controller[:] = nes.controller1.is_pressed[:]

            visited_patches = save_info.visited_patches.copy()
            visited_patches_x = save_info.visited_patches_x.copy()
            action_history = []
            world = get_world(ram)
            level = get_level(ram)
            x = get_x_pos(ram)
            y = get_y_pos(ram)
            lives = life(ram)

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

            prev_x = x
            prev_level = level
            prev_lives = lives
            level_ticks = save_info.level_ticks
            steps_since_load = 0

            ticks_left = get_time_left(ram)
            ticks_used = max(1, level_ticks - ticks_left)
            distance_x = save_info.distance_x

            speed = distance_x / ticks_used
            patches_per_tick = len(visited_patches) / ticks_used

            if True:
                print(f"Loaded save: save_id={save_info.save_id} level={_str_level(world, level)} x={x} y={y} lives={lives}")
                _print_saves_list(saves_list)

            force_terminate = False

        # If we died, skip.
        elif lives < prev_lives:
            print(f"Lost a life: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
            force_terminate = True

        # If we made progress, save state.
        #   * Assign (level, x, y) patch position to save state.
        #   * Add state to buffer, with vector-time (number of total actions taken)
        else:
            # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
            # Also dump state histogram.
            if level != prev_level:
                print(f"Starting level: {_str_level(world, level)}")

                assert lives > 1 and lives < 100, f"How did we end up with lives?: {lives}"

                visited_patches = set()
                visited_patches_x = set()
                distance_x = 0

                if False: # x >= 65500:
                    print("SOMETHING WENT WRONG WITH CURRENT STATE")
                    ticks_left = get_time_left(ram)
                    print(f"level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)}")
                    raise AssertionError("STOP")

                saves = PatchReservoir()
                saves.add(SaveInfo(
                    save_id=next_save_id,
                    x=x,
                    y=y,
                    level=level,
                    world=world,
                    level_ticks=level_ticks,
                    distance_x=distance_x,
                    save_state=nes.save(),
                    visited_patches=visited_patches.copy(),
                    visited_patches_x=visited_patches_x.copy(),
                ))
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
            #   10 distance remaining / 300 ticks used -> 0.03 units/ticks (required)
            #
            # Another way to think about all of this:
            #   * How much distance is left to cover?
            #   * How much time do we have left?
            #   * Distance per time is: speed
            #   * What's mario's max speed?
            #   * If our cumulative speed is too low, abort.
            #
            # World 8-1 is 6400 pixels long, and only 300 seconds

            # Use a ratio of <1.0, because we want to be able to slow down and speed up within a level.
            min_speed = 6400 / 300 * 1.0

            # We want to ensure Mario is finding new states at a reasonable rate, otherwise it means that we're
            # going back over too much ground we've seen before.
            #
            # If each patch is (say) 20 units, and we want mario to find a new one every
            #
            # A level is about 3000 pixels
            # A level is about 3000/20 patches = 150 patches
            # 150 patches in 300 ticks is 0.5 patches/tick
            # That doesn't account for vertical positions too, which will yield extra patches.
            # If Mario jumps, that will cover about 3 patches.

            min_patches_per_tick = 3000 / PATCH_SIZE / 300 * 2.5

            # Wait until we've used some ticks, so that the speed is meaningful.
            if ticks_used > 20 and speed < min_speed:
                print(f"Ending trajectory, traversal is too slow: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
                force_terminate = True
            elif ticks_used > 20 and patches_per_tick < min_patches_per_tick:
                print(f"Ending trajectory, patch discovery rate is too slow: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
                force_terminate = True
            elif False: # patch_id in visited_patches:
                # TODO(millman): This doesn't work right, because we always start on the same patch.
                #   Maybe need to consider transitioning patches?  But then, we'll always pick working off the frontier,
                #   which isn't right either.

                # We were already here, resample.
                print(f"Ending trajectory, revisited state: x={x} ticks_left={ticks_left} distance={distance_x} ratio={speed:.4f}")
                force_terminate = True
            else:
                patch_id = (world, level, x // PATCH_SIZE, y // PATCH_SIZE)

                valid_lives = lives > 1 and lives < 100
                valid_x = True  # x < 65500

                # NOTE: Some levels (like 4-4) are discontinuous.  We can get x values of > 65500.
                if not valid_x:
                    # TODO(millman): how did we get into a weird x state?  Happens on 4-4.
                    print(f"RAM values: ram[0x006D]={ram[0x006D]=} * 256 + ram[0x0086]={ram[0x0086]=}")
                    print(f"Something is wrong with the x position, don't save this state: level={_str_level(world, level)} x={x} y={y} lives={lives} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)}")

                if not valid_lives:
                    # TODO(millman): how did we get to a state where we don't have full lives?
                    print(f"Something is wrong with the lives, don't save this state: level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} lives={lives} steps_since_load={steps_since_load}")
                    raise AssertionError("STOP")

                if valid_lives and valid_x:
                    saves.add(SaveInfo(
                        save_id=next_save_id,
                        x=x,
                        y=y,
                        level=level,
                        world=world,
                        level_ticks=level_ticks,
                        distance_x=distance_x,
                        save_state=nes.save(),
                        visited_patches=visited_patches.copy(),
                        visited_patches_x=visited_patches_x.copy(),
                    ))
                    next_save_id += 1
                    visited_patches.add(patch_id)

                    patch_x_id = (world, level, x // PATCH_SIZE)
                    visited_patches_x.add(patch_x_id)

            patches_histogram[patch_id] += 1

        # Print stats every second:
        #   * Current position: (x, y)
        #   * Number of states in memory.
        #   * Elapsed time since level start.
        #   * Novel states found (across all trajectories)
        #   * Novel states/sec
        now = time.time()
        if now - last_print_time > args.print_freq_sec:
            steps_per_sec = step / (now - start_time)

            ticks_left = get_time_left(ram)
            ticks_used = max(1, level_ticks - ticks_left)

            speed = distance_x / ticks_used

            patches_per_tick = len(visited_patches) / ticks_used
            patches_x_per_tick = len(visited_patches_x) / ticks_used

            print(f"{_seconds_to_hms(now-start_time)} level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} ticks-used={ticks_used} states={len(saves)} visited={len(visited_patches)} steps/sec={steps_per_sec:.4f} speed={speed:.2f} (required={min_speed:.2f}) patches/tick={patches_per_tick:.2f} patches_x/tick={patches_x_per_tick:.2f} steps_since_load={steps_since_load}")
            last_print_time = now

        # Visualize the distribution of save states.
        if args.visualize_save_states and now - last_vis_time > args.vis_freq_sec:

            # Use number of saves as histogram.
            if True:
                patch_histogram = np.zeros((hr + 1, hc + 1), dtype=np.float64)
                for save_patch_id, saves_in_patch in saves._saves_by_patches.items():
                    _patch_world, _patch_level, patch_x, patch_y = save_patch_id

                    # What row of the level we're in.  Wrap around if past the end of the screen.
                    wrap_i = patch_x // hc

                    r = wrap_i * (max_patches_y + space_r) + patch_y
                    c = patch_x % hc

                    try:
                        patch_histogram[r][c] = len(saves_in_patch)
                    except IndexError:
                        print(f"PATCH LAYOUT: max_patches_x={max_patches_x} max_patches_y={max_patches_y} pixel_size={pixel_size} hr={hr} hc={hc}")
                        print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")
                        raise

                #print(f"HISTOGRAM min={patch_histogram.min()} max={patch_histogram.max()}")

                # Normalize counts to range (0, 255)
                grid_f = patch_histogram
                grid_g = (grid_f / grid_f.max() * 255).astype(np.uint8)
                grid_rgb = np.stack([grid_g]*3, axis=-1)

                # Mark current patch.
                _world, _level, px, py = patch_id
                wrap_i = px // hc
                patch_r = wrap_i * (max_patches_y + space_r) + py
                patch_c = px % hc
                grid_rgb[patch_r][patch_c] = (0, 255, 0)

                # print(f"HISTOGRAM_f min={patch_histogram.min()} max={patch_histogram.max()}")

                # Convert to screen.
                img_gray = Image.fromarray(grid_rgb, mode='RGB')
                img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST)

                screen.blit_image(img_rgb_240, screen_index=1)

            # Use seen counts as histogram.
            if True:
                patch_histogram = np.zeros((hr + 1, hc + 1), dtype=np.float64)
                for save_patch_id, seen_counts in saves._patch_seen_counts.items():
                    _patch_world, _patch_level, patch_x, patch_y = save_patch_id

                    # What row of the level we're in.  Wrap around if past the end of the screen.
                    wrap_i = patch_x // hc

                    r = wrap_i * (max_patches_y + space_r) + patch_y
                    c = patch_x % hc

                    try:
                        patch_histogram[r][c] = seen_counts
                    except IndexError:
                        print(f"PATCH LAYOUT: max_patches_x={max_patches_x} max_patches_y={max_patches_y} pixel_size={pixel_size} hr={hr} hc={hc}")
                        print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")
                        raise

                #print(f"HISTOGRAM min={patch_histogram.min()} max={patch_histogram.max()}")

                # Normalize counts to range (0, 255)
                grid_f = patch_histogram
                grid_g = (grid_f / grid_f.max() * 255).astype(np.uint8)
                grid_rgb = np.stack([grid_g]*3, axis=-1)

                # Mark current patch.
                _world, _level, px, py = patch_id
                wrap_i = px // hc
                patch_r = wrap_i * (max_patches_y + space_r) + py
                patch_c = px % hc
                grid_rgb[patch_r][patch_c] = (0, 255, 0)

                # print(f"HISTOGRAM_f min={patch_histogram.min()} max={patch_histogram.max()}")

                # Convert to screen.
                img_gray = Image.fromarray(grid_rgb, mode='RGB')
                img_rgb_240 = img_gray.resize((240, 224), resample=Image.NEAREST)

                screen.blit_image(img_rgb_240, screen_index=3)

            screen.show()

            last_vis_time = now

        step += 1
        steps_since_load += 1


if __name__ == "__main__":
    main()
