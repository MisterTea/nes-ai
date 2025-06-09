#!/usr/bin/env python3

import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import gymnasium as gym
import numpy as np

import tyro

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from super_mario_env_search import SuperMarioEnv, SCREEN_H, SCREEN_W, get_x_pos, get_y_pos, get_level, get_world, _to_controller_presses, get_time_left, life
from super_mario_env_ram_hacks import encode_world_level

from gymnasium.envs.registration import register

register(
    id="smb-search-v0",
    entry_point=SuperMarioEnv,
    max_episode_steps=None,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[3]], np.dtype[np.uint8]]


@dataclass(frozen=True)
class PatchId:
    patch_x: int
    patch_y: int

    def __post_init__(self):
        # Convert value from np.uint8 to int.
        object.__setattr__(self, 'patch_x', int(self.patch_x))
        object.__setattr__(self, 'patch_y', int(self.patch_y))


@dataclass(frozen=True)
class SaveInfo:
    save_id: int
    x: int
    y: int
    level: int
    world: int
    level_ticks: int
    ticks_left: int
    save_state: Any
    action_history: list
    state_history: list
    prev_patch_id: PatchId

    def __post_init__(self):
        # Convert value from np.uint8 to int.
        object.__setattr__(self, 'world', int(self.world))
        object.__setattr__(self, 'level', int(self.level))
        object.__setattr__(self, 'x', int(self.x))
        object.__setattr__(self, 'y', int(self.y))


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
    start_world: int = 7
    start_level: int = 4
    max_trajectory_steps: int = 32 * 4
    patch_size: int = 32

    # Visualization
    vis_freq_sec: float = 0.15

    # Algorithm specific arguments
    env_id: str = "smb-search-v0"


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, headless: bool, world_level: tuple[int, int]):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            raise RuntimeError("STOP")
        else:
            render_mode = "rgb" if headless else "human"
            env = gym.make(env_id, render_mode=render_mode, world_level=world_level, screen_rc=(2,2))

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
        print(f"  [{i}] {w:.4f}x {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id}")

    num_top = min(len(saves) - N, N)
    if num_top > 0:
        print('  ...')
        for e, (s, w) in enumerate(zip(saves[-num_top:], weights[-num_top:])):
            i = len(saves) - num_top + e
            print(f"  [{i}] {w:.4f}x {_str_level(s.world, s.level)} x={s.x} y={s.y} save_id={s.save_id}")


def _weight_hyperbolic(N: int) -> np.array:
    # Hyperbolic, last saves have the highest weighting.
    indices = np.arange(N)
    c = 1.0  # Offset to avoid divide-by-zero

    # Weights: highest at the end, slow decay toward the beginning
    # Formula: w_i ∝ 1 / (N - i + c)
    weights = 1.0 / (N - indices + c)
    weights /= weights.sum()  # Normalize to sum to 1

    return weights

@dataclass(frozen=True)
class ReservoirId:
    patch_x: int
    patch_y: int
    prev_patch_x: int
    prev_patch_y: int


class PatchReservoir:

    def __init__(self, patch_size: int, max_saves_per_reservoir: int = 1):
        self.patch_size = patch_size
        self.max_saves_per_reservoir = max_saves_per_reservoir

        self._reservoir_to_saves = defaultdict(list)
        self._patch_to_reservoir_ids = defaultdict(set)

        self._patch_seen_counts = Counter()
        self._patch_count_since_refresh = Counter()
        self._reservoir_seen_counts = Counter()
        self._reservoir_count_since_refresh = Counter()

    def patch_id_from_save(self, save: SaveInfo) -> tuple:
        patch_id = PatchId(save.x // self.patch_size, save.y // self.patch_size)
        return patch_id

    def reservoir_id_from_save(self, save: SaveInfo) -> tuple:
        reservoir_id = ReservoirId(save.x // self.patch_size, save.y // self.patch_size, save.prev_patch_id.patch_x, save.prev_patch_id.patch_y)
        return reservoir_id

    def add(self, save: SaveInfo):
        patch_id = self.patch_id_from_save(save)
        reservoir_id = self.reservoir_id_from_save(save)

        if self._reservoir_seen_counts[reservoir_id] < self.max_saves_per_reservoir:
            # Reservoir is still small, add it.
            self._reservoir_to_saves[reservoir_id].append(save)

            did_kick_new_item = False

        else:
            # Use traditional reservoir sampling.
            if False:
                seen_count = self._reservoir_seen_counts[reservoir_id]

                # Random chance of selecting an item in the reservoir.
                k = random.randint(0, seen_count)

                # Kick out the existing item in reservoir.
                if k < self.max_saves_per_reservoir:
                    self._saves_by_reservoir[reservoir_id][k] = save

            # Replace the save that took the longest to reach this patch.
            if True:
                if False:
                    # Find the save state with the most action steps.  We assume that it's better to
                    # get to a state with fewer action steps.
                    saves_in_reservoir = self._reservoir_to_saves[reservoir_id]
                    max_index, max_item = max(enumerate(saves_in_reservoir), key=lambda i_and_save: len(i_and_save[1].action_history))

                    # Replace the save state with the most action steps.  We assume that it's better to
                    # get to a state with fewer action steps.
                    if len(save.action_history) < len(max_item.action_history):
                        saves_in_reservoir[max_index] = save

                        self._patch_count_since_refresh[patch_id] -= self._reservoir_count_since_refresh[reservoir_id]
                        # self._patch_count_since_refresh[patch_id] = 0
                        self._reservoir_count_since_refresh[reservoir_id] = 0

                    did_kick_new_item = False

                if True:
                    saves_in_reservoir = self._reservoir_to_saves[reservoir_id]

                    assert len(saves_in_reservoir) == 1, f"Implement for non-single reservoir"

                    # Prefer replacing the save state with more action steps.  Base it on a random chance
                    # so that we maintain some sample diversity.  Treat the difference in action history
                    # as a difference in log odds.  A delta of 1 should be less meaningful if the action
                    # history is very long, compared to very short.

                    # Add the current save.
                    saves_in_reservoir.append(save)

                    # Choose a save to kick out.  Prefer to kick out the largest history.
                    action_counts = np.fromiter((len(s.action_history) for s in saves_in_reservoir), dtype=np.float64)
                    kick_weights = np.square(action_counts)
                    probs = kick_weights / kick_weights.sum()

                    # Pick one according to probabilities
                    choice_index = np.random.choice(len(saves_in_reservoir), p=probs)

                    did_kick_new_item = choice_index == len(saves_in_reservoir) - 1

                    # Remove from items in reservoir.
                    del saves_in_reservoir[choice_index]

                    if did_kick_new_item:
                        # Nothing happens, don't consider this patch refreshed.
                        pass
                    else:
                        # TODO(millman): does refreshing make this too slow?
                        if False:
                            self._patch_count_since_refresh[patch_id] -= self._reservoir_count_since_refresh[reservoir_id]

                            assert self._patch_count_since_refresh[patch_id] >= 0, f"How did we get a negative value?: {self._patch_count_since_refresh[patch_id]}"

                            self._reservoir_count_since_refresh[reservoir_id] = 0

                    assert len(saves_in_reservoir) == 1, f"Implement for non-single reservoir"

        # Update state.
        if not did_kick_new_item:
            self._patch_to_reservoir_ids[patch_id].add(reservoir_id)

        # Check that we don't have an ever-growing number of saves in a patch.
        # The exact number isn't actually that important, as long as we're
        # reasonably bounded.
        #
        # We can get:
        #   - 8 from transitioning from each 1-neighbor into the patch
        #   - 10 from transitioning from each 2-neighbor into the patch,
        #     the 2-neighbor transition can happen because mario can move multiple
        #     pixels sometimes
        #   - 6 from transitioning from a jump point 1 or 2 away
        #
        # Haven't yet seen more than:
        #   - 8 from transitioning from each 1-neighbor into the patch
        #   - 3 from transitioning from each 1-neighbor jump point
        #   - 2 more for some reason?
        #
        max_expected_in_patch = 13
        res_ids_in_patch = self._patch_to_reservoir_ids[patch_id]

        if len(res_ids_in_patch) > max_expected_in_patch:
            print(f"Should be no more than {max_expected_in_patch} items per patch, found: {len(res_ids_in_patch)}")
            print(f"  patch_id={patch_id}")
            for i, res_id in enumerate(res_ids_in_patch):
                print(f"  [{i}] res_id={res_id.prev_patch_x},{res_id.prev_patch_y} -> {res_id.patch_x}, {res_id.patch_y}")

        # Update count.
        self._patch_seen_counts[patch_id] += 1
        self._patch_count_since_refresh[patch_id] += 1

        self._reservoir_seen_counts[reservoir_id] += 1
        self._reservoir_count_since_refresh[reservoir_id] += 1

    def values(self) -> list[SaveInfo]:
        return (
            save
            for saves in self._reservoir_to_saves.values()
            for save in saves
        )

    def __len__(self) -> int:
        return len(self._reservoir_to_saves)


def _choose_save(saves_reservoir: PatchReservoir, rng: Any) -> SaveInfo:
    # Principles:
    #   - primary object: find new states
    #   - when a new state is found, explore it
    #   - when we need to choose among already-explored states, choose the states that
    #     we've explored the least
    #   - do not bake it x-progression into the game; that should happen naturally by preferring
    #     states that reach the same point in fewer actions
    #
    # Ideas:
    #   - Use multi-resolution grid?

    # Collect reservoirs into patches.
    patch_id_to_count = saves_reservoir._patch_count_since_refresh

    patch_id_list = list(patch_id_to_count.keys())

    # Pick patch by Boltzmann-explortation exponential weighting of count.
    patch_counts = np.fromiter(patch_id_to_count.values(), dtype=np.float64)

    # Include weighting based on x position.
    order_counts = np.fromiter((p.patch_x for p in patch_id_list), dtype=np.float64)

    combined_counts = -patch_counts + order_counts

    # Subtract max for numerical stability.
    combined_counts -= combined_counts.max()

    patch_log_weights = combined_counts

    beta = 1.0
    patch_weights = np.exp(beta * combined_counts)
    patch_weights /= patch_weights.sum()

    # Pick patch.
    chosen_patch_index = rng.choice(len(patch_counts), p=patch_weights)
    chosen_patch = patch_id_list[chosen_patch_index]

    # Pick reservoir by exponential weighting.
    reservoir_id_list = list(saves_reservoir._patch_to_reservoir_ids[chosen_patch])
    reservoir_counts = np.fromiter((
        saves_reservoir._reservoir_count_since_refresh[res_id]
        for res_id in reservoir_id_list
    ), dtype=np.float64)

    # Subtract max for numerical stability.
    reservoir_counts -= reservoir_counts.max()

    beta = 1.0
    res_weights = np.exp(beta * -reservoir_counts)
    res_weights /= res_weights.sum()

    # Pick reservoir.
    chosen_res_index = rng.choice(len(reservoir_counts), p=res_weights)
    chosen_res = reservoir_id_list[chosen_res_index]

    # Pick the first item out of the reservoir.
    saves_list = saves_reservoir._reservoir_to_saves[chosen_res]
    assert len(saves_list) == 1, f"Implement sampling within the reservoir"

    sample = saves_list[0]

    return sample, list(zip(patch_id_list, patch_weights))


def _choose_save_from_history(saves: list[SaveInfo], saves_reservoir: PatchReservoir, rng: Any) -> SaveInfo:
    # Determine how many times we've tried the patch that the save is in.

    save_counts = np.asarray([
        saves_reservoir._patch_seen_counts[saves_reservoir.patch_id_from_save(s)]
        for s in saves
    ], dtype=np.float64)

    order_counts = np.arange(len(saves), dtype=np.float64)

    # Weight the save by a combination of exponential decay and visit counts.
    if False:
        log_probs = -save_counts + order_counts

        # Subtract max to avoid overflow.
        log_probs -= log_probs.max()

        beta = 0.3
        probs = np.exp(beta * log_probs)
        probs /= probs.sum()

    # Weight the save by a combination of quadratic decay and visit counts.
    if True:
        combined_counts = -save_counts + order_counts

        # Subtract min to avoid negative values.
        combined_counts -= combined_counts.min()

        probs = np.square(combined_counts + 1)
        probs /= probs.sum()

    if False:
        combined_counts = -save_counts + order_counts

        # Subtract min to avoid negative values.
        combined_counts -= combined_counts.min()

        # Weights: highest at the end, slow decay toward the beginning
        # Formula: w_i ∝ 1 / (N - i + c)
        N = combined_counts.max()
        probs = 1.0 / (N - combined_counts + 1)

        probs /= probs.sum()

    chosen_save_index = rng.choice(len(saves), p=probs)
    sample = saves[chosen_save_index]

    patch_id_and_weight_pairs = [
        (saves_reservoir.patch_id_from_save(s), w)
        for s, w in zip(saves, probs)
    ]

    return sample, patch_id_and_weight_pairs


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


def _build_patch_histogram_rgb(
    patch_id_and_weight_pairs: list[PatchId, int],
    current_patch: PatchId,
    patch_size: int,
    max_patch_x: int,
    max_patch_y: int,
    hist_rows: int,
    hist_cols: int,
    pixel_size: int,
) -> NdArrayRGB8:
    hr, hc = hist_rows, hist_cols

    patch_histogram = np.zeros((hr + 1, hc + 1), dtype=np.float64)

    if False:
        patch_id_and_weight_pairs = list(patch_id_and_weight_pairs)
        print(f"FIRST patch_id_and_weight_pairs: {patch_id_and_weight_pairs[0]}")
        raise AssertionError("DEBUG")

    special_section_offsets = {}
    next_special_section_id = [0]

    # Sometimes we get a discontinuous jump, like:
    #   Discountinuous x position: 1013 -> 65526
    #
    # Seems to happen when crossing boundaries in discontinuous levels like 4-4.
    def _calc_c_for_special_section(check_patch_x: PatchId, hr: int, hc: int):
        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc}")
        # TODO(millman): this isn't right, special section is overlapping other things

        # The section here is the 255-width pixel section that the level is divided into.
        # Also called "screen" in the memory map.
        x = check_patch_x.patch_x * patch_size

        section_x = (x // 256) * 256
        section_patch_x = section_x // patch_size

        offset_x = x - section_x
        offset_patch_x = offset_x // patch_size

        # Determine how many full-screen offsets we need from the edge of the histogram.
        if section_patch_x not in special_section_offsets:
            special_section_offsets[section_patch_x] = next_special_section_id[0]
            next_special_section_id[0] += 1

        section_id = special_section_offsets[section_patch_x]
        patches_in_section = 256 // patch_size

        # Calculate the starting patch of the section.
        section_c = hc - (1 + section_id) * patches_in_section

        # Calculate how much offset we need from the start of the section.
        # Rewrite the x and y position, for display into this after-level section.
        c = section_c + offset_patch_x

        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc} -> c={c}")

        return c


    for save_patch_id, weight in patch_id_and_weight_pairs:
        patch_x, patch_y = save_patch_id.patch_x, save_patch_id.patch_y

        # For display purposes only, if we're at one of the special offscreen locations,
        # Use an offset, but still show it in the histogram.
        if patch_x <= max_patch_x:
            # What row of the level we're in.  Wrap around if past the end of the screen.
            wrap_i = patch_x // hc

            r = wrap_i * (max_patch_y + _SPACE_R) + patch_y
            c = patch_x % hc
        else:
            # Special case, we're past the end of the level for some special section.
            r = hr - max_patch_y + patch_y
            c = _calc_c_for_special_section(save_patch_id, hr=hr, hc=hc)

        try:
            patch_histogram[r][c] = weight
        except IndexError:
            print(f"PATCH LAYOUT: max_patches_x={max_patch_x} max_patches_y={max_patch_y} pixel_size={pixel_size} hr={hr} hc={hc}")
            print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")

            patch_histogram[hr][hc] += 1

    #print(f"HISTOGRAM min={patch_histogram.min()} max={patch_histogram.max()}")

    w_zero = patch_histogram == 0

    # Normalize counts to range (0, 255)
    grid_f = patch_histogram - patch_histogram.min()
    grid_g = (grid_f / grid_f.max() * 255).astype(np.uint8)

    # Reset untouched patches to zero.
    grid_g[w_zero] = 0

    # Convert to RGB,
    grid_rgb = np.stack([grid_g]*3, axis=-1)

    # Mark current patch.
    if current_patch.patch_x <= max_patch_x:
        px, py = current_patch.patch_x, current_patch.patch_y
        wrap_i = px // hc
        patch_r = wrap_i * (max_patch_y + _SPACE_R) + py
        patch_c = px % hc
    else:
        # Special case, we're past the end of the level for some special section.
        patch_r = hr - max_patch_y + current_patch.patch_y
        patch_c = _calc_c_for_special_section(current_patch, hr=hr, hc=hc)

    try:
        grid_rgb[patch_r][patch_c] = (0, 255, 0)
    except IndexError:
        print(f"PATCH LAYOUT: max_patches_x={max_patch_x} max_patches_y={max_patch_y} pixel_size={pixel_size} hr={hr} hc={hc}")
        print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")

        grid_rgb[hr][hc] = (0, 255, 0)

    # Convert to screen.
    img_gray = Image.fromarray(grid_rgb, mode='RGB')
    img_rgb_240 = img_gray.resize((SCREEN_W, SCREEN_H), resample=Image.NEAREST)

    return img_rgb_240


def _validate_save_state(save_info: SaveInfo, ram: NdArrayUint8):
    world = get_world(ram)
    level = get_level(ram)
    x = get_x_pos(ram)
    y = get_y_pos(ram)
    lives = life(ram)
    ticks_left = get_time_left(ram)

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


def _print_info(
    dt: float,
    world: int,
    level: int,
    x: int,
    y: int,
    ticks_left: int,
    ticks_used: int,
    saves: PatchReservoir,
    step: int,
    steps_since_load: int,
):
    steps_per_sec = step / dt

    # screen_x = ram[0x006D]
    # level_screen_x = ram[0x071A]
    # screen_pos = ram[0x0086]

    print(
        f"{_seconds_to_hms(dt)} "
        f"level={_str_level(world, level)} "
        f"x={x} y={y} ticks_left={ticks_left} "
        f"ticks_used={ticks_used} "
        f"states={len(saves)} "
        f"steps/sec={steps_per_sec:.4f} "
        f"steps_since_load={steps_since_load}")


_MAX_LEVEL_DIST = 6400
_SPACE_R = 0


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
    rng = np.random.default_rng()

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name, args.headless, (args.start_world, args.start_level))],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )

    first_env = envs.envs[0].unwrapped
    nes = first_env.nes
    screen = first_env.screen

    # Global state.
    patch_size = args.patch_size
    step = 0
    next_save_id = 0
    start_time = time.time()
    last_print_time = time.time()
    last_vis_time = time.time()

    # Per-trajectory state.  Resets after every death/level.
    action_history = []
    state_history = []
    controller = _to_controller_presses([])

    # Histogram visualization.

    # Approximate the size of the histogram based on how many patches we need.
    _MAX_PATCHES_X = int(np.ceil(_MAX_LEVEL_DIST / patch_size))
    _MAX_EXTRA_PATCHES_X = 5
    _MAX_PATCHES_Y = int(np.ceil(240 / patch_size))
    _NUM_MAX_PATCHES = _MAX_PATCHES_X * _MAX_PATCHES_Y

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
    _HIST_ROWS, _HIST_COLS, _HIST_PIXEL_SIZE = _optimal_patch_layout(screen_width=240, screen_height=224, n_patches=_NUM_MAX_PATCHES)

    # Start searching the Mario game tree.
    envs.reset()

    ram = nes.ram()

    # Initialize to invalid values.  The first step of the loop should act as a new level.
    world = -1
    level = -1
    x = -1
    y = -1
    lives = -1
    ticks_left = -1

    level_ticks = -1

    patch_id = PatchId(x // patch_size, y // patch_size)
    prev_patch_id = patch_id

    saves = PatchReservoir(patch_size=patch_size)
    force_terminate = False
    steps_since_load = 0

    patch_id_and_weight_pairs = []

    while True:
        step += 1
        steps_since_load += 1
        now = time.time()

        # Remember previous states.
        prev_world = world
        prev_level = level
        prev_x = x
        prev_lives = lives

        # Select an action, save in action history.
        controller = _flip_buttons(controller, flip_prob=0.05, ignore_button_mask=_MASK_START_AND_SELECT)

        action_history.append(controller)

        # Execute action.
        _next_obs, reward, termination, truncation, info = envs.step((controller,))

        # Read current state.
        world = get_world(ram)
        level = get_level(ram)
        x = get_x_pos(ram)
        y = get_y_pos(ram)
        lives = life(ram)
        ticks_left = get_time_left(ram)

        patch_id = PatchId(x // patch_size, y // patch_size)

        # Calculate derived states.
        ticks_used = max(1, level_ticks - ticks_left)

        # If we get teleported, or if the level boundary is discontinuous, the change in x position isn't meaningful.
        if abs(x - prev_x) > 50:
            if level != prev_level:
                # print(f"Discountinuous x position, level change: {prev_world},{prev_level} -> {world},{level}, x: {prev_x} -> {x}")
                pass
            elif lives != prev_lives:
                # print(f"Discontinuous x position, died, lives: {prev_lives} -> {lives}")
                pass
            else:
                print(f"Discountinuous x position: {prev_x} -> {x}")

        if False: # x > 65000:
            # 0x006D  Current screen (in which the player is currently in, increase or decrease depending on player's position)
            # 0x0086  Player x position on screen
            # 0x071A  Current screen (in level, always increasing)
            # 0x071C  ScreenEdge X-Position
            screen_x = ram[0x006D]
            level_screen_x = ram[0x071A]
            screen_pos = ram[0x0086]
            print(f"WEIRD X POS: level={_str_level(world, level)} screen[0x006D]={screen_x} level_screen[0x071A]={level_screen_x} screen_pos[0x0086]={screen_pos} x={x} y={y} lives={lives}")

        # TODO(millman): something is broken with the termination flag?
        if lives < prev_lives and not termination:
            print(f"Lost a life: x={x} ticks_left={ticks_left}")
            raise AssertionError("Missing termination flag for lost life")

        # If we died, reload from a game state based on heuristic.
        if termination or force_terminate:
            # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
            # Also dump state histogram.
            if world != prev_world or level != prev_level:
                raise AssertionError("Reached a new world ({prev_world}-{prev_level} -> {world}-{level}), but also terminated?")

            # In AutoresetMode.DISABLED, we have to reset ourselves.
            # Reset only if we hit a termination state.  Otherwise, we can just reload.
            if termination:
                resets_before = first_env.resets
                envs.reset()

                # Don't count this reset by us, we're trying to find where other things are calling reset.
                first_env.resets -= 1

                resets_after = first_env.resets

                # Check that stepping after termination resets appropriately.
                if resets_after == resets_before and level != prev_level:
                    raise AssertionError("Failed to reset on level change")

            # Start reload process.

            # Choose save.
            save_info, patch_id_and_weight_pairs = _choose_save(saves, rng=rng)
            # save_info, patch_id_and_weight_pairs = _choose_save_from_history(state_history, saves, rng=rng)

            # Reload and re-initialize.
            nes.load(save_info.save_state)
            ram = nes.ram()
            controller[:] = nes.controller1.is_pressed[:]

            # Flip the buttons with some probability.  If we're loading a state, we don't want to
            # be required to use the same action state that was tried before.  To get faster coverage
            # We flip buttons here with much higher probability than during a trajectory.
            controller = _flip_buttons(controller, flip_prob=0.3, ignore_button_mask=_MASK_START_AND_SELECT)

            action_history = save_info.action_history.copy()
            state_history = save_info.state_history.copy()

            # Read current state.
            world = get_world(ram)
            level = get_level(ram)
            x = get_x_pos(ram)
            y = get_y_pos(ram)
            lives = life(ram)
            ticks_left = get_time_left(ram)

            if save_info.world != world or save_info.level != level or save_info.x != x or save_info.y != y:
                _validate_save_state(save_info, ram)

            # Set prior frame values to current.  There is no difference at load.
            prev_world = world
            prev_level = level
            prev_x = x
            prev_lives = lives
            patch_id = PatchId(x // patch_size, y // patch_size)
            prev_patch_id = save_info.prev_patch_id

            level_ticks = save_info.level_ticks

            # Update derived state.
            ticks_used = max(1, level_ticks - ticks_left)

            if True:
                print(f"Loaded save: save_id={save_info.save_id} level={_str_level(world, level)}, x={x} y={y} lives={lives}")

                if False:
                    _print_saves_list(saves.values())

            steps_since_load = 0
            force_terminate = False

        # Stop after some fixed number of steps.  This will force the sampling logic to run more often,
        # which means we won't waste as much time running through old states.
        elif args.max_trajectory_steps > 0 and steps_since_load >= args.max_trajectory_steps:
            print(f"Ending trajectory, max steps for trajectory: {steps_since_load}: x={x} ticks_left={ticks_left}")
            force_terminate = True

        # If we died, skip.
        elif lives < prev_lives:
            print(f"Lost a life: x={x} ticks_left={ticks_left}")
            force_terminate = True

        # If we made progress, save state.
        #   * Assign (level, x, y) patch position to save state.
        #   * Add state to buffer, with vector-time (number of total actions taken)
        else:
            # If we reached a new level, serialize all of the states to disk, then clear the save state buffer.
            # Also dump state histogram.
            if world != prev_world or level != prev_level:
                # Print before-level-end info.
                if True:
                    _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, step=step, steps_since_load=steps_since_load)

                # Set number of ticks in level to the current ticks.
                level_ticks = get_time_left(ram)

                # Clear state.
                action_history = []
                state_history = []

                print(f"Starting level: {_str_level(world, level)}")

                # Update derived state.
                ticks_used = max(1, level_ticks - ticks_left)

                # Print after-level-start info.
                if True:
                    _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, step=step, steps_since_load=steps_since_load)

                assert lives > 1 and lives < 100, f"How did we end up with lives?: {lives}"

                saves = PatchReservoir(patch_size=patch_size)
                saves.add(SaveInfo(
                    save_id=next_save_id,
                    x=x,
                    y=y,
                    level=level,
                    world=world,
                    level_ticks=level_ticks,
                    ticks_left=ticks_left,
                    save_state=nes.save(),
                    action_history=action_history.copy(),
                    state_history=state_history.copy(),
                    prev_patch_id=prev_patch_id,
                ))
                next_save_id += 1

            elif patch_id != prev_patch_id:
                valid_lives = lives > 1 and lives < 100
                valid_x = True  # x < 65500

                # NOTE: Some levels (like 4-4) are discontinuous.  We can get x values of > 65500.
                if not valid_x:
                    # TODO(millman): how did we get into a weird x state?  Happens on 4-4.
                    print(f"RAM values: ram[0x006D]={ram[0x006D]=} * 256 + ram[0x0086]={ram[0x0086]=}")
                    print(f"Something is wrong with the x position, don't save this state: level={_str_level(world, level)} x={x} y={y} lives={lives} ticks-left={ticks_left} states={len(saves)} actions={len(action_history)}")
                    raise AssertionError("STOP")

                if not valid_lives:
                    # TODO(millman): how did we get to a state where we don't have full lives?
                    print(f"Something is wrong with the lives, don't save this state: level={_str_level(world, level)} x={x} y={y} ticks-left={ticks_left} lives={lives} steps_since_load={steps_since_load}")
                    raise AssertionError("STOP")

                # Can't avoid visiting a state, because if we jump up and down, we'll get back to the same
                # state we were just on.  TODO(millman): to make this work, need to keep track of frontier?  Otherwise we end up sampling non-sense trajectories?
                # What if we wait until we've hit the same state a few times in a row as a detection of if we're retracing ground?
                if False:
                    # Find the trajectory that got to this state, but took the most actions.
                    saves_in_patch = saves._saves_by_reservoir[patch_id]
                    if saves_in_patch:
                        # We've already been to this patch, get the slowest action history.
                        max_item = max(saves_in_patch, key=lambda save: len(save.action_history))
                        max_action_history = len(max_item.action_history)
                    else:
                        # We haven't been to this patch, use a virtual history that is slower than the current history.
                        max_action_history = len(action_history) + 1

                    # Stop this trajectory if we've already arrived at this state, but with more actions than other trajectories.
                    if len(action_history) >= max_action_history:
                        print(f"Ending trajectory, revisited state: actions {len(action_history)} > {len(max_item.action_history)}: x={x} ticks_left={ticks_left}")
                        force_terminate = True

                else:
                    save_info = SaveInfo(
                        save_id=next_save_id,
                        x=x,
                        y=y,
                        level=level,
                        world=world,
                        level_ticks=level_ticks,
                        ticks_left=ticks_left,
                        save_state=nes.save(),
                        action_history=action_history.copy(),
                        state_history=state_history.copy(),
                        prev_patch_id=prev_patch_id,
                    )
                    saves.add(save_info)
                    next_save_id += 1

                    state_history.append(save_info)

                prev_patch_id = patch_id

            else:
                assert patch_id == prev_patch_id, f"Missed case of transitioning patches: {patch_id} != {prev_patch_id}"

        # Print stats every second:
        #   * Current position: (x, y)
        #   * Number of states in memory.
        #   * Elapsed time since level start.
        #   * Novel states found (across all trajectories)
        #   * Novel states/sec
        if args.print_freq_sec > 0 and now - last_print_time > args.print_freq_sec:
            _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, step=step, steps_since_load=steps_since_load)
            last_print_time = now

        # Visualize the distribution of save states.
        if args.vis_freq_sec > 0 and now - last_vis_time > args.vis_freq_sec:

            # Histogram of number of saves in each patch reservoir.
            # Collect reservoirs into patches.
            patch_id_and_num_saves_pairs = [
                (p_id, len(res_ids))
                for p_id, res_ids in saves._patch_to_reservoir_ids.items()
            ]
            img_rgb_240 = _build_patch_histogram_rgb(
                patch_id_and_num_saves_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=1)

            # Histogram of seen counts.
            patch_id_and_seen_count_pairs = saves._patch_seen_counts.items()
            img_rgb_240 = _build_patch_histogram_rgb(
                patch_id_and_seen_count_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=3)

            # TODO(millman): avoid this?
            if False:
                _sample, patch_id_and_weight_pairs = _choose_save_from_history(state_history, saves, rng=rng)

            # Histogram of sampling weight.
            img_rgb_240 = _build_patch_histogram_rgb(
                patch_id_and_weight_pairs,
                current_patch=patch_id,
                patch_size=patch_size,
                max_patch_x=_MAX_PATCHES_X,
                max_patch_y=_MAX_PATCHES_Y,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=2)

            # Update display.
            screen.show()

            last_vis_time = now


if __name__ == "__main__":
    main()
