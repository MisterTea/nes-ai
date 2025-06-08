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
import torch

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
    distance_x: int
    ticks_left: int
    save_state: Any
    visited_patches: set
    visited_patches_x: set
    action_history: list
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
    start_world: int = 8
    start_level: int = 4

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


PATCH_SIZE = 10


@dataclass(frozen=True)
class ReservoirId:
    patch_x: int
    patch_y: int
    prev_patch_x: int
    prev_patch_y: int


class PatchReservoir:

    def __init__(self, max_saves_per_reservoir: int = 1):
        self.max_saves_per_reservoir = max_saves_per_reservoir
        self._saves_by_reservoir = defaultdict(list)
        self._reservoir_seen_counts = Counter()
        self._patch_seen_counts = Counter()
        self._reservoir_refreshed = set()
        self._reservoir_count_since_refresh = Counter()

    @staticmethod
    def patch_id_from_save(save: SaveInfo) -> tuple:
        patch_id = PatchId(save.x // PATCH_SIZE, save.y // PATCH_SIZE)
        return patch_id

    @staticmethod
    def reservoir_id_from_save(save: SaveInfo) -> tuple:
        reservoir_id = ReservoirId(save.x // PATCH_SIZE, save.y // PATCH_SIZE, save.prev_patch_id.patch_x, save.prev_patch_id.patch_y)
        return reservoir_id

    @staticmethod
    def reservoir_id_from_state(world: int, level: int, patch_id: PatchId, prev_patch_id: PatchId) -> ReservoirId:
        reservoir_id = ReservoirId(patch_id.patch_x, patch_id.patch_y, prev_patch_id.patch_x, prev_patch_id.patch_y)
        return reservoir_id

    def add(self, save: SaveInfo):
        patch_id = self.patch_id_from_save(save)
        reservoir_id = self.reservoir_id_from_save(save)

        if self._reservoir_seen_counts[reservoir_id] < self.max_saves_per_reservoir:
            # Reservoir is still small, add it.
            self._saves_by_reservoir[reservoir_id].append(save)

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
                # Find the save state with the most action steps.  We assume that it's better to
                # get to a state with fewer action steps.
                saves_in_patch = self._saves_by_reservoir[reservoir_id]
                max_index, max_item = max(enumerate(saves_in_patch), key=lambda i_save: len(i_save[1].action_history))

                # Replace the save state with the most action steps.  We assume that it's better to
                # get to a state with fewer action steps.
                if len(save.action_history) < len(max_item.action_history):
                    saves_in_patch[max_index] = save

                    # Mark this patch as newly refreshed.
                    self._reservoir_refreshed.add(reservoir_id)

                    # TODO(millman): is this a good idea?
                    # Don't increment since it was replaced.
                    # self._reservoir_count_since_refresh[reservoir_id] -= 1

        # Update count.
        self._reservoir_seen_counts[reservoir_id] += 1
        self._patch_seen_counts[patch_id] += 1
        self._reservoir_count_since_refresh[reservoir_id] += 1

    def values(self) -> list[SaveInfo]:
        return [
            save
            for saves in self._saves_by_reservoir.values()
            for save in saves
        ]

    def __len__(self) -> int:
        return len(self._saves_by_reservoir)


def _choose_save(saves_reservoir: PatchReservoir) -> SaveInfo:
    # Collect saves.
    saves = saves_reservoir.values()

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

    if False:
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

    if False:
        # Organize into 3 buckets:
        #   1.  Newly visited patches.
        #   2.  Newly refreshed patches (i.e. got to the patch in a faster time).
        #   3.  Everything else.
        #
        # Within each bucket, prefer the states with more visited_patches_x, which corresponds to
        # progress.
        saves_by_patch_new = []
        saves_by_patch_refreshed = []
        saves_by_patch_other = []
        for s in saves:
            reservoir_id = saves_reservoir.reservoir_id_from_save(s)
            if saves_reservoir._reservoir_seen_counts[reservoir_id] == 1:
                # Newly visited, only encountered this patch only once.
                saves_by_patch_new.append(s)
            elif reservoir_id in saves_reservoir._reservoir_refreshed:
                # Newly refreshed.
                saves_by_patch_refreshed.append(s)
            else:
                saves_by_patch_other.append(s)

        # Pick bucket in priority order.
        if saves_by_patch_new:
            saves_bucket = saves_by_patch_new
        elif saves_by_patch_new:
            saves_bucket = saves_by_patch_refreshed
        else:
            saves_bucket = saves_by_patch_other

        # Cluster patches along x dimension.
        saves_by_patch = {}
        for s in saves_bucket:
            patchx_id = (s.world, s.level, len(s.visited_patches_x))
            saves_by_patch.setdefault(patchx_id, []).append(s)

        # Select a patch location.
        patches = sorted(saves_by_patch.keys())

        # Choose across x dimension.
        weights = _weight_hyperbolic(len(patches))
        patch_indices = np.arange(len(patches))
        chosen_patch_x_index = np.random.choice(patch_indices, p=weights)
        chosen_patch_x = patches[chosen_patch_x_index]

        # TODO(millman): Choose across y dimension based on counts.  We want to explore the under-explored areas.
        # Choose uniformly across y dimension.
        sample = random.choice(saves_by_patch[chosen_patch_x])

        # TODO(millman): return weights for all patches?, for visualization
        return sample, [(saves_reservoir.patch_id_from_save(saves_bucket[0]), 1)]

    if True:
        # Use Boltzmann Exploration.
        #
        # Maybe upgrade to Boltzmann-Gumbel explortation later?:
        # https://proceedings.neurips.cc/paper_files/paper/2017/file/b299ad862b6f12cb57679f0538eca514-Paper.pdf

        # Focus on the least-visited patches, independent of anything else.  We can think about the
        # exploration problem as wanting full coverage -- we have no reward or other idea about how
        # we should expand states.

        if True:
            # Pentalize counts.  Explore lower count states.
            reservoir_id_list = list(saves_reservoir._reservoir_count_since_refresh.keys())
            counts = np.fromiter(saves_reservoir._reservoir_count_since_refresh.values(), dtype=np.float64)
        else:
            # Reward progress.  Explore higher count states.
            reservoir_id_list = []
            counts = []
            for r_id, saves_in_reservoir in saves_reservoir._saves_by_reservoir.items():
                for s in saves_in_reservoir:
                    reservoir_id_list.append(r_id)
                    counts.append(len(s.visited_patches_x))

            counts = -np.asarray(counts)

        # Normalize by number of saves in each patch.  Note that multiple reservoir_id may map
        # into the same patch.
        patch_id_list = [
            PatchId(reservoir_id.patch_x, reservoir_id.patch_y)
            for reservoir_id in reservoir_id_list
        ]

        if False:
            # Count the number of items in each patch.
            patch_id_to_count = Counter()
            for i, patch_id in enumerate(patch_id_list):
                patch_id_to_count[patch_id] += 1

            # Build normalizing list for weights.
            patch_counts = np.zeros(len(patch_id_list))
            for i, patch_id in enumerate(patch_id_list):
                patch_counts[i] = patch_id_to_count[patch_id]

            # Normalize weight by the number of items in the patch.
            counts /= patch_counts

        # Boltzmann-exploration weighting.
        beta = 1.0
        weights = beta * np.exp(-counts)

        # Normalize to sum to 1.
        weights /= weights.sum()

        # Pick reservoir.
        reservoir_list_indices = np.arange(len(reservoir_id_list))
        chosen_reservoir_index = np.random.choice(reservoir_list_indices, p=weights)
        reservoir_id = reservoir_id_list[chosen_reservoir_index]

        # Pick uniformly among saves in the reservoir.
        sample = random.choice(saves_reservoir._saves_by_reservoir[reservoir_id])

        return sample, zip(patch_id_list, weights)

    # Use Boltzmann Exploration.
    #
    # Focus on the least-visited reservoir buckets.  Don't aggregate into patches.
    if False:
        counts = [
            # saves_reservoir._reservoir_count_since_refresh[saves_reservoir.reservoir_id_from_save(s)]
            saves_reservoir._reservoir_seen_counts[saves_reservoir.reservoir_id_from_save(s)]
            for s in saves
        ]

        # Boltzmann-exploration weighting.
        beta = 1.0
        weights = np.exp(beta * -np.asarray(counts))
        weights /= weights.sum()

        # Pick reservoir.
        save_indicies = np.arange(len(saves))
        chosen_save_index = np.random.choice(save_indicies, p=weights)

        sample = saves[chosen_save_index]

        patch_id_list = [
            saves_reservoir.patch_id_from_save(s)
            for s in saves
        ]

        return sample, zip(patch_id_list, weights)

    # TODO(millman): NOT TESTED
    # Weight across x patch hyperbolically.
    # Weight across y patch exponentially (Boltzmann).
    if False:
        # Collect into xy buckets.
        xy_to_patches = {}
        for s in saves:
            patch_id = saves_reservoir.patch_id_from_save(s)
            xy_to_patches.setdefault(patch_id.patch_x, {}).setdefault(patch_id.patch_y, []).append((s, patch_id))

        rng = np.random.default_rng()

        # Hyperbolic weighting for x.  Normalize by number of items in x dimension.
        x_list = list(xy_to_patches.keys())
        x_weights = _weight_hyperbolic(len(x_list))
        x_counts = np.fromiter((len(ys) for ys in xy_to_patches.values()), dtype=np.int64)
        x_weights /= x_counts
        x_probs = x_weights / x_weights.sum()

        # Pick an x.
        x_id = rng.choice(x_list, p=x_probs)

        y_to_patches = xy_to_patches[x_id]

        # Exponential weighting for y.
        y_list = list(y_to_patches.keys())

        beta = 1.0
        y_counts = np.fromiter((count for _s, count in y_list), dtype=np.int64)
        y_weights = np.exp(beta * -y_counts)
        y_probs = y_weights / y_weights.sum()

        # Pick a y.
        y_id = rng.choice(y_list, p=y_probs)

        sample, sample_patch_id = y_to_patches[y_id]

    # Weight across x patch hyperbolically.
    # Weight across y patch exponentially (Boltzmann).
    if False:
        reservoir_id_list = [
            saves_reservoir.reservoir_id_from_save(s)
            for s in saves
        ]

        patch_id_list = [
            saves_reservoir.patch_id_from_save(s)
            for s in saves
        ]

        patch_id_to_reservoir_ids = {}
        for i,(p_id, r_id) in enumerate(zip(patch_id_list, reservoir_id_list)):
            patch_id_to_reservoir_ids.setdefault(p_id, []).append((r_id, i))

        # Adjust any end-of-level discontinuous values for display.

        # Build a dense grid of the max patch values.
        # Adjust any end-of-level discontinuous values.
        max_px = 0
        max_py = 0
        for p in patch_id_list:
            if p.patch_x <= _MAX_PATCHES_X:
                max_px = max(max_px, p.patch_x)
                max_py = max(max_py, p.patch_y)
            else:
                if False:
                    # Determine the section the patch is in.

                    # Create an offset for the given section.

                    # Adjust to that section.

                    # The section here is the 255-width pixel section that the level is divided into.
                    # Also called "screen" in the memory map.
                    x = p.patch_x * PATCH_SIZE

                    section_x = (x // 256) * 256
                    section_patch_x = section_x // PATCH_SIZE

                    offset_x = x - section_x
                    offset_patch_x = offset_x // PATCH_SIZE

                    # Determine how many full-screen offsets we need from the edge of the histogram.
                    if section_patch_x not in special_section_offsets:
                        special_section_offsets[section_patch_x] = next_special_section_id[0]
                        next_special_section_id[0] += 1

                    section_id = special_section_offsets[section_patch_x]
                    patches_in_section = 256 // PATCH_SIZE

                    # Calculate the starting patch of the section.
                    section_c = hc - (1 + section_id) * patches_in_section

                    # Calculate how much offset we need from the start of the section.
                    # Rewrite the x and y position, for display into this after-level section.
                    c = section_c + offset_patch_x


                    # TODO(millman): IMPLEMENT OFFSETS CORRECTLY
                    # raise AssertionError("IMPLEMENT")
                    pass
        max_px += 1
        max_py += 1

        count_dim_x = np.zeros(max_px)
        count_dim_y = np.zeros(max_py)
        count_grid = np.zeros((max_py, max_px))
        # weight_dim_y = np.zeros(max_py)
        weight_grid = np.zeros((max_py, max_px))

        # Populate the grid with counts.
        for p, r in zip(patch_id_list, reservoir_id_list):
            if p.patch_x >= _MAX_PATCHES_X:
                # TODO(millman): implement
                continue

            count_in_patch = saves_reservoir._reservoir_count_since_refresh[r]

            count_dim_x[p.patch_x] += 1
            count_dim_y[p.patch_y] += 1

            count_grid[p.patch_y, p.patch_x] += 1

            # weight_dim_y[p.patch_y] += count_in_patch
            weight_grid[p.patch_y, p.patch_x] += count_in_patch

        # Normalize x and y dimension by multiplicity (count of non-zero values).

        # Avoid division by zero.
        x_counts_safe = np.where(count_dim_x == 0, 1, count_dim_x)
        y_counts_safe = np.where(count_dim_y == 0, 1, count_dim_y)

        # Broadcast and normalize by counts.
        weight_grid_norm = weight_grid / (x_counts_safe[None, :] * y_counts_safe[:, None])

        rng = np.random.default_rng()

        # Hyperbolic weighting for x.  Normalize by number of items in x dimension.
        # We want the larger x values to be preferred.
        x_weights = _weight_hyperbolic(max_px)

        # Exponential weighting for y.  We want the least visited y values to be preferred.
        beta = 1.0
        y_weights_grid = np.exp(beta * -count_grid)
        # print(f"count_grid SHAPE: {count_grid.shape} {y_weights_grid.shape=}")

        # Weights are independent, so can be combined.
        # print(f"WEIGHT GRID NORM SHAPE: {weight_grid_norm.shape} {x_weights[None,:].shape=} {y_weights_grid[:, None].shape=}")
        weights_grid_result = weight_grid_norm * x_weights[None, :] * y_weights_grid
        # print(f"WEIGHT RESULT SHAPE: {weights_grid_result.shape}")

        # Select a random r,c value out of the grid.

        # Flatten the grid to 1D.
        flat_weights = weights_grid_result.ravel()

        # Normalize weights to sum to 1.
        probs = flat_weights / flat_weights.sum()

        # Select a single random index based on the weights.
        chosen_flat_idx = np.random.choice(flat_weights.size, p=probs)

        # Convert back to 2D indices.
        # print(f"CHOSEN FLAT INDEX: {chosen_flat_idx} weight_grid_result={weights_grid_result.shape}")
        chosen_y, chosen_x = np.unravel_index(chosen_flat_idx, weights_grid_result.shape)

        # Reconstruct the patch_id to find all saves in the patch.
        selected_patch_id = PatchId(chosen_x, chosen_y)
        reservoir_and_i_pairs = patch_id_to_reservoir_ids[selected_patch_id]

        # Choose one of the reservoir ids, uniform sampling.
        selected_reservoir_id, selected_i = random.choice(reservoir_and_i_pairs)

        sample = saves[selected_i]

        return sample, weights_grid_result

    return sample, None


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


def _get_min_speed() -> int:
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
    # Since units/tick is distance/time, that's also "speed".
    #
    # Here are some sample numbers:
    #   3266 units / 401 ticks used ->  8.1 units/tick (min)
    #   3266 units / 200 ticks used -> 16.3 units/tick (twice normal rate, good)
    #    100 units /  20 ticks used ->  5.0 units/tick (bad, too slow)
    #    100 units / 100 ticks used ->  1.0 units/tick (bad, too slow)
    #
    # Level 1-3 is 3100 units, 300 ticks available ->  10.1 units/tick (min)
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

    return min_speed


def _get_min_patches_per_tick() -> int:
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

    return min_patches_per_tick


# Histogram visualization

# Approximate the size of the histogram based on how many patches we need.
_MAX_LEVEL_DIST = 6400
_MAX_PATCHES_X = int(np.ceil(_MAX_LEVEL_DIST / PATCH_SIZE))
_MAX_EXTRA_PATCHES_X = 5
_MAX_PATCHES_Y = int(np.ceil(240 / PATCH_SIZE))
_NUM_MAX_PATCHES = _MAX_PATCHES_X * _MAX_PATCHES_Y
_SPACE_R = 0

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


def _build_patch_histogram_rgb(
    patch_id_and_count_pairs: list[PatchId, int],
    current_patch: PatchId,
    hist_rows: int,
    hist_cols: int,
    pixel_size: int,
) -> NdArrayRGB8:
    hr, hc = hist_rows, hist_cols

    patch_histogram = np.zeros((hr + 1, hc + 1), dtype=np.float64)

    if False:
        patch_id_and_count_pairs = list(patch_id_and_count_pairs)
        print(f"FIRST patch_id_and_count_pairs: {patch_id_and_count_pairs[0]}")
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
        x = check_patch_x.patch_x * PATCH_SIZE

        section_x = (x // 256) * 256
        section_patch_x = section_x // PATCH_SIZE

        offset_x = x - section_x
        offset_patch_x = offset_x // PATCH_SIZE

        # Determine how many full-screen offsets we need from the edge of the histogram.
        if section_patch_x not in special_section_offsets:
            special_section_offsets[section_patch_x] = next_special_section_id[0]
            next_special_section_id[0] += 1

        section_id = special_section_offsets[section_patch_x]
        patches_in_section = 256 // PATCH_SIZE

        # Calculate the starting patch of the section.
        section_c = hc - (1 + section_id) * patches_in_section

        # Calculate how much offset we need from the start of the section.
        # Rewrite the x and y position, for display into this after-level section.
        c = section_c + offset_patch_x

        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc} -> c={c}")

        return c


    for save_patch_id, count in patch_id_and_count_pairs:
        patch_x, patch_y = save_patch_id.patch_x, save_patch_id.patch_y

        # For display purposes only, if we're at one of the special offscreen locations,
        # Use an offset, but still show it in the histogram.
        if patch_x <= _MAX_PATCHES_X:
            # What row of the level we're in.  Wrap around if past the end of the screen.
            wrap_i = patch_x // hc

            r = wrap_i * (_MAX_PATCHES_Y + _SPACE_R) + patch_y
            c = patch_x % hc
        else:
            # Special case, we're past the end of the level for some special section.
            r = hr - _MAX_PATCHES_Y + patch_y
            c = _calc_c_for_special_section(save_patch_id, hr=hr, hc=hc)

        try:
            patch_histogram[r][c] = count
        except IndexError:
            print(f"PATCH LAYOUT: max_patches_x={_MAX_PATCHES_X} max_patches_y={_MAX_PATCHES_Y} pixel_size={pixel_size} hr={hr} hc={hc}")
            print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")

            patch_histogram[hr][hc] += 1

    #print(f"HISTOGRAM min={patch_histogram.min()} max={patch_histogram.max()}")

    # Normalize counts to range (0, 255)
    grid_f = patch_histogram
    grid_g = (grid_f / grid_f.max() * 255).astype(np.uint8)
    grid_rgb = np.stack([grid_g]*3, axis=-1)

    # Mark current patch.
    if current_patch.patch_x <= _MAX_PATCHES_X:
        px, py = current_patch.patch_x, current_patch.patch_y
        wrap_i = px // hc
        patch_r = wrap_i * (_MAX_PATCHES_Y + _SPACE_R) + py
        patch_c = px % hc
    else:
        # Special case, we're past the end of the level for some special section.
        patch_r = hr - _MAX_PATCHES_Y + current_patch.patch_y
        patch_c = _calc_c_for_special_section(current_patch, hr=hr, hc=hc)

    try:
        grid_rgb[patch_r][patch_c] = (0, 255, 0)
    except IndexError:
        print(f"PATCH LAYOUT: max_patches_x={_MAX_PATCHES_X} max_patches_y={_MAX_PATCHES_Y} pixel_size={pixel_size} hr={hr} hc={hc}")
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
    visited_patches: set[Any],
    visited_patches_x: set[Any],
    distance_x: int,
    min_speed: float,
    step: int,
    steps_since_load: int,
):
    steps_per_sec = step / dt

    # speed = distance_x / ticks_used
    # patches_per_tick = len(visited_patches) / ticks_used
    # patches_x_per_tick = len(visited_patches_x) / ticks_used

    # screen_x = ram[0x006D]
    # level_screen_x = ram[0x071A]
    # screen_pos = ram[0x0086]

    print(
        f"{_seconds_to_hms(dt)} "
        f"level={_str_level(world, level)} "
        f"x={x} y={y} ticks_left={ticks_left} "
        f"ticks_used={ticks_used} "
        f"states={len(saves)} "
        # f"visited={len(visited_patches)} "
        # f"visited_x={len(visited_patches_x)} "
        f"steps/sec={steps_per_sec:.4f} "
        # f"speed={speed:.2f} "
        # f"(required={min_speed:.2f}) "
        # f"patches/tick={patches_per_tick:.2f} "
        # f"patches_x/tick={patches_x_per_tick:.2f} "
        f"steps_since_load={steps_since_load}")


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
        [make_env(args.env_id, 0, args.capture_video, run_name, args.headless, (args.start_world, args.start_level))],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
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

    min_speed = _get_min_speed()
    min_patches_per_tick = _get_min_patches_per_tick()

    # Per-trajectory state.  Resets after every death/level.
    action_history = []
    visited_patches = set()
    visited_patches_x = set()
    controller = _to_controller_presses([])

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
    distance_x = 0

    patch_id = PatchId(x // PATCH_SIZE, y // PATCH_SIZE)
    prev_patch_id = patch_id

    saves = PatchReservoir()
    force_terminate = False
    steps_since_load = 0

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
        controller = _flip_buttons(controller, flip_prob=0.025, ignore_button_mask=_MASK_START_AND_SELECT)

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

        patch_id = PatchId(x // PATCH_SIZE, y // PATCH_SIZE)

        # Calculate derived states.
        ticks_used = max(1, level_ticks - ticks_left)

        speed = distance_x / ticks_used
        patches_per_tick = len(visited_patches) / ticks_used

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
        else:
            distance_x += x - prev_x

        if False: # x > 65000:
            # 0x006D  Current screen (in which the player is currently in, increase or decrease depending on player's position)
            # 0x0086  Player x position on screen
            # 0x071A  Current screen (in level, always increasing)
            # 0x071C  ScreenEdge X-Position
            screen_x = ram[0x006D]
            level_screen_x = ram[0x071A]
            screen_pos = ram[0x0086]
            print(f"WEIRD X POS: level={_str_level(world, level)} screen[0x006D]={screen_x} level_screen[0x071A]={level_screen_x} screen_pos[0x0086]={screen_pos} x={x} y={y} lives={lives}")

        # Mark current patch as visted.
        reservoir_id = saves.reservoir_id_from_state(world=world, level=level, patch_id=patch_id, prev_patch_id=prev_patch_id)
        if reservoir_id in saves._reservoir_refreshed:
            saves._reservoir_refreshed.remove(reservoir_id)

        # TODO(millman): something is broken with the termination flag?
        if lives < prev_lives and not termination:
            print(f"Lost a life: x={x} ticks_left={ticks_left} distance={distance_x}")
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
            save_info, _sample_weights = _choose_save(saves)

            # Reload and re-initialize.
            nes.load(save_info.save_state)
            ram = nes.ram()
            controller[:] = nes.controller1.is_pressed[:]

            # Flip the buttons with some probability.  If we're loading a state, we don't want to
            # be required to use the same action state that was tried before.  To get faster coverage
            # We flip buttons here with much higher probability than during a trajectory.
            controller = _flip_buttons(controller, flip_prob=0.3, ignore_button_mask=_MASK_START_AND_SELECT)

            visited_patches = save_info.visited_patches.copy()
            visited_patches_x = save_info.visited_patches_x.copy()
            action_history = save_info.action_history.copy()

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
            patch_id = PatchId(x // PATCH_SIZE, y // PATCH_SIZE)
            prev_patch_id = save_info.prev_patch_id

            level_ticks = save_info.level_ticks

            # Update derived state.
            ticks_used = max(1, level_ticks - ticks_left)
            distance_x = save_info.distance_x

            speed = distance_x / ticks_used
            patches_per_tick = len(visited_patches) / ticks_used

            if True:
                saves_list = saves.values()
                save_i = saves_list.index(save_info)
                weight = save_i

                # Hyperbolic weight.
                N = len(saves_list)
                weights = _weight_hyperbolic(N)
                weights /= weights[0]

                weight = weights[save_i]

                print(f"Loaded save: [{save_i}] {weight:.4f}x: save_id={save_info.save_id} level={_str_level(world, level)}, x={x} y={y} lives={lives}")

                if False:
                    _print_saves_list(saves.values())

            steps_since_load = 0
            force_terminate = False

        # Stop after some fixed number of steps.  This will force the sampling logic to run more often,
        # which means we won't waste as much time running through old states.
        elif steps_since_load >= PATCH_SIZE * 3:
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
                    _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, visited_patches=visited_patches, visited_patches_x=visited_patches_x, distance_x=distance_x, min_speed=min_speed, step=step, steps_since_load=steps_since_load)

                # Set number of ticks in level to the current ticks.
                level_ticks = get_time_left(ram)

                # Clear state.
                action_history = []
                visited_patches = set()
                visited_patches_x = set()

                distance_x = 0

                print(f"Starting level: {_str_level(world, level)}")

                # Update derived state.
                ticks_used = max(1, level_ticks - ticks_left)

                # Print after-level-start info.
                if True:
                    _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, visited_patches=visited_patches, visited_patches_x=visited_patches_x, distance_x=distance_x, min_speed=min_speed, step=step, steps_since_load=steps_since_load)

                assert lives > 1 and lives < 100, f"How did we end up with lives?: {lives}"

                saves = PatchReservoir()
                saves.add(SaveInfo(
                    save_id=next_save_id,
                    x=x,
                    y=y,
                    level=level,
                    world=world,
                    level_ticks=level_ticks,
                    distance_x=distance_x,
                    ticks_left=ticks_left,
                    save_state=nes.save(),
                    visited_patches=visited_patches.copy(),
                    visited_patches_x=visited_patches_x.copy(),
                    action_history=action_history.copy(),
                    prev_patch_id=prev_patch_id,
                ))
                next_save_id += 1

            # If time left is too short, this creates a bad feedback loop because we can keep
            # dying due to timer.  That would encourage the agent to finish quick, but it may not
            # be possible to actually finish.
            #
            # Wait until we've used some ticks, so that the speed is meaningful.
            if False:  # ticks_used > 20 and speed < min_speed:
                print(f"Ending trajectory, traversal is too slow: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
                force_terminate = True

            # We want to ensure Mario is finding new states at a reasonable rate, otherwise it means that we're
            # going back over too much ground we've seen before.
            elif False:  # ticks_used > 20 and patches_per_tick < min_patches_per_tick:
                print(f"Ending trajectory, patch discovery rate is too slow: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
                force_terminate = True

            elif patch_id != prev_patch_id:
                valid_lives = lives > 1 and lives < 100
                valid_x = True  # x < 65500

                # NOTE: Some levels (like 4-4) are discontinuous.  We can get x values of > 65500.
                if not valid_x:
                    # TODO(millman): how did we get into a weird x state?  Happens on 4-4.
                    print(f"RAM values: ram[0x006D]={ram[0x006D]=} * 256 + ram[0x0086]={ram[0x0086]=}")
                    print(f"Something is wrong with the x position, don't save this state: level={_str_level(world, level)} x={x} y={y} lives={lives} ticks-left={ticks_left} states={len(saves)} visited={len(visited_patches)}")
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
                        print(f"Ending trajectory, revisited state: actions {len(action_history)} > {len(max_item.action_history)}: x={x} ticks_left={ticks_left} distance={distance_x} speed={speed:.2f} patches/tick={patches_per_tick:.2f}")
                        force_terminate = True

                else:
                    saves.add(SaveInfo(
                        save_id=next_save_id,
                        x=x,
                        y=y,
                        level=level,
                        world=world,
                        level_ticks=level_ticks,
                        distance_x=distance_x,
                        ticks_left=ticks_left,
                        save_state=nes.save(),
                        visited_patches=visited_patches.copy(),
                        visited_patches_x=visited_patches_x.copy(),
                        action_history=action_history.copy(),
                        prev_patch_id=prev_patch_id,
                    ))
                    next_save_id += 1
                    visited_patches.add(patch_id)

                    patch_x_id = (world, level, x // PATCH_SIZE)
                    visited_patches_x.add(patch_x_id)

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
            _print_info(dt=now-start_time, world=world, level=level, x=x, y=y, ticks_left=ticks_left, ticks_used=ticks_used, saves=saves, visited_patches=visited_patches, visited_patches_x=visited_patches_x, distance_x=distance_x, min_speed=min_speed, step=step, steps_since_load=steps_since_load)
            last_print_time = now

        # Visualize the distribution of save states.
        if args.vis_freq_sec > 0 and now - last_vis_time > args.vis_freq_sec:

            # Histogram of number of saves in each patch reservoir.
            # Collect reservoirs into patches.
            patch_id_to_count = Counter()
            for reservoir_id, saves_in_reservoir in saves._saves_by_reservoir.items():
                patch_id = PatchId(reservoir_id.patch_x, reservoir_id.patch_y)
                patch_id_to_count[patch_id] += len(saves_in_reservoir)

            patch_id_and_count_pairs = patch_id_to_count.items()

            img_rgb_240 = _build_patch_histogram_rgb(
                patch_id_and_count_pairs,
                current_patch=patch_id,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=1)

            # Histogram of seen counts.
            patch_id_and_count_pairs = saves._patch_seen_counts.items()

            img_rgb_240 = _build_patch_histogram_rgb(
                patch_id_and_count_pairs,
                current_patch=patch_id,
                hist_rows=_HIST_ROWS,
                hist_cols=_HIST_COLS,
                pixel_size=_HIST_PIXEL_SIZE,
            )
            screen.blit_image(img_rgb_240, screen_index=3)

            # Histogram of sampling weight.
            if False:
                reservoir_id_list = saves._reservoir_count_since_refresh.keys()
                counts = np.fromiter(saves._reservoir_count_since_refresh.values(), dtype=np.int64)

                # Boltzmann-exploration weighting.
                beta = 1.0
                weights = beta * np.exp(-counts)
                weights /= weights.sum()

                patch_id_to_count = Counter()
                patch_id_to_weight = Counter()
                for i, reservoir_id in enumerate(reservoir_id_list):
                    p_id = PatchId(reservoir_id.patch_x, reservoir_id.patch_y)
                    patch_id_to_weight[p_id] += weights[i]
                    patch_id_to_count[p_id] += 1

                # Normalize weight by the number of items in the patch.
                for p_id, count in patch_id_to_count.items():
                    patch_id_to_weight[p_id] /= count

                patch_id_and_weight_pairs = patch_id_to_weight.items()

                img_rgb_240 = _build_patch_histogram_rgb(
                    patch_id_and_weight_pairs,
                    current_patch=patch_id,
                    hist_rows=_HIST_ROWS,
                    hist_cols=_HIST_COLS,
                    pixel_size=_HIST_PIXEL_SIZE,
                )
                screen.blit_image(img_rgb_240, screen_index=2)

            if True:
                _selected, patch_id_and_weight_pairs = _choose_save(saves)

                img_rgb_240 = _build_patch_histogram_rgb(
                    patch_id_and_weight_pairs,
                    current_patch=patch_id,
                    hist_rows=_HIST_ROWS,
                    hist_cols=_HIST_COLS,
                    pixel_size=_HIST_PIXEL_SIZE,
                )
                screen.blit_image(img_rgb_240, screen_index=2)

            if False:
                _selected, sample_weights_grid = _choose_save(saves)

                # Convert grid into a list for the histogram.
                patch_id_and_weight_pairs = []
                for r, rows in enumerate(sample_weights_grid):
                    for c, weight in enumerate(rows):
                        p_id = PatchId(c, r)
                        patch_id_and_weight_pairs.append((p_id, weight))

                img_rgb_240 = _build_patch_histogram_rgb(
                    patch_id_and_weight_pairs,
                    current_patch=patch_id,
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
