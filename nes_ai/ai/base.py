from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from pydantic import BaseModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributions.categorical import Categorical
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets, transforms

from nes_ai.ai.rollout_data import RolloutData

# avail_pretrained_models = timm.list_models(pretrained=True)
# print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 32
REWARD_VECTOR_SIZE = 9


class RewardIndex(IntEnum):
    SCORE = 0
    TIME_LEFT = 1
    COINS = 2
    LIVES = 3
    WORLD = 4
    LEVEL = 5
    LEFT_POS = 6
    TOP_POS = 7
    POWERUP_LEVEL = 8


class RewardMap(BaseModel):
    score: int
    time_left: int
    coins: int
    lives: int
    world: int
    level: int
    left_pos: int
    top_pos: int
    powerup_level: int

    @staticmethod
    def combine_reward_vector_single(reward_vector) -> float:
        expected_shape = (REWARD_VECTOR_SIZE,)
        assert reward_vector.shape == expected_shape, f"Unexpected reward_vector.shape: {reward_vector.shape} != {expected_shape}"
        retval = (
            (1 * reward_vector[RewardIndex.SCORE])
            + (0 * reward_vector[RewardIndex.TIME_LEFT])
            + (1 * reward_vector[RewardIndex.COINS])

            # NOTE: This will give a big reward for gaining a life, and a big negative reward for
            #   losing a life.  The reward_vector is the delta, so when a life is lost, the
            #   reward_vector will be -1.
            + (10 * reward_vector[RewardIndex.LIVES])

            + (100 * reward_vector[RewardIndex.WORLD])
            + (100 * reward_vector[RewardIndex.LEVEL])
            + (
                0.01 * reward_vector[RewardIndex.LEFT_POS]
                # if reward_vector[RewardIndex.LEFT_POS] > 0
                # else -0.1 * reward_vector[RewardIndex.LEFT_POS]
            )
            + (0 * reward_vector[RewardIndex.TOP_POS])
            + (3 * reward_vector[RewardIndex.POWERUP_LEVEL])
            - 0.01  # penalty for spending time.  Use this instead of time_left
        ).item()

        if type(retval) is not float:
            print(f"Unexpected reward retval: {type(retval)} != float, {retval=}")

        assert type(retval) is float, f"Unexpected reward retval: {type(retval)} != float"
        return retval

    @staticmethod
    def combine_reward_vector(reward_vector):
        expected_shape = (REWARD_VECTOR_SIZE, 1)
        assert reward_vector.shape == expected_shape, f"Unexpected reward_vector.shape: {reward_vector.shape} != {expected_shape}"
        retval = (
            (100 * reward_vector[:, RewardIndex.SCORE])
            + (1 * reward_vector[:, RewardIndex.TIME_LEFT])
            + (100 * reward_vector[:, RewardIndex.COINS])
            + (0 * reward_vector[:, RewardIndex.LIVES])
            + (10000 * reward_vector[:, RewardIndex.WORLD])
            + (10000 * reward_vector[:, RewardIndex.LEVEL])
            + (1 * reward_vector[:, RewardIndex.LEFT_POS])
            + (0 * reward_vector[:, RewardIndex.TOP_POS])
            + (10000 * reward_vector[:, RewardIndex.POWERUP_LEVEL])
        )
        assert retval.shape == (reward_vector.shape[0],)
        return retval

    @staticmethod
    def reward_vector(last_reward_map: RewardMap | None, reward_map: RewardMap):
        retval = torch.zeros((REWARD_VECTOR_SIZE,), dtype=torch.float)
        if last_reward_map is not None:
            retval[RewardIndex.SCORE] = reward_map.score - last_reward_map.score
            retval[RewardIndex.TIME_LEFT] = (
                reward_map.time_left - last_reward_map.time_left
            )
            retval[RewardIndex.COINS] = reward_map.coins - last_reward_map.coins
            retval[RewardIndex.LIVES] = reward_map.lives - last_reward_map.lives
            retval[RewardIndex.WORLD] = reward_map.world - last_reward_map.world
            retval[RewardIndex.LEVEL] = reward_map.level - last_reward_map.level
            retval[RewardIndex.LEFT_POS] = (
                reward_map.left_pos - last_reward_map.left_pos
            )
            if retval[RewardIndex.LEFT_POS] < -10:
                retval[RewardIndex.LEFT_POS] = 0  # teleported
            retval[RewardIndex.TOP_POS] = 0
            retval[RewardIndex.POWERUP_LEVEL] = (
                reward_map.powerup_level - last_reward_map.powerup_level
            )

        return retval


def _linear_block(in_features, out_features):
    return [
        torch.nn.Linear(in_features, out_features),
        torch.nn.LeakyReLU(),
        # torch.nn.BatchNorm1d(out_features),
    ]


A = 0
B = 1
SELECT = 2
START = 3
UP = 4
DOWN = 5
LEFT = 6
RIGHT = 7

# laksjdlaskjdsalkj

# IMAGE_MODEL_NAME = "levit_128s.fb_dist_in1k"
# IMAGE_MODEL_NAME = "levit_256.fb_dist_in1k"
# IMAGE_MODEL_NAME = "vit_tiny_patch16_224"
# IMAGE_MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
IMAGE_MODEL_NAME = "vit_small_patch32_224.augreg_in21k_ft_in1k"
# IMAGE_MODEL_NAME = "efficientvit_m5.r224_in1k"


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print(timm.list_models(pretrained=True))
        self.trunk = timm.create_model(IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        # data_config = timm.data.resolve_data_config(model=self.trunk, verbose=True)
        # self.trunk_transforms = timm.data.create_transform(
        #     **data_config,
        #     is_training=True,
        # )
        # print(self.trunk_transforms)
        # for x in self.trunk_transforms.transforms:
        #     print(x)
        # for x in self.trunk_transforms:
        #     print(x)
        self.head = torch.nn.Sequential(
            *_linear_block((self.trunk.num_features * 4) + (8 * 3), 1024),
            *_linear_block(1024, 1024),
            torch.nn.Linear(1024, self.num_actions),
        )

    @property
    def num_actions(self):
        return 3 * 3 * 2 * 2

    @staticmethod
    def input_array_to_index(input_array):
        assert input_array.dim() == 2, f"{input_array.shape}"

        input_index = torch.zeros(
            (input_array.shape[0],), dtype=torch.int, device=input_array.device
        )

        input_index += input_array[:, LEFT] * 1
        input_index += input_array[:, RIGHT] * 2

        input_index *= 3

        input_index += input_array[:, UP] * 1
        input_index += input_array[:, DOWN] * 2

        input_index <<= 1
        input_index += input_array[:, A]

        input_index <<= 1
        input_index += input_array[:, B]

        # print("input_index", input_index)
        return input_index

    def convert_input_array_to_index(self, input_array):
        input_index = Actor.input_array_to_index(input_array)

        assert torch.all(
            input_index < self.num_actions
        ), f"{input_index} >= {self.num_actions} from {input_array}"
        return input_index

    def convert_index_to_input_array(self, input_index):
        assert input_index.dim() == 1, f"{input_index.shape}"

        assert torch.all(input_index < self.num_actions)

        input_array = torch.zeros(
            (input_index.shape[0], 8), dtype=torch.int, device=input_index.device
        )
        input_array[:, B] = input_index & 1
        input_index >>= 1

        input_array[:, A] = input_index & 1
        input_index >>= 1

        input_array[:, DOWN] = input_index % 3 == 2
        input_array[:, UP] = input_index % 3 == 1
        input_index //= 3

        input_array[:, RIGHT] = input_index % 3 == 2
        input_array[:, LEFT] = input_index % 3 == 1

        assert torch.all(input_index < 3), f"{input_index}"
        return input_array

    def forward(self, images, past_inputs):
        batch_size = images.shape[0]
        assert images.shape[1:] == (4, 3, 224, 224), f"{images.shape}"

        # trunk_output = torch.cat(
        #     (
        #         *[
        #             self.trunk(self.trunk_transforms(images[:, x, :, :, :]))
        #             for x in range(4)
        #         ],
        #         past_inputs.reshape(-1, 3 * 8),
        #     ),
        #     dim=1,
        # )
        trunk_output = torch.cat(
            (
                *[self.trunk(images[:, x, :, :, :]) for x in range(4)],
                past_inputs.reshape(-1, 3 * 8),
            ),
            dim=1,
        ).detach()

        images = images.reshape(-1, 3, 224, 224)
        assert past_inputs.shape[1:] == (3, 8), f"{past_inputs.shape}"
        outputs = self.head(trunk_output)
        return outputs

    def get_action(self, images, past_inputs, action_taken=None):
        logits = self.forward(images, past_inputs)
        action_probs = Categorical(logits=logits)
        # print("ACTION PROBS", action_probs.logits)
        if action_taken is None:
            action = action_probs.sample()
            action_array = self.convert_index_to_input_array(action)
        else:
            action = self.convert_input_array_to_index(action_taken)
            action_array = action_taken
            action_array2 = self.convert_index_to_input_array(action)
            assert torch.equal(
                action_array, action_array2
            ), f"{action_array} != {action_array2}"
        # print("ACTION INFO", action_array, action)
        return action_array, action_probs.log_prob(action), action_probs.entropy()


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.trunk = trunk
        # self.trunk_transforms = trunk_transforms
        self.trunk = timm.create_model(IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_data_config(model=self.trunk, verbose=True)
        self.trunk_transforms = timm.data.create_transform(
            **data_config,
            is_training=True,
        )
        self.head = torch.nn.Sequential(
            *_linear_block(
                (self.trunk.num_features * 4) + ((8 + REWARD_VECTOR_SIZE) * 3), 1024
            ),
            *_linear_block(1024, 1024),
            torch.nn.Linear(1024, REWARD_VECTOR_SIZE),
        )

    def forward(self, images, past_inputs, past_rewards):
        assert images.shape[1:] == (4, 3, 224, 224), f"{images.shape}"
        assert past_inputs.shape[1:] == (3, 8), f"{past_inputs.shape}"
        assert past_rewards.shape[1:] == (
            3,
            REWARD_VECTOR_SIZE,
        ), f"{past_rewards.shape}"
        trunk_output = torch.cat(
            (
                *[self.trunk(images[:, x, :, :, :]) for x in range(4)],
                past_inputs.reshape(-1, 3 * 8),
                past_rewards.reshape(-1, 3 * REWARD_VECTOR_SIZE),
            ),
            dim=1,
        ).detach()
        # trunk_output = torch.cat(
        #     (
        #         *[
        #             self.trunk(self.trunk_transforms(images[:, x, :, :, :]))
        #             for x in range(4)
        #         ],
        #         past_inputs.reshape(-1, 3 * 8),
        #         past_rewards.reshape(-1, 3 * REWARD_VECTOR_SIZE),
        #     ),
        #     dim=1,
        # )
        outputs = self.head(trunk_output)
        return outputs


def bcdToInt(bcd_bytes):
    value = 0
    for x in range(0, len(bcd_bytes)):
        value *= 10
        value += int(bcd_bytes[x])
    return value


def get_time_left(ram) -> int:
    time_left_bytes = ram[0x07F8:0x07FB]
    time_left = bcdToInt(time_left_bytes)
    return time_left


def get_level(ram) -> int:
    return ram[0x760]


def get_world(ram) -> int:
    return ram[0x75F]


def compute_reward_map(last_reward_map: RewardMap | None, ram):

    # From: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map

    high_score_bytes = ram[0x07DD:0x07E3]
    score = bcdToInt(high_score_bytes) * 10

    time_left_bytes = ram[0x07F8:0x07FB]
    time_left = bcdToInt(time_left_bytes)

    # For now, consider us at the next level if we are sliding down the flagpole
    sliding_down_flagpole = ram[0x001D] == 0x03

    coins = ram[0x75E]

    world = ram[0x75F]
    level = ram[0x760] + sliding_down_flagpole

    powerup_level = ram[0x756]
    left_pos = int(ram[0x006D]) * 256 + int(ram[0x0086])

    #  0 = above viewport
    #  1 = within viewport
    #  >1 = below viewport
    vertical_screen_pos = ram[0x00B5]

    # 0=top of screen, 255=bottom of screen
    screen_pos_y = ram[0x00CE]

    top_pos = vertical_screen_pos * screen_pos_y

    lives = ram[0x75A]

    reward_map = RewardMap(
        score=score,
        time_left=time_left,
        coins=coins,
        lives=lives,
        world=world,
        level=level,
        left_pos=left_pos,
        top_pos=top_pos,
        powerup_level=powerup_level,
    )

    return reward_map, RewardMap.reward_vector(last_reward_map, reward_map)
