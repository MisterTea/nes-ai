from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
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

# avail_pretrained_models = timm.list_models(pretrained=True)
# print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 32
REWARD_VECTOR_SIZE = 8


class RewardMap(BaseModel):
    score: int
    coins: int
    lives: int
    world: int
    level: int
    left_pos: int
    powerup_level: int
    player_is_dying: bool

    @staticmethod
    def reward_vector(last_reward_map: RewardMap | None, reward_map: RewardMap):
        retval = torch.zeros((REWARD_VECTOR_SIZE,), dtype=torch.float)
        if last_reward_map is not None:
            retval[0] = reward_map.score - last_reward_map.score
            retval[1] = reward_map.coins - last_reward_map.coins
            retval[2] = reward_map.lives - last_reward_map.lives
            retval[3] = reward_map.world - last_reward_map.world
            retval[4] = reward_map.level - last_reward_map.level
            retval[5] = reward_map.left_pos - last_reward_map.left_pos
            retval[6] = reward_map.powerup_level - last_reward_map.powerup_level
            retval[7] = reward_map.player_is_dying - last_reward_map.player_is_dying

        return retval

    @staticmethod
    def compute_step_reward(last_reward_map: RewardMap | None, reward_map: RewardMap):
        if last_reward_map is None:
            return 0
        reward = 0
        reward += 100 * (reward_map.score - last_reward_map.score)
        reward += 100 * (reward_map.coins - last_reward_map.coins)
        reward += 10000 * (reward_map.lives - last_reward_map.lives)
        reward += 10000 * (reward_map.world - last_reward_map.world)
        reward += 10000 * (reward_map.level - last_reward_map.level)
        reward += 1 * (reward_map.left_pos - last_reward_map.left_pos)
        reward += 10000 * (reward_map.powerup_level - last_reward_map.powerup_level)
        reward += -10000 * (
            reward_map.player_is_dying - last_reward_map.player_is_dying
        )

        return reward


def _linear_block(in_features, out_features):
    return [
        torch.nn.Linear(in_features, out_features),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(out_features),
    ]


A = 0
B = 1
SELECT = 2
START = 3
UP = 4
DOWN = 5
LEFT = 6
RIGHT = 7


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )
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
        input_index = 0

        if input_array[LEFT]:
            input_index += 1
        elif input_array[RIGHT]:
            input_index += 2

        input_index *= 3
        if input_array[UP]:
            input_index += 1
        elif input_array[DOWN]:
            input_index += 2

        input_index <<= 1
        input_index += input_array[A]

        input_index <<= 1
        input_index += input_array[B]

        return input_index

    def convert_input_array_to_index(self, input_array):
        input_index = Actor.input_array_to_index(input_array)

        assert (
            input_index < self.num_actions
        ), f"{input_index} >= {self.num_actions} from {input_array}"
        return input_index

    def convert_index_to_input_array(self, input_index):
        original_input_index = input_index
        assert input_index < self.num_actions
        input_array = torch.zeros((8,), dtype=torch.int)
        input_array[B] = input_index & 1
        input_index >>= 1

        input_array[A] = input_index & 1
        input_index >>= 1

        input_array[DOWN] = input_index % 3 == 2
        input_array[UP] = input_index % 3 == 1
        input_index //= 3

        input_array[RIGHT] = input_index % 3 == 2
        input_array[LEFT] = input_index % 3 == 1

        assert input_index < 3, f"{original_input_index} -> {input_array}"
        return input_array

    def forward(self, images, past_inputs):
        assert images.shape[1:] == (4, 3, 224, 224), f"{images.shape}"
        assert past_inputs.shape[1:] == (3, 8), f"{past_inputs.shape}"
        trunk_output = torch.cat(
            [self.trunk(images[:, x, :, :, :]) for x in range(4)], dim=1
        )
        trunk_output = torch.cat((trunk_output, past_inputs.reshape(-1, 3 * 8)), dim=1)
        outputs = self.head(trunk_output)
        return outputs


class Critic(torch.nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
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
            [self.trunk(images[:, x, :, :, :]) for x in range(4)], dim=1
        )
        trunk_output = torch.cat(
            (
                trunk_output,
                past_inputs.reshape(-1, 3 * 8),
                past_rewards.reshape(-1, 3 * REWARD_VECTOR_SIZE),
            ),
            dim=1,
        )
        outputs = self.head(trunk_output)
        return outputs
