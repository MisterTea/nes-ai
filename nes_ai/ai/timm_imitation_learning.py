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

from nes_ai.ai.helpers import upscale_and_get_labels
from nes_ai.ai.nes_dataset import NESDataset

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

    def convert_input_array_to_index(self, input_array):
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


class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic(self.actor.trunk)
        self.actor_loss_fn = torch.nn.CrossEntropyLoss()
        self.reward_loss_fn = torch.nn.SmoothL1Loss()
        self.automatic_optimization = False

    def forward(self, images, past_inputs, past_rewards):
        return self.actor.forward(images, past_inputs), self.critic.forward(
            images, past_inputs, past_rewards
        )

    def training_step(self, batch):
        actor_opt, critic_opt = self.optimizers()

        images, reward_vectors, past_inputs, past_rewards, targets = batch

        actor_outputs = self.actor.forward(images, past_inputs)

        target_indices = torch.zeros(
            (images.shape[0],), dtype=torch.long, device=images.device
        )
        for i in range(images.shape[0]):
            target_indices[i] = self.actor.convert_input_array_to_index(targets[i])

        actor_loss = self.actor_loss_fn(actor_outputs, target_indices)
        self.log("actor_train_loss", actor_loss, prog_bar=True)

        actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        actor_opt.step()

        reward_map_outputs = self.critic.forward(images, past_inputs, past_rewards)

        assert (
            reward_map_outputs.shape == reward_vectors.shape
        ), f"{reward_map_outputs.shape} != {reward_vectors.shape}"
        reward_loss = 0.1 * self.reward_loss_fn(reward_map_outputs, reward_vectors)
        self.log("reward_train_loss", reward_loss, prog_bar=True)

        critic_opt.zero_grad()
        self.manual_backward(reward_loss)
        critic_opt.step()

        self.log("train_loss", actor_loss + reward_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):

        # print(batch)
        images, reward_vectors, past_inputs, past_rewards, targets = batch

        actor_outputs = self.actor.forward(images, past_inputs)

        target_indices = torch.zeros(
            (images.shape[0],), dtype=torch.long, device=images.device
        )
        for i in range(images.shape[0]):
            target_indices[i] = self.actor.convert_input_array_to_index(targets[i])

        actor_loss = self.actor_loss_fn(actor_outputs, target_indices)
        self.log("actor_val_loss", actor_loss, prog_bar=True)

        metric = MulticlassAccuracy()
        metric.update(actor_outputs, target_indices)
        accuracy = metric.compute()
        self.log("val_acc", accuracy, prog_bar=True)

        reward_map_outputs = self.critic.forward(images, past_inputs, past_rewards)

        assert (
            reward_map_outputs.shape == reward_vectors.shape
        ), f"{reward_map_outputs.shape} != {reward_vectors.shape}"
        reward_loss = 0.1 * self.reward_loss_fn(reward_map_outputs, reward_vectors)
        self.log("reward_val_loss", reward_loss, prog_bar=True)

        self.log("val_loss", actor_loss + reward_loss, prog_bar=False)

        return actor_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.00001 * BATCH_SIZE)
        actor_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2
                ),
                "monitor": "actor_train_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.00001 * BATCH_SIZE)
        critic_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2
                ),
                "monitor": "critic_train_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return actor_opt, critic_opt


DATA_PATH = Path("./")


class ClassificationData(pl.LightningDataModule):

    def train_dataloader(self):
        train_dataset = NESDataset(DATA_PATH, train=True)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
            num_workers=10,  # shuffle=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                train_dataset.example_weights, BATCH_SIZE * 100, replacement=True
            ),
        )

    def val_dataloader(self):
        val_dataset = NESDataset(DATA_PATH, train=False)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
            num_workers=10,
            # sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        )


hashes = {}


def find_image_input(image, data_frame):
    import imagehash
    from PIL import Image

    global hashes
    if len(hashes) == 0:
        # Populate hashes
        for path in Path("expert_images").glob("*.png"):
            hashes[imagehash.phash(Image.open(path))] = path
    input_hash = imagehash.phash(image)

    replay_image_filename = "expert_images/" + str(data_frame) + ".png"
    replay_image_hash = imagehash.phash(Image.open(replay_image_filename))
    # assert input_hash == replay_image_hash, f"{input_hash} != {replay_image_hash}"

    if input_hash in hashes:
        # print("MATCH", hashes[input_hash])
        return hashes[input_hash]
    return None


inference_model = None
inference_dataset = None


def score(
    images, controller_buffer, ground_truth_controller, reward_history, data_frame
):
    global inference_model
    global inference_dataset
    print("Scoring", data_frame)
    if inference_model is None:
        inference_model = LitClassification.load_from_checkpoint(
            "timm_il_models/best_model-v6.ckpt"
        ).cpu()
        inference_model.eval()
        inference_dataset = NESDataset(DATA_PATH, train=False)
    with torch.no_grad():
        label_logits, value = inference_model(
            images.unsqueeze(0),
            controller_buffer.unsqueeze(0),
            reward_history.unsqueeze(0),
        )
        label_logits = label_logits.squeeze(0)
        value = value.squeeze(0)
        print("label_logits", label_logits)
        probs = Categorical(logits=label_logits)
        # label_probs = torch.nn.functional.softmax(label_logits)
        drawn_action_index = probs.sample()
        # print(label_probs)
        # drawn_action_index = torch.argmax(label_probs).item()
        # print(drawn_action_index)
        # print(inference_model.int_label_map)
        # drawn_action = inference_model.int_label_map[drawn_action_index.item()]
        drawn_action = inference_model.actor.convert_index_to_input_array(
            drawn_action_index.item()
        )

        if False:  # Check against ground truth
            image_stack, past_inputs, label_int = inference_dataset[int(data_frame) - 3]
            assert torch.equal(
                past_inputs, controller_buffer
            ), f"{past_inputs} != {controller_buffer}"
            if not torch.equal(image_stack, images):
                print(image_stack[3].mean(), images[3].mean())
                assert torch.equal(
                    image_stack[0], images[0]
                ), f"{image_stack[0]} != {images[0]}"
                assert torch.equal(
                    image_stack[1], images[0]
                ), f"{image_stack[1]} != {images[1]}"
                assert torch.equal(
                    image_stack[2], images[0]
                ), f"{image_stack[2]} != {images[2]}"
                assert torch.equal(
                    image_stack[3], images[0]
                ), f"{image_stack[3]} != {images[3]}"

        return drawn_action, probs.log_prob(drawn_action_index), probs.entropy(), value


def bcdToInt(bcd_bytes):
    value = 0
    for x in range(0, len(bcd_bytes)):
        value *= 10
        value += bcd_bytes[x]
    return value


def compute_reward_map(last_reward_map: RewardMap | None, ram):
    high_score_bytes = ram[0x07DD:0x07E3]
    score = bcdToInt(high_score_bytes) * 10

    time_left_bytes = ram[0x07F8:0x07FB]
    time_left = bcdToInt(time_left_bytes)

    coins = ram[0x75E]
    world = ram[0x75F]
    level = ram[0x760]
    powerup_level = ram[0x756]
    left_pos = (ram[0x006D] * 256) + ram[0x0086]
    player_is_dying = (ram[0xE] & 0x6) > 0
    lives = ram[0x75A] - player_is_dying.int()

    reward_map = RewardMap(
        score=score,
        time_left=time_left,
        coins=coins,
        lives=lives,
        world=world,
        level=level,
        left_pos=left_pos,
        powerup_level=powerup_level,
        player_is_dying=player_is_dying,
    )

    return reward_map, RewardMap.reward_vector(last_reward_map, reward_map)


if __name__ == "__main__":
    data = ClassificationData()
    model = LitClassification()
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="auto",
        logger=TensorBoardLogger("logs/", name="timm_il_logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                mode="min",
                patience=100,
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="timm_il_models",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model",
            ),
        ],
    )
    trainer.fit(model, data)
