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

from nes_ai.ai.base import Actor, Critic, RewardMap
from nes_ai.ai.helpers import upscale_and_get_labels
from nes_ai.ai.nes_dataset import NESDataset
from nes_ai.ai.timm_imitation_learning import ClassificationData

# avail_pretrained_models = timm.list_models(pretrained=True)
# print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 32
REWARD_VECTOR_SIZE = 8


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
        reward_loss = 0.01 * self.reward_loss_fn(reward_map_outputs, reward_vectors)
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
        reward_loss = 0.01 * self.reward_loss_fn(reward_map_outputs, reward_vectors)
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


DATA_PATH = Path("./data/1_1_expert_shelve")


if __name__ == "__main__":
    data = ClassificationData(imitation_learning=False)
    model = LitClassification.load_from_checkpoint("timm_il_models/best_model-v3.ckpt")
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="auto",
        logger=TensorBoardLogger("logs/", name="timm_rl_logs"),
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
                dirpath="timm_rl_models",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model",
            ),
        ],
    )
    trainer.fit(model, data)
