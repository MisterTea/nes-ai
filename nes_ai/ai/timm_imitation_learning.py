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
from nes_ai.ai.rollout_data import RolloutData

# avail_pretrained_models = timm.list_models(pretrained=True)
# print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 32
REWARD_VECTOR_SIZE = 8

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def l1_regularization(model):
    l1_reg = 0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_reg / len(list(model.parameters()))


class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic(self.actor.trunk, self.actor.trunk_transforms)
        self.actor_loss_fn = torch.nn.CrossEntropyLoss()
        self.value_loss_fn = torch.nn.SmoothL1Loss()
        self.automatic_optimization = False
        self.rollout_data = RolloutData(DATA_PATH, readonly=True)

    def forward(self, images, past_inputs, past_values):
        return self.actor.forward(images, past_inputs), self.critic.forward(
            images, past_inputs, past_values
        )

    def _image_frames_to_image(self, image_frames, device):
        images = torch.stack(
            [
                torch.stack(
                    [
                        DEFAULT_TRANSFORM(
                            self.rollout_data.get_image(image_frames[i][b])
                        )
                        for i in range(4)
                    ],
                )
                for b in range(len(image_frames[0]))
            ]
        ).to(device=device)
        return images

    def training_step(self, batch):
        actor_opt, critic_opt = self.optimizers()

        (
            images,
            value_vectors,
            past_inputs,
            past_values,
            targets,
            log_prob,
            advantages,
        ) = batch

        # images = self._image_frames_to_image(image_frames, past_inputs.device)

        value_map_outputs = self.critic.forward(images, past_inputs, past_values)

        assert (
            value_map_outputs.shape == value_vectors.shape
        ), f"{value_map_outputs.shape} != {value_vectors.shape}"
        value_loss = 0.01 * self.value_loss_fn(value_map_outputs, value_vectors)
        self.log("value_train_loss", value_loss, prog_bar=True)

        actor_outputs = self.actor.forward(images, past_inputs)

        target_indices = self.actor.convert_input_array_to_index(targets)

        actor_loss = self.actor_loss_fn(actor_outputs, target_indices.long())
        self.log("actor_train_loss", actor_loss, prog_bar=True)

        loss = actor_loss + 0.01 * l1_regularization(self.actor)

        actor_opt.zero_grad()
        self.manual_backward(loss)
        actor_opt.step()

        loss = value_loss + 0.01 * l1_regularization(self.critic)

        actor_opt.zero_grad()
        self.manual_backward(loss)
        actor_opt.step()

        self.log("train_loss", actor_loss + value_loss, prog_bar=False)

        actor_sch, critic_sch = self.lr_schedulers()

        actor_sch.step(self.trainer.callback_metrics["actor_train_loss"])
        self.log("actor_lr", actor_sch.get_last_lr()[0], prog_bar=True)
        actor_sch.step(self.trainer.callback_metrics["value_train_loss"])
        self.log("critic_lr", critic_sch.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):

        # print(batch)
        (
            images,
            value_vectors,
            past_inputs,
            past_values,
            targets,
            log_prob,
            advantages,
        ) = batch

        # images = self._image_frames_to_image(image_frames, past_inputs.device)

        actor_outputs = self.actor.forward(images, past_inputs)

        target_indices = self.actor.convert_input_array_to_index(targets)

        # print(target_indices.shape)
        # print(target_indices)
        # print(actor_outputs)
        # print(actor_outputs.shape)
        actor_loss = self.actor_loss_fn(actor_outputs, target_indices.long())
        self.log("actor_val_loss", actor_loss, prog_bar=True)

        metric = MulticlassAccuracy()
        metric.update(actor_outputs, target_indices)
        accuracy = metric.compute()
        self.log("val_acc", accuracy, prog_bar=True)

        value_map_outputs = self.critic.forward(images, past_inputs, past_values)

        assert (
            value_map_outputs.shape == value_vectors.shape
        ), f"{value_map_outputs.shape} != {value_vectors.shape}"
        value_loss = 0.01 * self.value_loss_fn(value_map_outputs, value_vectors)
        self.log("value_val_loss", value_loss, prog_bar=True)

        self.log("val_loss", actor_loss + value_loss, prog_bar=False)

        return actor_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001 * BATCH_SIZE)
        actor_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=3000
                ),
                # "monitor": "actor_train_loss",
                # "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.00001 * BATCH_SIZE)
        critic_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2000
                ),
                # "monitor": "critic_train_loss",
                # "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return actor_opt, critic_opt


DATA_PATH = Path("./data/1_1_expert")


class ClassificationData(pl.LightningDataModule):
    def __init__(self, imitation_learning=True):
        super().__init__()
        self.imitation_learning = imitation_learning

    def train_dataloader(self):
        # Spin up dummy set to compute example weights
        dummy_train_dataset = NESDataset(
            DATA_PATH, train=True, imitation_learning=self.imitation_learning
        )
        dummy_train_dataset.bootstrap()
        example_weights = dummy_train_dataset.example_weights
        assert len(example_weights) == len(
            dummy_train_dataset
        ), f"{len(example_weights)} != {len(dummy_train_dataset)}"

        train_dataset = NESDataset(
            DATA_PATH, train=True, imitation_learning=self.imitation_learning
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
            num_workers=10,  # shuffle=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                example_weights, BATCH_SIZE * 100, replacement=True
            ),
        )

    def val_dataloader(self):
        val_dataset = NESDataset(DATA_PATH, train=False, imitation_learning=True)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
            num_workers=10,
            # sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        )


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
                monitor="actor_val_loss",
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
