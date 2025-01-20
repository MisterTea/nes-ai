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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets, transforms

from nes.ai.helpers import upscale_and_get_labels
from nes.ai.nes_dataset import NESDataset

avail_pretrained_models = timm.list_models(pretrained=True)
print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 64


def _linear_block(in_features, out_features):
    return [
        torch.nn.Linear(in_features, out_features),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(out_features),
    ]


class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )
        self.head = torch.nn.Sequential(
            *_linear_block(self.trunk.num_features * 4, 1024),
            *_linear_block(1024, 1024),
            torch.nn.Linear(1024, 5),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch

        trunk_output = torch.cat(
            [self.trunk(images[:, x, :, :, :]) for x in range(4)], dim=1
        )
        outputs = self.head(trunk_output)
        # print("***")
        # print(targets)
        # print(outputs)
        # print(targets.shape)
        # print(outputs.shape)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # print(batch)
        images, targets = batch

        trunk_output = torch.cat(
            [self.trunk(images[:, x, :, :, :]) for x in range(4)], dim=1
        )
        outputs = self.head(trunk_output)

        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        metric = MulticlassAccuracy()
        metric.update(outputs, targets)
        accuracy = metric.compute()
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00001 * BATCH_SIZE)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2
                ),
                "monitor": "train_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class ClassificationData(pl.LightningDataModule):

    def train_dataloader(self):
        train_dataset = NESDataset()
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10
        )

    def val_dataloader(self):
        val_dataset = NESDataset()
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
            sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        )


if __name__ == "__main__":
    model = LitClassification()
    data = ClassificationData()
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="mps",
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
