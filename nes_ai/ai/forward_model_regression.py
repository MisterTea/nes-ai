import math
import time
from multiprocessing import Process

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import multiprocessing
from torch.utils.data import IterableDataset
from torcheval.metrics import MulticlassAccuracy

from nes_ai.ai.game_sim_dataset import GameSimulationDataset


def _linear_block(in_features, out_features):
    return [
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_features),
    ]


NUM_OBJECTS = 256
HIDDEN_SIZE = 1024


class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.model = nn.Sequential(
            *_linear_block(NUM_OBJECTS, HIDDEN_SIZE),
            *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            # *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            # *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            # *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            *_linear_block(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, NUM_OBJECTS),
        )

    def l1_regularization(self):
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg / len(list(self.parameters()))

    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param**2)
        return l2_reg / len(list(self.parameters()))

    def forward(self, x):
        scores = self.model(x.float())
        return scores


class ForwardLightningModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = ForwardModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]
        y = y[0]

        scores = self(x.float() / 255.0)
        assert scores.shape == y.shape

        gt_loss = nn.SmoothL1Loss()(scores, y.float())

        # # compute the variance
        # var = 1.0
        # log_scale = 0
        # log_probs2 = (
        #     -((y.float() - scores) ** 2) / (2 * var)
        #     - log_scale
        #     - math.log(math.sqrt(2 * math.pi))
        # )
        # gt_loss = -torch.mean(log_probs2)

        l1_loss = 0.00001 * self.model.l1_regularization()
        l2_loss = 0.00001 * self.model.l2_regularization()

        loss = gt_loss + l1_loss + l2_loss

        with torch.no_grad():
            self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
            # self.log('x_mean', scores.mean(), prog_bar=True)
            # self.log('y_mean', y.float().mean(), prog_bar=True)
            self.log("train_loss", loss, prog_bar=True)
            self.log("gt_loss", gt_loss, prog_bar=True)
            self.log("l1_loss", l1_loss, prog_bar=True)
            self.log("l2_loss", l2_loss, prog_bar=True)

        if False:
            with torch.no_grad():
                quantized_scores = torch.round(
                    torch.clamp(scores, min=0.0, max=255.0)
                ).to(dtype=torch.int)

                metric = MulticlassAccuracy()
                metric.update(quantized_scores.flatten(), y.flatten())

                self.log("train_acc", metric.compute().item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]
        y = y[0]

        scores = self(x.float() / 255.0)

        quantized_scores = torch.round(torch.clamp(scores, min=0.0, max=255.0)).to(
            dtype=torch.int
        )

        # print("SCORES")
        # print(quantized_scores)
        # print(y)

        val_diff = torch.mean(torch.abs(quantized_scores - y).float())

        metric = MulticlassAccuracy(num_classes=256)
        metric.update(quantized_scores.flatten(), y.flatten())

        self.log("val_acc", metric.compute().item(), prog_bar=True)

        self.log("val_diff", val_diff, prog_bar=True)
        return val_diff

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2
                ),
                "monitor": "val_diff",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

        # return optim.Shampoo(
        #     self.parameters(),
        #     lr=1e-3,
        #     momentum=0.0,
        #     weight_decay=0.0,
        #     epsilon=1e-4,
        #     update_freq=1,
        # )

        # return DistributedShampoo(
        #     self.parameters(),
        #     lr=0.0001,
        #     betas=(0.9, 0.999),
        #     epsilon=1e-4,
        #     weight_decay=1e-05,
        #     max_preconditioner_dim=8192,
        #     precondition_frequency=100,
        #     use_decoupled_weight_decay=False,
        #     grafting_config=AdamGraftingConfig(
        #         beta2=0.999,
        #         epsilon=1e-4,
        #     ),
        # )


def main(queue: multiprocessing.Queue):
    model = ForwardLightningModel()

    train_dataset = GameSimulationDataset(
        "train",
        NUM_OBJECTS,
        100,
        3000,
        queue,
    )
    val_dataset = GameSimulationDataset(
        "val",
        NUM_OBJECTS,
        10,
        50,
        queue,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="mps",
        logger=TensorBoardLogger("logs/", name="fwr_logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_diff",
                min_delta=0.0001,
                mode="min",
                patience=100,
                verbose=True,
            ),
        ],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def push_games(queue: multiprocessing.Queue):
    while True:
        x = torch.randint(0, 256, (NUM_OBJECTS,), dtype=torch.int)
        y = (x > 128).int()  # * x.int()
        try:
            queue.put((x.numpy(), y.numpy()), timeout=1.0)
        except BaseException as e:
            print(e)
            time.sleep(1.0)
            pass


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    # multiprocessing.set_sharing_strategy('file_system')

    queue = multiprocessing.Queue()

    p = Process(target=push_games, args=(queue,))
    p.start()

    main(queue)

    p.kill()
