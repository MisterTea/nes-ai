import math
import multiprocessing
import time
from multiprocessing import Process

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.distributions import Normal
from torch.utils.data import IterableDataset
from torcheval.metrics import MulticlassAccuracy

from nes_ai.ai.deep_parallel_gmm import DeepParallelGMM
from nes_ai.ai.game_sim_dataset import GameSimulationDataset

WORLD_DIMENSIONS = 1


class ForwardLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DeepParallelGMM(WORLD_DIMENSIONS, 1024, WORLD_DIMENSIONS, 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_data, output_data = batch

        assert input_data.shape[0] == 1
        input_data = input_data.squeeze(0).float()

        assert output_data.shape[0] == 1
        output_data = output_data.squeeze(0).float()

        outputs, means, stds, weights = self.model(input_data / 255.0, output_data)
        model_loss = 1.0 * (-torch.mean(outputs))

        weight_l1_loss = 1.0 * torch.abs(weights).mean()
        mean_l1_loss = 100.0 * torch.nn.L1Loss()(means.mean(), output_data.mean())

        weight_clamp_loss = 1000 * torch.nn.L1Loss()(
            weights, torch.clamp(weights.detach(), min=2e-2)
        )
        std_clamp_loss = 1000 * torch.nn.L1Loss()(
            stds, torch.clamp(stds.detach(), min=2e-2)
        )

        loss = (
            model_loss
            + weight_l1_loss
            + mean_l1_loss
            + weight_clamp_loss
            + std_clamp_loss
        )

        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        self.log("model_loss", model_loss, prog_bar=True)
        self.log("weight_l1_loss", weight_l1_loss, prog_bar=True)
        self.log("mean_l1_loss", mean_l1_loss, prog_bar=True)
        self.log("weight_clamp_loss", weight_clamp_loss, prog_bar=True)
        self.log("std_clamp_loss", std_clamp_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_data, output_data = batch

        assert input_data.shape[0] == 1
        input_data = input_data.squeeze(0)
        input_data_float = input_data.float()

        assert output_data.shape[0] == 1
        output_data = output_data.squeeze(0)
        output_data_float = output_data

        model_output_float = self.model.sample(input_data_float / 255.0)

        assert model_output_float.shape == output_data_float.shape

        model_output = torch.round(
            torch.clamp(model_output_float, min=0, max=255.0)
        ).to(dtype=torch.int)

        metric = MulticlassAccuracy(num_classes=256)
        metric.update(model_output.flatten(), output_data.flatten())

        self.log("val_acc", metric.compute().item(), prog_bar=True)

        # print("SCORES")
        # print(quantized_scores)
        # print(y)

        l1_diff = torch.mean(torch.abs(model_output - output_data_float).float())

        self.log("val_diff", l1_diff, prog_bar=True)

        return metric.compute().item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=10
                ),
                "monitor": "loss",
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
        WORLD_DIMENSIONS,
        10,
        3000,
        queue,
    )
    val_dataset = GameSimulationDataset(
        "val",
        WORLD_DIMENSIONS,
        1,
        3000,
        queue,
    )

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="mps",
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_acc",
                min_delta=0.0001,
                mode="max",
                patience=500,
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
        x = torch.randint(0, 256, (WORLD_DIMENSIONS,), dtype=torch.int)
        y = (x > 128).int()
        try:
            queue.put((x.numpy(), y.numpy()), timeout=1.0)
        except BaseException as e:
            print(e)
            time.sleep(5.0)
            pass


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    queue = multiprocessing.Queue()

    p = Process(target=push_games, args=(queue,))
    p.start()

    main(queue)

    p.kill()
