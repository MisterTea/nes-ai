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
from torch.utils.data import IterableDataset
from torcheval.metrics import MulticlassAccuracy

from nes_ai.ai.game_sim_dataset import GameSimulationDataset


def _linear_block(in_features, out_features):
    return [
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_features),
    ]


NUM_CLASSES = 2
NUM_OBJECTS = 512


class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.model = nn.Sequential(
            *_linear_block(NUM_OBJECTS, 2048),
            *_linear_block(2048, 2048),
            *_linear_block(2048, 2048),
            nn.Linear(2048, NUM_OBJECTS * NUM_CLASSES),
        )

    def l1_regularization(self):
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg / len(list(self.parameters()))

    def forward(self, x):
        logits = self.model(x.float())
        logits = logits.reshape(-1, NUM_OBJECTS, NUM_CLASSES)
        return logits


class ForwardLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ForwardModel()
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]
        y = y[0]
        logits = self(x)

        logits = logits.reshape(-1, NUM_CLASSES)

        y = y.reshape(-1)

        assert logits.shape[0] == y.shape[0]
        assert logits.shape[1] == NUM_CLASSES

        l1_penalty = self.model.l1_regularization()

        loss = self.loss_fn(logits, y) + (0.0001 * l1_penalty)

        predicted_class = torch.argmax(logits, dim=1)

        metric = MulticlassAccuracy()
        metric.update(predicted_class, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", metric.compute().item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]
        y = y[0]
        logits = self(x)

        logits = logits.reshape(-1, NUM_CLASSES)

        y = y.reshape(-1)

        assert logits.shape[0] == y.shape[0]
        assert logits.shape[1] == NUM_CLASSES

        predicted_class = torch.argmax(logits, dim=1)

        metric = MulticlassAccuracy()
        metric.update(predicted_class, y)

        self.log("val_acc", metric.compute().item(), prog_bar=True)
        return metric.compute().item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=2, verbose=True
                ),
                "monitor": "val_acc",
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
        100,
        3000,
        queue,
    )
    val_dataset = GameSimulationDataset(
        "val",
        10,
        50,
        queue,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_acc",
                min_delta=0.0001,
                mode="max",
                patience=20,
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
        x = torch.randint(0, 256, (NUM_OBJECTS,), dtype=torch.uint8)
        y = (x > 128).int() * x.int()
        try:
            queue.put((x, y), timeout=1.0)
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
