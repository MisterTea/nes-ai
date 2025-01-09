import multiprocessing
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from multiprocessing import Process
from torcheval.metrics import MulticlassAccuracy
import torch_optimizer as optim
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo

def _linear_block(in_features, out_features):
    return [
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_features)
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

        loss = self.loss_fn(logits, y)

        predicted_class = torch.argmax(logits, dim=1)

        metric = MulticlassAccuracy()
        metric.update(predicted_class, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', metric.compute().item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

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

class GameSimulationIterator:
    def __init__(
        self,
        name: str,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        queue: multiprocessing.Queue,
    ):
        super().__init__()
        self.name = name
        self.minibatches_per_epoch = minibatches_per_epoch
        self.games_per_minibatch = games_per_minibatch
        self.on_game = 0
        self.on_iter = 0
        self.queue = queue

    def __iter__(self):
        self.on_game = 0
        self.on_iter = 0
        return self

    def __next__(self):
        if self.on_iter == self.minibatches_per_epoch:
            raise StopIteration
        self.on_iter += 1
        with torch.no_grad():
            x = torch.zeros((self.games_per_minibatch, NUM_OBJECTS), dtype=torch.uint8)
            y = torch.zeros((self.games_per_minibatch, NUM_OBJECTS), dtype=torch.uint8)
            for i in range(self.games_per_minibatch):
                x[i,:], y[i, :] = self.queue.get()
            return (x,y)



class GameSimulationDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        queue: multiprocessing.Queue,
    ):
        assert minibatches_per_epoch > 0
        self.minibatches_per_epoch = minibatches_per_epoch
        self.name = name
        self.games_per_minibatch = games_per_minibatch
        self.queue = queue

    def __len__(self):
        return self.minibatches_per_epoch

    def __iter__(self):
        gsi = GameSimulationIterator(
            self.name,
            self.minibatches_per_epoch,
            self.games_per_minibatch,
            self.queue,
        )
        return gsi

def main(queue: multiprocessing.Queue):
    model = ForwardLightningModel()

    train_dataset = GameSimulationDataset(
        "train",
        1000,
        300,
        queue,
    )
    val_dataset = GameSimulationDataset(
        "val",
        100,
        50,
        queue,
    )

    trainer = pl.Trainer(max_epochs=100)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            pin_memory=True,
            num_workers=0,
            #persistent_workers=True,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        pin_memory=True,
        num_workers=0,
        #persistent_workers=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    queue = multiprocessing.Queue()

    p = Process(target=main, args=(queue,))
    p.start()

    while p.is_alive():
        x = torch.randint(0, 10, (NUM_OBJECTS,), dtype=torch.uint8)
        y = (x>5).int()
        queue.put((x, y))

    p.join()