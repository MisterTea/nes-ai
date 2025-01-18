import multiprocessing
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from multiprocessing import Process
from torcheval.metrics import MulticlassAccuracy
import torch_optimizer as optim
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class GameSimulationIterator:
    def __init__(
        self,
        name: str,
        world_dim: int,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        queue: multiprocessing.Queue,
    ):
        super().__init__()
        self.name = name
        self.world_dim = world_dim
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
            x = torch.zeros((self.games_per_minibatch, self.world_dim), dtype=torch.int)
            y = torch.zeros((self.games_per_minibatch, self.world_dim), dtype=torch.int)
            for i in range(self.games_per_minibatch):
                x_list, y_list = self.queue.get(timeout=1.0)
                x[i,:] = torch.from_numpy(x_list)
                y[i, :] = torch.from_numpy(y_list)
            return (x,y)



class GameSimulationDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        world_dim: int,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        queue: multiprocessing.Queue,
    ):
        assert minibatches_per_epoch > 0
        self.minibatches_per_epoch = minibatches_per_epoch
        self.name = name
        self.world_dim = world_dim
        self.games_per_minibatch = games_per_minibatch
        self.queue = queue

    def __len__(self):
        return self.minibatches_per_epoch

    def __iter__(self):
        gsi = GameSimulationIterator(
            self.name,
            self.world_dim,
            self.minibatches_per_epoch,
            self.games_per_minibatch,
            self.queue,
        )
        return gsi

