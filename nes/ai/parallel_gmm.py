import math
import torch

from torch.distributions import Normal
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
from nes.ai.deep_parallel_gmm import DeepParallelGMM

class ParallelGMM(torch.nn.Module):
    def __init__(self, n_dim, n_components, device="mps"):
        super().__init__()
        self.n_dim = n_dim
        self.n_components = n_components
        self.means = torch.nn.Parameter(torch.randn(n_dim, n_components).to(device))
        self.stds = torch.nn.Parameter(torch.ones(n_dim, n_components).to(device))
        self.weights = torch.nn.Parameter(torch.ones(n_dim, n_components) / n_components).to(device)



    def forward(self, x):
        # Batch / dim / component
        x = x.unsqueeze(2).expand(-1, -1, self.n_components)

        # compute the variance
        var = self.stds**2
        log_scale = self.stds.log()
        log_probs2 = (
            -((x - self.means.reshape(1,self.n_dim,self.n_components)) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        #log_probs2 = log_probs2.unsqueeze(1)
        log_probs2 += torch.nan_to_num(torch.log(self.weights), nan=0.0).reshape(1,self.n_dim,self.n_components)

        return torch.logsumexp(log_probs2, dim=2)

    def sample(self):
        #print("SAMPLING")
        #print(self.weights)
        #print("MEANS",self.means.tolist())
        #print("STDS",self.stds.tolist())
        #print("WEIGHTS",torch.clamp(self.weights, min=1e-1))

        c = torch.distributions.Categorical(logits=torch.clamp(self.weights, min=1e-1))
        #print(c.sample((10000,)))
        #print(c.sample((10000,)).shape)
        encoded_c = torch.nn.functional.one_hot(c.sample((10000,)), num_classes=self.n_components).to(self.means.device)
        #print(encoded_c.shape)
        #print(encoded_c)
        draws = (torch.randn((10000,self.n_dim, self.n_components), device=self.means.device) * self.stds.expand(10000,self.n_dim, self.n_components)) + self.means.expand(10000,self.n_dim, self.n_components)
        assert draws.shape == encoded_c.shape
        #print(retval.shape)
        retval_fast = (encoded_c * draws).sum(dim=2)
        #print(retval.shape)
        return retval_fast
        #aljkdlaskjds

        retval = torch.zeros((10000,self.n_dim), dtype=torch.float)
        assert retval.shape == (10000,self.n_dim), f"{retval.shape} != (10000,{self.n_dim})"
        for x in range(self.n_dim):
            c = torch.distributions.Categorical(logits=torch.clamp(self.weights[x], min=1e-1))
            encoded_c = torch.nn.functional.one_hot(c.sample((10000,)), num_classes=self.n_components)
            #draws = torch.stack([Normal(self.means[x,i], self.stds[x,i]).sample((10000,)) for i in range(self.n_components)], dim=0).t()
            #draws = (torch.randn((10000,self.n_components)) * self.stds[x].expand(10000,self.n_components)) + self.means[x].expand(10000,self.n_components)
            draws_x = draws[:,x,:]
            assert draws_x.shape == encoded_c.shape
            print(encoded_c.shape)
            print(draws_x.shape)
            print(retval.shape)
            print((encoded_c * draws_x).sum(dim=1).shape)
            retval[:,x] = (encoded_c * draws_x).sum(dim=1)

        assert retval.shape == (10000,self.n_dim), f"{retval.shape} != (10000,{self.n_dim})"
        assert retval.shape == retval_fast.shape, f"{retval.shape} != {retval_fast.shape}"
        assert torch.equal(retval, retval_fast), f"{retval} != {retval_fast}"
        return retval
