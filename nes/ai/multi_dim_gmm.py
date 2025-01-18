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

def train_gmm(n_dim, n_components, learning_rate, epochs):
    INPUT_DIMENSIONS = 128
    model = DeepParallelGMM(INPUT_DIMENSIONS, INPUT_DIMENSIONS//2, n_dim, n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    TRAIN_SAMPLES = 1000
    input_data = torch.ones((TRAIN_SAMPLES,INPUT_DIMENSIONS), dtype=torch.float, device="mps")
    sample_input_data = torch.ones((TRAIN_SAMPLES*10,INPUT_DIMENSIONS), dtype=torch.float, device="mps")

    for epoch in range(epochs):
        with torch.no_grad():
            data_1 = (torch.randn((TRAIN_SAMPLES * n_dim)))
            data_2 = ((torch.randn((TRAIN_SAMPLES * n_dim))) + 10)
            c = torch.distributions.Categorical(logits=torch.ones((2,), dtype=torch.float))
            encoded_c = torch.nn.functional.one_hot(c.sample((TRAIN_SAMPLES * n_dim,)), num_classes=2)
            #print(encoded_c.shape)
            #print(torch.stack([data_1, data_2], dim=1).shape)
            data = torch.sum(encoded_c * torch.stack([data_1, data_2], dim=1), dim=1)
            #print(data.shape)
            assert data.shape == data_1.shape, f"{data.shape} != {data_1.shape}"
            data = data.reshape(-1, n_dim).to("mps")

            #print("WEIGHT SUM", model.weights.sum())

        outputs, means, stds, weights = model(input_data, data)
        model_loss = 1.0 * (-torch.mean(outputs)) #+ 0.1 * (torch.norm(model.weights, 2) ** 2)# + (0.01 * torch.nn.MSELoss()(model.weights.sum(),torch.ones(1))) + (10 * torch.nn.L1Loss(reduction="sum")(model.weights,torch.clamp(model.weights.detach(), min=1e-2)))

        weight_l1_loss = 1.0 * torch.abs(weights).mean()
        mean_l1_loss = 100.0 * torch.nn.L1Loss()(means.mean(),data.mean())

        weight_clamp_loss = (1000 * torch.nn.L1Loss()(weights,torch.clamp(weights.detach(), min=2e-2)))
        std_clamp_loss = (1000 * torch.nn.L1Loss()(stds,torch.clamp(stds.detach(), min=2e-2)))

        optimizer.zero_grad()
        loss = model_loss + weight_l1_loss + mean_l1_loss + weight_clamp_loss + std_clamp_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            #print(epoch, loss.item(), model.weights.tolist(), data.mean().item(), data.var().item())
            samples = model.sample(sample_input_data)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}:\n{model_loss.item()=:.2f}\n{weight_l1_loss.item()=:.2f}\n{mean_l1_loss.item()=:.2f}\n{weight_clamp_loss.item()=:.2f}\n{std_clamp_loss.item()=:.2f}\n \
                      \n{weights.mean(dim=0).tolist()=}\n{means.mean(dim=0).tolist()=}\n{(stds.mean(dim=0)**2).tolist()=}\n{data.mean(dim=0)=}\n{data.var(dim=0)=}\n{samples.mean(dim=0)=}\n{samples.var(dim=0)=}")
                #print(epoch, loss.item(), model.weights.tolist(), model.means.tolist(), (model.stds**2).tolist(), data.mean(dim=0), data.var(dim=0), samples.mean().item(), samples.var().item())
                print("")

    return model



if __name__ == "__main__":
    model = train_gmm(n_dim=16, n_components=2, learning_rate=0.001, epochs=30000)

