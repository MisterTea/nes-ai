import math
import torch

import torch
import torch.nn as nn


def _linear_block(in_features, out_features):
    return [
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_features)
    ]


class DeepParallelGMM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_dim, n_components, device="mps"):
        super().__init__()
        self.input_dim = input_dim
        self.n_dim = n_dim
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            *_linear_block(input_dim, self.hidden_dim),
            #*_linear_block(self.hidden_dim, self.hidden_dim),
        ).to(device)
        self.means = torch.nn.Linear(self.hidden_dim, n_dim * n_components).to(device)
        self.stds = torch.nn.Linear(self.hidden_dim, n_dim * n_components).to(device)
        self.weights = torch.nn.Linear(self.hidden_dim, n_dim * n_components).to(device)
        # self.means = torch.nn.Parameter(torch.randn(n_dim, n_components).to(device))
        # self.stds = torch.nn.Parameter(torch.ones(n_dim, n_components).to(device))
        # self.weights = torch.nn.Parameter(torch.ones(n_dim, n_components) / n_components).to(device)

        with torch.no_grad():
            self.stds.weight[:] = 1.0
            self.weights.weight[:] = 1.0 / n_components


    def get_gmm_params(self, inputs):
        batch_size = inputs.shape[0]
        # Compute means/stds/weights
        trunked_x = self.trunk(inputs)
        means = self.means(trunked_x).reshape(batch_size, self.n_dim, self.n_components)
        stds = torch.clamp(self.stds(trunked_x).reshape(batch_size, self.n_dim, self.n_components), min=1e-3)
        weights = torch.clamp(self.weights(trunked_x).reshape(batch_size, self.n_dim, self.n_components), min=1e-3)

        return means, stds, weights

    def forward(self, inputs, ground_truth):
        batch_size = inputs.shape[0]
        assert batch_size == ground_truth.shape[0]

        # Batch / dim / component
        ground_truth = ground_truth.unsqueeze(2).expand(-1, -1, self.n_components)

        # Compute means/stds/weights
        means, stds, weights = self.get_gmm_params(inputs)

        # compute the variance
        var = stds**2
        log_scale = stds.log()
        log_probs2 = (
            -((ground_truth - means) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        #log_probs2 = log_probs2.unsqueeze(1)
        log_weights = torch.log(weights)
        if torch.any(torch.isnan(log_weights)):
            raise ValueError("Oops")
        log_probs2 += log_weights #torch.nan_to_num(torch.log(weights), nan=0.0)

        return torch.logsumexp(log_probs2, dim=2), means, stds, weights

    def sample(self, inputs):
        # Compute means/stds/weights
        batch_size = inputs.shape[0]

        means, stds, weights = self.get_gmm_params(inputs)


        #print("SAMPLING")
        #print(self.weights)
        #print("MEANS",self.means.tolist())
        #print("STDS",self.stds.tolist())
        #print("WEIGHTS",torch.clamp(self.weights, min=1e-1))

        c = torch.distributions.Categorical(logits=weights)
        #print(c.sample((10000,)))
        #print(c.sample((10000,)).shape)
        component_chosen = c.sample((1,)).squeeze(0)
        encoded_c = torch.nn.functional.one_hot(component_chosen, num_classes=self.n_components).to(means.device)
        # print("***")
        # print(weights.shape)
        # print(component_chosen.shape)
        # print(encoded_c.shape)
        # print(encoded_c)
        draws = (torch.randn((batch_size,self.n_dim, self.n_components), device=means.device) * stds.expand(batch_size,self.n_dim, self.n_components)) + means.expand(batch_size,self.n_dim, self.n_components)
        assert draws.shape == encoded_c.shape, f"{draws.shape} != {encoded_c.shape}"
        #print(retval.shape)
        retval_fast = (encoded_c * draws).sum(dim=2)
        #print(retval.shape)
        return retval_fast
