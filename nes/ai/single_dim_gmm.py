import math
import torch

from torch.distributions import Normal



class GMM1D(torch.nn.Module):
    def __init__(self, n_components, device="cpu"):
        super().__init__()
        self.n_components = n_components
        self.means = torch.nn.Parameter(torch.randn(n_components).to(device))
        self.stds = torch.nn.Parameter(torch.ones(n_components).to(device))
        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components).to(device)



    def forward(self, x):
        # compute the variance
        var = self.stds**2
        log_scale = self.stds.log()
        log_probs2 = (
            -((x.expand(-1, self.n_components) - self.means) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        log_probs2 = log_probs2.t().unsqueeze(2)
        log_probs2[:,:,0] += torch.nan_to_num(torch.log(self.weights), nan=0.0).unsqueeze(1)

        return torch.logsumexp(log_probs2, dim=0)

    def sample(self):
        #print("SAMPLING")
        retval = torch.zeros((10000,), dtype=torch.float)
        #print(self.weights)
        #print("MEANS",self.means.tolist())
        #print("STDS",self.stds.tolist())
        #print("WEIGHTS",torch.clamp(self.weights, min=1e-1))
        c = torch.distributions.Categorical(logits=torch.clamp(self.weights, min=1e-1))
        #print("PROBS", c.probs)
        encoded_c = torch.nn.functional.one_hot(c.sample((10000,)), num_classes=self.n_components)
        #print(encoded_c.shape)
        #print(encoded_c)
        draws = torch.stack([Normal(self.means[i], self.stds[i]).sample((10000,)) for i in range(self.n_components)], dim=0).t()
        #print(draws)
        assert draws.shape == encoded_c.shape
        #print(draws.shape)
        retval = (encoded_c * draws).sum(dim=1)
        assert retval.shape == (10000,), retval.shape
        return retval




def train_gmm(n_components, learning_rate, epochs):
    model = GMM1D(n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        with torch.no_grad():
            TRAIN_SAMPLES = 10000
            data_1 = (torch.randn((TRAIN_SAMPLES)))
            data_2 = ((torch.randn((TRAIN_SAMPLES))) + 10)
            c = torch.distributions.Categorical(logits=torch.ones((2,), dtype=torch.float))
            encoded_c = torch.nn.functional.one_hot(c.sample((TRAIN_SAMPLES,)), num_classes=2)
            #print(encoded_c.shape)
            #print(torch.stack([data_1, data_2], dim=1).shape)
            data = torch.sum(encoded_c * torch.stack([data_1, data_2], dim=1), dim=1)
            #print(data.shape)
            assert data.shape == data_1.shape, f"{data.shape} != {data_1.shape}"
            data = data.unsqueeze(1)

            #print("WEIGHT SUM", model.weights.sum())
        loss = (-torch.mean(model(data))) #+ 0.1 * (torch.norm(model.weights, 2) ** 2)# + (0.01 * torch.nn.MSELoss()(model.weights.sum(),torch.ones(1))) + (10 * torch.nn.L1Loss(reduction="sum")(model.weights,torch.clamp(model.weights.detach(), min=1e-2)))

        loss += 0.5 * torch.abs(model.weights).sum()
        loss += 0.5 * torch.nn.L1Loss(reduction="sum")(model.means.mean(),data.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            #print(epoch, loss.item(), model.weights.tolist(), data.mean().item(), data.var().item())
            samples = model.sample()
            if epoch % 100 == 0:
                print(epoch, loss.item(), model.weights.tolist(), model.means.tolist(), (model.stds**2).tolist(), data.mean().item(), data.var().item(), samples.mean().item(), samples.var().item())
                print("")

    return model



# Example usage

model = train_gmm(n_components=10, learning_rate=0.001, epochs=30000)
