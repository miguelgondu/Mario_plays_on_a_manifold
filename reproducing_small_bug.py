import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.distributions import Dirichlet, Categorical


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.one_layer = nn.Sequential(nn.Linear(2, 4 * 3), nn.Softplus())

    def forward(self, z: Tensor):
        alphas = self.one_layer(z)
        d = Dirichlet(alphas.reshape(-1, 4, 3))
        sample = d.rsample()
        return sample

    def loss(self, sample: Tensor):
        # sample = self.forward(z)
        cat1 = Categorical(sample)
        cat2 = Categorical((1 / 3) * torch.ones_like(sample))
        return torch.distributions.kl_divergence(cat1, cat2).mean()


ex = Example()

for _ in range(100000):
    zs = torch.randn((64, 2))
    # print(ex(zs))
    # print(ex(zs).shape)

    # print(ex.loss(ex(zs)))
    # print(ex.loss(ex(zs)).shape)

    optimizer = torch.optim.Adam(ex.parameters())
    samples = ex(zs)
    loss = ex.loss(samples)
    # print(loss)
    loss.backward()
    optimizer.step()

    print(loss)
    print(samples[0, 0, :])
    # print(ex(zs)[0, 0, :].mean())
