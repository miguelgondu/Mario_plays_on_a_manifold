from typing import List

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.tensor import Tensor
from torch.utils.tensorboard import SummaryWriter

from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import get_img_from_level


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class CondVAEMario(nn.Module):
    def __init__(
        self,
        w: int,
        h: int,
        z_dim: int,
        n_sprites: int = 11,
        h_dims: List[int] = None,
    ):
        super(CondVAEMario, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites + 1

        self.z_dim = z_dim or 64
        self.h_dims = h_dims or [256, 128]

        # Adding the input layer with onehot encoding
        # (assuming that the views are inside the net)
        self.h_dims = [self.input_dim] + self.h_dims
        modules = []
        for dim_1, dim_2 in zip(self.h_dims[:-1], self.h_dims[1:]):
            modules.append(nn.Sequential(nn.Linear(dim_1, dim_2), nn.Tanh()))

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(nn.Linear(self.h_dims[-1], z_dim))
        self.fc_var = nn.Sequential(nn.Linear(self.h_dims[-1], z_dim))

        dec_dims = self.h_dims.copy() + [z_dim + 1]
        dec_dims.reverse()
        dec_modules = []
        for dim_1, dim_2 in zip(dec_dims[:-1], dec_dims[1:]):
            if dim_2 != dec_dims[-1]:
                dec_modules.append(nn.Sequential(nn.Linear(dim_1, dim_2), nn.Tanh()))
            else:
                dec_modules.append(
                    nn.Sequential(
                        nn.Linear(dim_1, dim_2),
                        nn.LogSoftmax(dim=1),
                    )
                )

        self.decoder = nn.Sequential(*dec_modules)
        # print(self)

    def encode(self, x: Tensor, c: Tensor) -> List[Tensor]:
        x_and_class = torch.cat((x, c.unsqueeze(1)), dim=-1).type(torch.float)
        result = self.encoder(x_and_class)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        z_and_class = torch.cat((z, c.unsqueeze(1)), dim=-1).type(torch.float)
        result = self.decoder(z_and_class)
        return result

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        rvs = torch.randn_like(std)

        return rvs.mul(std).add_(mu)

    def forward(self, x: Tensor, c: Tensor) -> List[Tensor]:
        # Does a forward pass through the network.
        # batch_size, n_classes, h, w = x.shape
        x = x.view(-1, self.input_dim - 1)
        mu, log_var = self.encode(x, c)

        # Sample z from p(z|x)
        z = self.reparametrize(mu, log_var)

        # Decode this z
        x_prime = self.decode(z, c)

        return [x_prime, x, mu, log_var]

    def report(
        self,
        writer: SummaryWriter,
        batch_id: int,
        KLD: float,
        CEL: float,
        zs: Tensor,
        epoch: int,
    ):
        writer.add_scalar("KLD", KLD, batch_id)
        writer.add_scalar("CEL", CEL, batch_id)
        writer.add_scalar("loss", KLD + CEL, batch_id)

        samples = self.decoder(zs)
        samples = onehot_to_levels(samples.detach().numpy())
        samples = np.array([get_img_from_level(level) for level in samples])

        writer.add_images(f"samples_{epoch}", samples, batch_id, dataformats="NHWC")

    # def loss_function(self, x_prime, x, mu, log_var):
    #     BCE = F.binary_cross_entropy(x_prime.view(-1, 28*28), x.view(-1, 784), reduction="sum")
    #     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #     return BCE + KLD
