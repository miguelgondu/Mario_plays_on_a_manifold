"""
We consider something like a one-layer hierarchical VAE
to see if this solves the problem of having high cost outside
the support.        
"""
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.tensor import Tensor
from torch.distributions import Dirichlet
from torch.utils.tensorboard import SummaryWriter

from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import get_img_from_level


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class VAEMarioDirichlet(nn.Module):
    def __init__(
        self,
        w: int,
        h: int,
        z_dim: int,
        n_sprites: int = 11,
        h_dims: List[int] = None,
    ):
        super(VAEMarioDirichlet, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites

        self.z_dim = z_dim or 64
        self.h_dims = h_dims or [256, 128]

        # Adding the input layer with onehot encoding
        # (assuming that the views are inside the net)
        self.h_dims = [self.input_dim] + self.h_dims
        modules = []
        for dim_1, dim_2 in zip(self.h_dims[:-1], self.h_dims[1:]):
            if dim_1 == self.h_dims[0]:
                modules.append(
                    nn.Sequential(
                        View([-1, self.input_dim]), nn.Linear(dim_1, dim_2), nn.Tanh()
                    )
                )
            else:
                modules.append(nn.Sequential(nn.Linear(dim_1, dim_2), nn.Tanh()))

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(nn.Linear(self.h_dims[-1], z_dim))
        self.fc_var = nn.Sequential(nn.Linear(self.h_dims[-1], z_dim))

        # dec_dims = self.h_dims.copy() + [z_dim]
        # dec_dims.reverse()
        # dec_modules = []
        # for dim_1, dim_2 in zip(dec_dims[:-1], dec_dims[1:]):
        #     if dim_2 != dec_dims[-1]:
        #         dec_modules.append(nn.Sequential(nn.Linear(dim_1, dim_2), nn.Tanh()))
        #     else:
        #         dec_modules.append(
        #             nn.Sequential(
        #                 nn.Linear(dim_1, dim_2),
        #                 # View([-1, self.n_sprites, self.h, self.w]),
        #                 # nn.LogSoftmax(dim=1),
        #             )
        #         )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim),
            nn.Tanh(),
        )

        self.dec_alpha = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Softplus()
        )

        # print(self)

    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def _decode_sample(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        dec_alphas = self.dec_alpha(result)

        return dec_alphas.contiguous()

    def decode(self, z: Tensor) -> Tensor:
        # Decode this z
        dec_alphas = self._decode_sample(z)

        # This doesn't work
        d = Dirichlet(dec_alphas.reshape(-1, self.h, self.w, self.n_sprites))
        x_prime = torch.log(d.rsample())

        # This does work
        # x_prime = torch.ones_like(dec_alphas).reshape(
        #     -1, self.h, self.w, self.n_sprites
        # )

        return x_prime.permute(0, 3, 1, 2)

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        rvs = torch.randn_like(std)

        return rvs.mul(std).add_(mu)

    def forward(self, x: Tensor) -> List[Tensor]:
        # Does a forward pass through the network.

        mu, log_var = self.encode(x.view(-1, self.input_dim))

        # Sample z from p(z|x)
        z = self.reparametrize(mu, log_var)

        # Decode this z
        x_prime = self.decode(z)

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
