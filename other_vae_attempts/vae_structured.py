"""
We try to replicate 'Structured Uncertainty Prediction Networks'
by Dorta et al. The idea is to learn the Cholesky factor of the precision
of a multivariate distribution. That way, we are enforcing non-diagonal
covariance.
"""
from typing import List

import numpy as np
import torch as t
from torch.distributions import (
    Distribution,
    MultivariateNormal,
    Normal,
    Categorical,
    kl_divergence,
)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.mario.plotting import get_img_from_level
from vae_models.vae_mario_hierarchical import load_data


class VAEStructured(nn.Module):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super(VAEStructured, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = device or t.device("cuda" if t.cuda.is_available() else "cpu")

        all_indices = t.tril_indices(self.input_dim, self.input_dim)
        mask = all_indices[0, :] != all_indices[1, :]
        self.L_lower_indices = all_indices[:, mask]
        self.L_diagonal_indices = all_indices[:, t.logical_not(mask)]

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
            nn.Tanh(),
        ).to(self.device)
        self.dec_mu = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.dec_L = nn.Linear(
            self.input_dim, (self.input_dim * (self.input_dim - 1)) // 2
        ).to(self.device)
        self.dec_L_diag = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Softplus()
        ).to(self.device)

        self.p_z = Normal(
            t.zeros(self.z_dim, device=self.device),
            t.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)

    def encode(self, x: t.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, t.exp(0.5 * log_var))

    def _intermediate_distribution(self, z: t.Tensor) -> Normal:
        result = self.decoder(z.to(self.device))
        dec_mu = self.dec_mu(result)
        dec_L = self.dec_L(result)
        dec_L_diag = self.dec_L_diag(result)

        # Populate the Cholesky factor of Sigma.
        L = t.zeros((self.input_dim, self.input_dim))
        L[self.L_lower_indices[0, :], self.L_lower_indices[1, :]] = dec_L
        L[self.L_diagonal_indices[0, :], self.L_diagonal_indices[1, :]] = dec_L_diag

        return MultivariateNormal(dec_mu, scale_tril=L)

    def decode(self, z: t.Tensor) -> Categorical:
        # Returns p(x | z) = Cat(logits=samples from _intermediate_distribution)
        dec_dist = self._intermediate_distribution(z.to(self.device))
        samples = dec_dist.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: t.Tensor) -> List[Distribution]:
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: t.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> t.Tensor:
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
    ):
        if self.z_dim != 2:
            return np.zeros((16 * 14, 16 * 14, 3))

        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = t.Tensor([[a, b] for a in reversed(z1) for b in z2])

        images_dist = self.decode(zs)
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )

        final_img = np.vstack(
            [
                np.hstack([im for im in row])
                for row in images.reshape((n_cols, n_rows, 16 * 14, 16 * 14, 3))
            ]
        )
        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img

    def report(
        self,
        writer: SummaryWriter,
        step_id: int,
        train_loss: float,
        test_loss: float,
    ):
        writer.add_scalar("train loss", train_loss, step_id)
        writer.add_scalar("test loss", test_loss, step_id)

        grid = self.plot_grid()
        grid = 255 - grid
        writer.add_image(
            "grid", grid.reshape(1, *grid.shape), step_id, dataformats="NHWC"
        )

        zs = self.p_z.sample((64,)).to(self.device)
        samples_dist = self.decode(zs)
        ress = samples_dist.sample().cpu().detach().numpy()
        levels = np.array([get_img_from_level(res) for res in ress])
        levels = 255 - levels
        writer.add_image("random samples", levels, step_id, dataformats="NHWC")

        og_levels = self.test_data[:64].to(self.device)
        _, p_x_given_z = self.forward(og_levels)
        reconstructions = p_x_given_z.probs.argmax(dim=-1).cpu().detach().numpy()
        levels = np.array([get_img_from_level(rec) for rec in reconstructions])
        levels = 255 - levels

        writer.add_image(
            "reconstructions from test set", levels, step_id, dataformats="NHWC"
        )
