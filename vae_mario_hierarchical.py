"""
We consider something like a one-layer hierarchical VAE
to see if this solves the problem of having high cost outside
the support.        
"""
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
from torch.nn import functional as F
import torch.nn as nn
from torch.tensor import Tensor
from torch.utils.tensorboard import SummaryWriter

from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import get_img_from_level


def load_data(
    training_percentage=0.8,
    test_percentage=None,
    shuffle_seed=0,
    only_playable=False,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    if only_playable:
        data = np.load("./data/processed/all_playable_levels_onehot.npz")["levels"]
    else:
        data = np.load("./data/processed/all_levels_onehot.npz")["levels"]

    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors.to(device), test_tensors.to(device)


class VAEMarioHierarchical(nn.Module):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
    ):
        super(VAEMarioHierarchical, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim),
            nn.Tanh(),
        ).to(self.device)
        self.dec_mu = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.dec_var = nn.Linear(self.input_dim, self.input_dim).to(self.device)

        self.p_z = Normal(torch.zeros(self.z_dim), torch.ones(self.z_dim)).to(
            self.device
        )

        self.train_data, self.test_data = load_data(device=self.device)

        print(self)

    def encode(self, x: Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def _intermediate_distribution(self, z: Tensor) -> Normal:
        result = self.decoder(z.to(self.device))
        dec_mu = self.dec_mu(result)
        dec_log_var = self.dec_var(result)

        return Normal(dec_mu, torch.exp(0.5 * dec_log_var))

    def decode(self, z: Tensor) -> Categorical:
        # Returns p(x | z) = Cat(logits=samples from _intermediate_distribution)
        dec_dist = self._intermediate_distribution(z.to(self.device))
        samples = dec_dist.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: Tensor) -> List[Distribution]:
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> Tensor:
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def plot_grid(
        self, x_lims=(-5, 5), y_lims=(-5, 5), n_rows=10, n_cols=10, argmax=True, ax=None
    ):
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])

        images_dist = self.decode(zs)
        if argmax:
            images = images_dist.probs.argmax(dim=-1)
        else:
            images = images_dist.sample()

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
        recons_dist = self.decode(zs)
        recons = recons_dist.sample().cpu().detach().numpy()
        levels = np.array([get_img_from_level(recon) for recon in recons])
        levels = 255 - levels
        writer.add_image("reconstructions", levels, step_id, dataformats="NHWC")
