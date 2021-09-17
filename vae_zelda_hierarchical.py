from typing import List
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch as t
from torch.distributions import Normal, Categorical, kl_divergence
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter

from shapeguard import ShapeGuard

from zelda_utils.plotting import get_img_from_level


def preprocess_raw_data() -> np.ndarray:
    levels_text = np.load("./data/raw/zelda_levels_text.npz")["levels"]

    encoding = {
        "w": 0,
        "A": 1,
        ".": 2,
        "g": 3,
        "1": 4,
        "2": 5,
        "3": 6,
        "+": 7,
    }

    levels_encoded = np.zeros_like(levels_text)
    for k, v in encoding.items():
        levels_encoded[levels_text == k] = v

    levels_encoded = levels_encoded.astype(int)
    B, h, w = levels_encoded.shape
    print(levels_encoded)

    levels_onehot = np.zeros((B, len(encoding), h, w))
    for b in range(B):
        for i, j in product(range(h), range(w)):
            c = levels_encoded[b, i, j]
            levels_onehot[b, c, i, j] = 1.0

    return levels_onehot


def load_data(training_percentage=0.8, shuffle_seed=0) -> List[t.Tensor]:
    data = preprocess_raw_data()

    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = t.from_numpy(training_data).type(t.float)
    test_tensors = t.from_numpy(testing_data).type(t.float)

    return training_tensors, test_tensors


class VAEZeldaHierarchical(t.nn.Module):
    def __init__(self) -> None:
        super(VAEZeldaHierarchical, self).__init__()
        self.w = self.h = 16
        self.n_sprites = 8
        self.z_dim = 2

        self.input_dim = 16 * 16 * 8
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        self.encoder = t.nn.Sequential(
            t.nn.Linear(self.input_dim, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, 128),
            t.nn.ReLU(),
        )
        self.enc_mu = t.nn.Sequential(t.nn.Linear(128, self.z_dim))
        self.enc_var = t.nn.Sequential(t.nn.Linear(128, self.z_dim))

        self.decoder = t.nn.Sequential(
            t.nn.Linear(self.z_dim, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, self.input_dim),
            t.nn.ReLU(),
        )
        self.dec_mu = t.nn.Linear(self.input_dim, self.input_dim)
        self.dec_var = t.nn.Linear(self.input_dim, self.input_dim)

        self.p_z = Normal(
            t.zeros(self.z_dim, device=self.device),
            t.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data()

        print(self)

    def encode(self, x: t.Tensor) -> Normal:
        x.sg(("b", self.n_sprites, self.h, self.w))
        x = x.view(-1, self.input_dim).sg("bx")
        res = self.encoder(x)
        mu = self.enc_mu(res)
        log_var = self.enc_var(res)

        return Normal(mu, t.exp(0.5 * log_var))

    def _intermediate_distribution(self, z: t.Tensor) -> Normal:
        z.sg(("b", self.z_dim))
        result = self.decoder(z)
        dec_mu = self.dec_mu(result)
        dec_log_var = self.dec_var(result)

        return Normal(dec_mu, t.exp(0.5 * dec_log_var))

    def decode(self, z: t.Tensor) -> Categorical:
        dec_dist = self._intermediate_distribution(z).sg("bx")
        samples = dec_dist.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: t.Tensor) -> List[Distribution]:
        ShapeGuard.reset()
        x.sg("bshw")
        q_z_given_x = self.encode(x)

        z = q_z_given_x.rsample().sg(("bz"))

        p_x_given_z = self.decode(z).sg("bhw")

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: t.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> t.Tensor:
        x_ = x.argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def plot_grid(
        self, x_lims=(-5, 5), y_lims=(-5, 5), n_rows=10, n_cols=10, argmax=True, ax=None
    ):
        ShapeGuard.reset()
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = t.Tensor([[a, b] for a in reversed(z1) for b in z2])

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
                for row in images.reshape((n_cols, n_rows, 24 * self.h, 24 * self.w, 3))
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

        ShapeGuard.reset()
        zs = self.p_z.sample((64,)).to(self.device)
        recons_dist = self.decode(zs)
        recons = recons_dist.sample().cpu().detach().numpy()
        levels = np.array([get_img_from_level(recon) for recon in recons])
        levels = 255 - levels
        writer.add_image("reconstructions", levels, step_id, dataformats="NHWC")
