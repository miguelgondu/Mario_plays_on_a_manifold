"""
This shows the dual of Nicki's trick, but only for
the usual extrapolation-to-1/C for the trick
"""

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal
from vae_models.vae_vanilla_mario import VAEMario
from sklearn.cluster import KMeans

from stochman.nnj import TranslatedSigmoid
from stochman.manifold import Manifold

from utils.metric_approximation.finite_difference import (
    approximate_metric,
    plot_approximation,
)


class VAEWithCenters(VAEMario, Manifold):
    """
    The dual of nicki's trick.
    """

    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super().__init__(w, h, z_dim, n_sprites=n_sprites, device=device)
        self.translated_sigmoid = None
        self.cluster_centers = []
        self.encodings = []

    # This method overwrites the decode of the vanilla one.
    def decode(self, z: t.Tensor, reweight: bool = True) -> Categorical:
        if reweight and len(self.cluster_centers) > 0:
            dist_to_centers = self.translated_sigmoid(self.min_distance(z)).view(
                -1, 1, 1, 1
            )
            original_categorical = super().decode(z)

            original_probs = original_categorical.probs

            reweighted_probs = (1 - dist_to_centers) * original_probs + (
                dist_to_centers
            ) * ((1 / self.n_sprites) * t.ones_like(original_probs))
            p_x_given_z = Categorical(probs=reweighted_probs)
        else:
            p_x_given_z = super().decode(z)

        return p_x_given_z

    def update_centers(
        self,
        # obstacles: t.Tensor,
        beta: float = -3.0,
        n_clusters: int = 550,
    ):
        """
        Updates the points to avoid.
        """
        training_data = self.train_data
        encodings = self.encode(training_data).mean
        self.encodings = encodings

        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(encodings.cpu().detach().numpy())
        cluster_centers = self.kmeans.cluster_centers_

        self.cluster_centers = t.from_numpy(cluster_centers).type(t.float32)

        self.translated_sigmoid = TranslatedSigmoid(beta=beta)

    def min_distance(self, z: t.Tensor) -> t.Tensor:
        """
        A function that measures the main distance w.r.t
        the cluster centers.
        """
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z ** 2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.cluster_centers ** 2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm + center_norm - 2.0 * t.mm(z, self.cluster_centers.transpose(0, 1))
        )  # Nx(num_clusters)
        d2.clamp_(min=0.0)  # Nx(num_clusters)
        min_dist, _ = d2.min(dim=1)  # N

        return min_dist.view(zsh[:-1])

    def metric(self, z: t.Tensor) -> t.Tensor:
        return approximate_metric(self.decode, z)

    def curve_length(self, curve):
        dt = (curve[:-1] - curve[1:]).pow(2).sum(dim=-1, keepdim=True)  # (N-1)x1
        full_cat = self.decode(curve)
        probs = full_cat.probs

        cat1 = Categorical(probs=probs[:-1])
        cat2 = Categorical(probs=probs[1:])

        inner_term = (1 - (cat1.probs * cat2.probs).sum(dim=-1)).sum(dim=(1, 2))
        energy = (inner_term * dt).sum()
        return energy

    def plot_latent_space(self, ax=None, plot_points=True):
        """
        Plots the latent space, colored by entropy.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))

        n_x, n_y = 100, 100
        x_lims = (-5, 5)
        y_lims = (-5, 5)
        z1 = t.linspace(*x_lims, n_x)
        z2 = t.linspace(*y_lims, n_x)

        entropy_K = np.zeros((n_y, n_x))
        zs = t.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(z1)
            for i, y in enumerate(reversed(z2))
        }

        dist_ = self.decode(zs)
        entropy_ = dist_.entropy().mean(axis=1).mean(axis=1)
        if len(entropy_.shape) > 1:
            entropy_ = t.mean(entropy_, dim=1)

        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            entropy_K[i, j] = entropy_[l]

        if plot_points:
            ax.scatter(
                self.encodings[:, 0],
                self.encodings[:, 1],
                # marker="x",
                # c="k",
            )
        ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])

    def plot_metric_volume(self, ax=None, x_lims=(-5, 5), y_lims=(-5, 5), cmap="Blues"):
        plot_approximation(self, ax=ax, x_lims=x_lims, y_lims=y_lims, cmap=cmap)
