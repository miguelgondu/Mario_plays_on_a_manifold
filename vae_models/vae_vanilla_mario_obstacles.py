"""
This is a VAE manifold that uses the usual
K-means on the support to extrapolate.
"""
from itertools import product

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal
from vae_models.vae_vanilla_mario import VAEMario

from stochman.nnj import TranslatedSigmoid
from stochman.manifold import Manifold

from utils.metric_approximation.finite_difference import (
    approximate_metric,
    plot_approximation,
)

from utils.mario.plotting import get_img_from_level


class VAEVanillaMarioObstacles(VAEMario, Manifold):
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
        self.obstacles = []

    # This method overwrites the decode of the vanilla one.
    def decode(self, z: t.Tensor, reweight: bool = True) -> Categorical:
        if reweight and len(self.obstacles) > 0:
            dist_to_obst = self.translated_sigmoid(self.min_distance(z)).view(
                -1, 1, 1, 1
            )
            original_categorical = super().decode(z)

            original_probs = original_categorical.probs

            reweighted_probs = (dist_to_obst) * original_probs + (1 - dist_to_obst) * (
                (1 / self.n_sprites) * t.ones_like(original_probs)
            )
            p_x_given_z = Categorical(probs=reweighted_probs)
        else:
            p_x_given_z = super().decode(z)

        return p_x_given_z

    def update_obstacles(
        self,
        obstacles: t.Tensor,
        beta: float = -3.0,
    ):
        """
        Updates the points to avoid.
        """
        self.obstacles = obstacles.to(self.device)
        self.translated_sigmoid = TranslatedSigmoid(beta=beta)

    def min_distance(self, z: t.Tensor) -> t.Tensor:
        """
        A function that measures the main distance w.r.t
        the cluster centers.
        """
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z ** 2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.obstacles ** 2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm + center_norm - 2.0 * t.mm(z, self.obstacles.transpose(0, 1))
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
                self.obstacles[:, 0],
                self.obstacles[:, 1],
                marker="x",
                c="k",
            )
        ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])

    def plot_w_geodesics(self, ax=None, plot_points=True, n_geodesics=10):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        self.plot_latent_space(ax=ax, plot_points=plot_points)

        data = t.Tensor(
            [
                [-4.0, -4.0],
                [-4.0, 4.0],
                [3.0, -4.0],
                # [4.0, 4.0],
                [0.0, -4.0],
                [0.0, 0.0],
                [2.0, 4.0],
                [-2.0, 4.0],
                [3.5, 3.5],
            ]
        )
        N = data.shape[0]
        for _ in range(n_geodesics):
            idx = t.randint(N, (2,))
            try:
                c, _ = self.connecting_geodesic(data[idx[0]], data[idx[1]])
                c.plot(ax=ax, c="red", linewidth=2.0)
            except Exception as e:
                print(f"Couldn't, got {e}")

    def plot_metric_volume(self, ax=None, x_lims=(-5, 5), y_lims=(-5, 5), cmap="Blues"):
        plot_approximation(self, ax=ax, x_lims=x_lims, y_lims=y_lims, cmap=cmap)

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
        return_imgs=False,
        return_probs=False,
    ):
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(t.from_numpy(zs).type(t.float))
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = 16 * 14
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[
                i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
            ] = img_dict[z]

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        if return_probs:
            return final_img, images_dist.probs

        if return_imgs:
            return final_img, images

        return final_img
