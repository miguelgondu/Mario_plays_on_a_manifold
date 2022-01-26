import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal
from vae_mario_hierarchical import VAEMarioHierarchical

from geoml.nnj import TranslatedSigmoid
from geoml.manifold import Manifold

from metric_approximation_with_jacobians import approximate_metric, plot_approximation


class VAEWithObstacles(VAEMarioHierarchical, Manifold):
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

    # This method overwrites the decode of the vanilla one.
    def decode(self, z: t.Tensor, reweight: bool = True) -> Categorical:
        if reweight:
            dist_to_obst = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
            intermediate_normal = self._intermediate_distribution(z)

            dec_mu, dec_std = intermediate_normal.mean, intermediate_normal.scale
            reweighted_std = (dist_to_obst) * dec_std + (1 - dist_to_obst) * (
                10.0 * t.ones_like(dec_std)
            )
            reweighted_normal = Normal(dec_mu, reweighted_std)
            samples = reweighted_normal.rsample()

            p_x_given_z = Categorical(
                logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
            )
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
        self.obstacles = obstacles
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

    def curve_energy(self, curve):
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

    def plot_metric_volume(self, ax=None, x_lims=(-5, 5), y_lims=(-5, 5)):
        plot_approximation(self, ax=ax, x_lims=x_lims, y_lims=y_lims)
