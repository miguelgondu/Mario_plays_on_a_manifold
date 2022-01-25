from geoml.manifold import Manifold

import torch
from torch.distributions import Categorical, Dirichlet, Normal
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from geoml.nnj import TranslatedSigmoid
from vae_hierarchical_text import VAEHierarchicalText, load_data

from geoml.discretized_manifold import DiscretizedManifold
from metric_approximation import MetricApproximation
from metric_approximation_with_jacobians import approximate_metric, plot_approximation

Tensor = torch.Tensor


class VAEGeometryHierarchicalText(VAEHierarchicalText, Manifold):
    """
    Interface for possible extrapolations. This class leaves
    self.reweight without implementation.
    """

    def __init__(
        self,
        length: int = 10,
        z_dim: int = 2,
        device: str = None,
    ):
        super().__init__(length=length, z_dim=z_dim, device=device)
        self.cluster_centers = None
        self.translated_sigmoid = None
        self.encodings = None

    def theoretical_KL(self, p: Categorical, q: Categorical) -> torch.Tensor:
        """
        Returns the theoretical KL between the two Categoricals
        """
        # TODO: change this to take the mean of the whole array. (?)
        return torch.distributions.kl_divergence(p, q).sum(dim=(1))

    def update_cluster_centers(
        self,
        beta: float = -3.0,
        n_clusters: int = 50,
        encodings: Tensor = None,
        cluster_centers: Tensor = None,
    ):
        """
        Updates the cluster centers with the support of the data.
        If only_playable is True, the support becomes only the
        playable levels in the training set.
        """
        if encodings is None:
            latent_codes = self.encode(self.train_tensor).mean
            self.encodings = latent_codes
        else:
            self.encodings = encodings.type(torch.float32)

        if cluster_centers is None:
            self.kmeans = KMeans(n_clusters=n_clusters)
            self.kmeans.fit(self.encodings.cpu().detach().numpy())
            cluster_centers = self.kmeans.cluster_centers_
            self.cluster_centers = torch.from_numpy(cluster_centers).type(torch.float32)
        else:
            self.cluster_centers = cluster_centers.type(torch.float32)

        self.translated_sigmoid = TranslatedSigmoid(beta=beta)

    def min_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        A function that measures the main distance w.r.t
        the cluster centers.
        """
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z ** 2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.cluster_centers ** 2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm
            + center_norm
            - 2.0 * torch.mm(z, self.cluster_centers.transpose(0, 1))
        )  # Nx(num_clusters)
        d2.clamp_(min=0.0)  # Nx(num_clusters)
        min_dist, _ = d2.min(dim=1)  # N

        return min_dist.view(zsh[:-1])

    def reweight(self, z: Tensor) -> Categorical:
        similarity = self.translated_sigmoid(self.min_distance(z)).view(-1, 1)
        intermediate_normal = self._intermediate_distribution(z)
        dec_mu, dec_std = intermediate_normal.mean, intermediate_normal.scale

        reweighted_std = (1 - similarity) * dec_std + similarity * (
            5.0 * torch.ones_like(dec_std)
        )
        reweighted_normal = Normal(dec_mu, reweighted_std)
        samples = reweighted_normal.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.length, self.n_symbols)
        )

        return p_x_given_z

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        return approximate_metric(self.reweight, z, input_size=self.input_dim)

    def curve_length(self, curve):
        dt = (curve[:-1] - curve[1:]).pow(2).sum(dim=-1, keepdim=True)  # (N-1)x1

        full_cat = self.reweight(curve)
        probs = full_cat.probs

        cat1 = Categorical(probs=probs[:-1])
        cat2 = Categorical(probs=probs[1:])

        inner_term = (1 - (cat1.probs * cat2.probs).sum(dim=-1)).sum(dim=1)
        energy = (inner_term * dt).sum()
        return energy

    def plot_latent_space(self, ax=None, plot_points=True):
        """
        Plots the latent space, colored by entropy.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))

        n_x, n_y = 50, 50
        x_lims = (-5, 5)
        y_lims = (-5, 5)
        z1 = torch.linspace(*x_lims, n_x)
        z2 = torch.linspace(*y_lims, n_x)

        entropy_K = np.zeros((n_y, n_x))
        zs = torch.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(z1)
            for i, y in enumerate(reversed(z2))
        }

        dist_ = self.reweight(zs)
        entropy_ = dist_.entropy().mean(axis=1)
        if len(entropy_.shape) > 1:
            entropy_ = torch.mean(entropy_, dim=1)

        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            entropy_K[i, j] = entropy_[l]

        if not (isinstance(ax, list) or isinstance(ax, tuple)):
            ax = [ax]

        for axi in ax:
            if plot_points:
                axi.scatter(
                    self.cluster_centers[:, 0],
                    self.cluster_centers[:, 1],
                    marker="x",
                    c="k",
                )
            axi.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])

    def plot_w_geodesics(self, ax=None, plot_points=True, n_geodesics=20):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        self.plot_latent_space(ax=ax, plot_points=plot_points)

        grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
        Mx, My = torch.meshgrid(grid[0], grid[1])
        grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)

        DM = DiscretizedManifold(self, grid2, use_diagonals=True)
        data = self.encodings
        N = data.shape[0]
        for _ in range(n_geodesics):
            idx = torch.randint(N, (2,))
            try:
                c = DM.connecting_geodesic(data[idx[0]], data[idx[1]])
                c.plot(ax=ax, c="red", linewidth=2.0)
            except Exception as e:
                print(f"Couldn't, got {e}")

    def plot_metric_volume(self, ax=None, x_lims=(-5, 5), y_lims=(-5, 5)):
        plot_approximation(self, ax=ax, x_lims=x_lims, y_lims=y_lims)


if __name__ == "__main__":
    vae = VAEGeometryHierarchicalText()
    vae.load_state_dict(
        torch.load(f"./models/text/hierarchical_vae_text_final.pt", map_location="cpu")
    )
    beta = -3.5
    vae.update_cluster_centers(beta=beta, n_clusters=500)
    _, (ax1, ax2) = plt.subplots(1, 2)
    correctness = vae.get_correctness_img("syntactic")
    ax1.imshow(correctness, extent=[-5, 5, -5, 5])
    vae.plot_latent_space(ax=ax1)
    vae.plot_metric_volume(ax=ax2)
    vae.plot_w_geodesics(ax=ax2, plot_points=False)
    plt.show()