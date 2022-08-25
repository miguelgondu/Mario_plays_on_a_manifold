import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal
from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

from sklearn.cluster import KMeans
from geoml.nnj import TranslatedSigmoid
from geoml.manifold import Manifold
from vae_models.vae_mario_hierarchical import load_data

from geoml.discretized_manifold import DiscretizedManifold
from metric_approximation_with_jacobians import approximate_metric, plot_approximation


Tensor = torch.Tensor


class VAEGeometryHierarchical(VAEMarioHierarchical, Manifold):
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
    def decode(self, z: Tensor, reweight: bool = True) -> Categorical:
        if reweight:
            similarity = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
            intermediate_normal = self._intermediate_distribution(z)
            dec_mu, dec_std = intermediate_normal.mean, intermediate_normal.scale
            reweighted_std = (1 - similarity) * dec_std + similarity * (
                10.0 * torch.ones_like(dec_std)
            )
            reweighted_normal = Normal(dec_mu, reweighted_std)
            samples = reweighted_normal.rsample()
            p_x_given_z = Categorical(
                logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
            )
        else:
            p_x_given_z = super().decode(z)

        return p_x_given_z

    def theoretical_KL(self, p: Categorical, q: Categorical) -> torch.Tensor:
        """
        Returns the theoretical KL between the two Categoricals
        """
        # TODO: change this to take the mean of the whole array. (?)
        return torch.distributions.kl_divergence(p, q).sum(dim=(1, 2))

    def update_cluster_centers(
        self,
        only_playable: bool = False,
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
            training_tensors, _ = load_data(only_playable=only_playable)
            latent_codes = self.encode(training_tensors).mean
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
            if encodings is None:
                self.encodings = cluster_centers.type(torch.float32)

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

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        return approximate_metric(self.decode, z)
        # return self.metric_approximation(z)

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
        z1 = torch.linspace(*x_lims, n_x)
        z2 = torch.linspace(*y_lims, n_x)

        entropy_K = np.zeros((n_y, n_x))
        zs = torch.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(z1)
            for i, y in enumerate(reversed(z2))
        }

        dist_ = self.decode(zs)
        entropy_ = dist_.entropy().mean(axis=1).mean(axis=1)
        if len(entropy_.shape) > 1:
            entropy_ = torch.mean(entropy_, dim=1)

        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            entropy_K[i, j] = entropy_[l]

        if plot_points:
            ax.scatter(
                self.cluster_centers[:, 0],
                self.cluster_centers[:, 1],
                marker="x",
                c="k",
            )
        ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])

    def plot_w_geodesics(self, ax=None, plot_points=True, n_geodesics=5):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        self.plot_latent_space(ax=ax, plot_points=plot_points)

        data = self.encodings
        N = data.shape[0]
        for _ in range(n_geodesics):
            idx = torch.randint(N, (2,))
            try:
                c, _ = self.connecting_geodesic(data[idx[0]], data[idx[1]])
                c.plot(ax=ax, c="red", linewidth=2.0)
            except Exception as e:
                print(f"Couldn't, got {e}")

    def plot_metric_volume(self, ax=None, x_lims=(-5, 5), y_lims=(-5, 5)):
        plot_approximation(self, ax=ax, x_lims=x_lims, y_lims=y_lims)


if __name__ == "__main__":
    model_name = "hierarchical_final_playable_final"
    vae = VAEGeometryHierarchical()
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt", map_location="cpu"))
    beta = -3.5
    vae.update_cluster_centers(beta=beta, n_clusters=500)
    # _, (ax1, ax2) = plt.subplots(1, 2)
    # vae.plot_latent_space(ax=ax1)
    # vae.plot_w_geodesics(ax=ax2, plot_points=False)
    # plt.show()
    # """
    # Returns some plots for the circle dataset,
    # plotting geodescis and approximating metrics.
    # """
    # vae = Model()
    # vae.load_state_dict(t.load(f"models/{model_name}.pt", map_location="cpu"))
    # print("Updating cluster centers")
    # print(encodings)

    # angles = torch.rand((100,)) * 2 * np.pi
    # encodings = 3.0 * torch.vstack((torch.cos(angles), torch.sin(angles))).T
    # vae.update_cluster_centers(beta=-2.5, encodings=encodings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 10))
    x_lims = (-5, 5)
    y_lims = (-5, 5)

    print("Plotting geodesics and latent space")
    try:
        vae.plot_w_geodesics(ax=ax1, plot_points=True)
    except Exception as e:
        print(f"couldn't get geodesics for reason {e}")

    n_x, n_y = 50, 50
    x_lims = (-5, 5)
    y_lims = (-5, 5)
    z1 = torch.linspace(*x_lims, n_x)
    z2 = torch.linspace(*y_lims, n_x)
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    zs = torch.Tensor([[x, y] for x in z1 for y in z2])
    metric_volume = np.zeros((n_y, n_x))
    for z in zs:
        (x, y) = z
        i, j = positions[(x.item(), y.item())]
        Mz = vae.metric(z)

        detMz = torch.det(Mz).item()
        if detMz < 0:
            metric_volume[i, j] = np.nan
        else:
            metric_volume[i, j] = np.log(detMz)

    cbar = ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")
    plt.colorbar(cbar)

    ax1.set_title("Latent space and geodesics")
    ax2.set_title("Estimated metric volume")
    fig.suptitle(f"beta = {beta}")
    plt.tight_layout()
    plt.show()
