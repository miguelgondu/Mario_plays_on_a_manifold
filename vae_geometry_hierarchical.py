from typing import List
from pathlib import Path

import torch
from torch.distributions import Categorical, Dirichlet
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from geoml.nnj import TranslatedSigmoid
from vae_mario_hierarchical import VAEMarioHierarchical
from train_vae import load_data
from geoml.discretized_manifold import DiscretizedManifold

Tensor = torch.Tensor


class VAEGeometryHierarchical(VAEMarioHierarchical):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        h_dims: List[int] = None,
    ):
        super().__init__(w, h, z_dim, n_sprites=n_sprites, h_dims=h_dims)

        self.distribution = Categorical
        self.cluster_centers = None
        self.translated_sigmoid = None
        self.encodings = None

    def theoretical_KL(self, p: Categorical, q: Categorical) -> torch.Tensor:
        # TODO: change this to take the mean of the whole array.
        return torch.distributions.kl_divergence(p, q).mean(dim=0)

    def update_cluster_centers(
        self,
        model_name: str,
        only_playable: bool,
        beta: float = -3.0,
        n_clusters: int = 50,
        encodings: Tensor = None,
        cluster_centers: Tensor = None,
    ):
        """
        Updates the cluster centers with the parts of latent space that are actually playable.

        i.e. I'll have to load up the playability experiment
        data and take it from there. Define the cluster
        centers myself.
        """
        if encodings is None:
            training_tensors, _ = load_data(only_playable=only_playable)
            latent_codes = self.encode(training_tensors)[0]
            self.encodings = latent_codes
        else:
            self.encodings = encodings.type(torch.float32)

        if cluster_centers is None:
            self.kmeans = KMeans(n_clusters=n_clusters)
            self.kmeans.fit(self.encodings.detach().numpy())
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

    def reweight(self, z: Tensor, return_logsigma=False) -> List[Tensor]:
        similarity = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
        dec_mu, dec_logsigma = self._decode_sample(z)

        dec_std = torch.exp(0.5 * dec_logsigma)

        reweighted_std = (1 - similarity) * dec_std + similarity * (
            10.0 * torch.ones_like(dec_std)
        )
        reweighted_logsigma = 2.0 * torch.log(reweighted_std)

        res = torch.nn.LogSoftmax(dim=1)(
            self.reparametrize(dec_mu, reweighted_logsigma).view(
                -1, self.n_sprites, self.h, self.w
            )
        )

        if return_logsigma:
            return [res, reweighted_logsigma]

        return [res]

    def curve_length(self, curve):
        dt = (curve[:-1] - curve[1:]).pow(2).sum(dim=-1, keepdim=True)  # (N-1)x1

        # -------------------------------------------
        #     CHANGE THIS DEPENDING ON THE DIST.
        log_probs = self.reweight(curve)[0]
        log_probs.transpose(1, 3)
        c_size, n_classes, h, w = log_probs.shape
        log_probs = log_probs.view(c_size, n_classes, h * w)

        cat1 = Categorical(logits=log_probs[:-1])
        cat2 = Categorical(logits=log_probs[1:])

        # If there's a theoretical KL that's easy to implement:
        try:
            kl = self.theoretical_KL(cat1, cat2).abs()
            # print(kl)
            # print(kl.shape)
        except NotImplementedError:
            # Otherwise, we can do it by sampling (but it's numerically unstable
            # and takes foreeeeeever)
            print("Defaulting to KL-by-sampling, this might take a while")
            kl = self.KL_by_sampling(cat1, cat2, n_samples=10000).abs()
        # -------------------------------------------

        return (kl.sqrt() * dt).sum()

    def plot_latent_space(self, ax=None, plot_points=True):
        """
        FOR DIMITRIS:

        To plot the mean entropy, you'll need to
        change the output of self.reweight to
        the distribution's parameters. I commented
        the relevant piece of code.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))

        encodings = self.encodings.detach().numpy()
        enc_x, enc_y = encodings[:, 0], encodings[:, 1]

        n_x, n_y = 100, 100
        x_lims = (-6, 6)
        y_lims = (-6, 6)
        z1 = torch.linspace(*x_lims, n_x)
        z2 = torch.linspace(*y_lims, n_x)

        entropy_K = np.zeros((n_y, n_x))
        zs = torch.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(z1)
            for i, y in enumerate(reversed(z2))
        }

        dist_ = Categorical(logits=self.reweight(zs)[0])
        entropy_ = dist_.entropy().mean(axis=1).mean(axis=1)
        # print(entropy_)
        # print(entropy_.shape)
        if len(entropy_.shape) > 1:
            # In some distributions, we decode
            # to a higher dimensional space.
            entropy_ = torch.mean(entropy_, dim=1)

        for l, (x, y) in enumerate(zs):
            i, j = positions[(x.item(), y.item())]
            entropy_K[i, j] = entropy_[l]

        # encodings = self.encodings.detach().numpy()
        # ax.scatter(
        #     encodings[:, 0],
        #     encodings[:, 1],
        #     marker="o",
        #     c="w",
        #     edgecolors="k",
        # )
        if plot_points:
            ax.scatter(
                self.cluster_centers[:, 0],
                self.cluster_centers[:, 1],
                marker="x",
                c="k",
            )
        plot = ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])
        # ax.set_title(self.fig_title)
        # raise NotImplementedError

    def plot_w_geodesics(self, ax=None, plot_points=True, n_geodesics=20):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        self.plot_latent_space(ax=ax, plot_points=plot_points)
        # This plots the geodesics.
        grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
        Mx, My = torch.meshgrid(grid[0], grid[1])
        grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)

        DM = DiscretizedManifold(self, grid2, use_diagonals=True)
        data = self.encodings
        N = data.shape[0]
        for _ in range(n_geodesics):
            idx = torch.randint(N, (2,))
            c = DM.connecting_geodesic(data[idx[0]], data[idx[1]])
            c.plot(ax=ax, c="#FADADD", linewidth=2.5)
