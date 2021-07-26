from typing import List
from pathlib import Path

import torch
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from geoml.nnj import TranslatedSigmoid
from vae_mario import VAEMario
from train_vae import load_data
from geoml.discretized_manifold import DiscretizedManifold

Tensor = torch.Tensor


class VAEGeometry(VAEMario):
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
        return torch.distributions.kl_divergence(p, q).mean(dim=0)

    def update_cluster_centers(
        self, model_name: str, only_playable: bool, beta: float = -3.0
    ):
        """
        Updates the cluster centers with the parts of latent space that are actually playable.

        i.e. I'll have to load up the playability experiment
        data and take it from there. Define the cluster
        centers myself.
        """
        # playable_points = np.load(f"data/processed/playable_clusters_{model_name}.npz")[
        #     "clusters"
        # ]
        # self.kmeans = KMeans(n_clusters=50)
        # self.kmeans.fit(playable_points)
        # print(cluster_centers)
        theta = 2 * np.pi * np.random.rand(50)
        x, y = np.cos(theta), np.sin(theta)
        cluster_centers = 3 * np.vstack((x, y)).T
        # cluster_centers = np.array(
        #     [[k, -0.5] for k in np.linspace(-6, 1, 50)]
        # )
        self.cluster_centers = torch.from_numpy(cluster_centers).type(torch.float32)
        self.translated_sigmoid = TranslatedSigmoid(beta=beta)
        training_tensors, _ = load_data(only_playable=only_playable)
        self.encodings = torch.from_numpy(cluster_centers).type(torch.float)

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

    def reweight(self, z: Tensor) -> List[Tensor]:
        dec_x = self.decode(z)
        batch_size, n_classes, h, w = dec_x.shape
        dec_x = dec_x.view(batch_size, h * w * n_classes)
        # I think we're expecting this to be flat?

        # similarity = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
        similarity = 1 - self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)

        res = (1 - similarity) * dec_x + similarity * (
            (1 / self.n_sprites) * torch.ones_like(dec_x)
        )

        # Putting the classes in the last one.
        res = res.view(batch_size, n_classes, h, w)
        # |res| = batch_size, n_classes, h, w
        # res: Tensor
        # print(res.shape)
        # res = res.permute(0, 2, 3, 1)
        # res = torch.transpose(res, 1, 2)
        # print(res.shape)
        # res = torch.transpose(res, 2, 3)
        # print(res.shape)
        # raise

        return [res]

    def curve_length(self, curve):
        """
        FOR DIMITRIS:

        This is a function that computes the KL
        divergence along curves. You'll have to:
        1. modify `alpha, beta = self.reweight(curve)`
           to the appropiate parameters,
        2. create the instances of the distributions,
           just like `beta1` and `beta2` in this case,
        3. either compute the KL with a theoretical expression
           that you implement, or by sampling.
        """
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

        return 10.0 * (kl.sqrt() * dt).sum()

    def plot_latent_space(self, ax=None):
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
        print(entropy_)
        print(entropy_.shape)
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
        ax.scatter(
            self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker="x", c="k"
        )
        plot = ax.imshow(entropy_K, extent=[*x_lims, *y_lims], cmap="Blues")
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_yticklabels([entropy_K.min(), entropy_K.max()])
        # ax.set_title(self.fig_title)
        # raise NotImplementedError

    def plot_w_geodesics(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        self.plot_latent_space(ax=ax)
        # This plots the geodesics.
        grid = [torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100)]
        Mx, My = torch.meshgrid(grid[0], grid[1])
        grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)

        # -------------------------------------------
        #     CHANGE THIS DEPENDING ON THE DIST.
        #           beta_params -> your_dist_params.
        DM = DiscretizedManifold(self, grid2, use_diagonals=True)
        # -------------------------------------------
        data = self.encodings
        N = data.shape[0]
        for _ in range(20):
            idx = torch.randint(N, (2,))
            c = DM.connecting_geodesic(data[idx[0]], data[idx[1]])
            c.plot(ax=ax, c="g", linewidth=0.5)
