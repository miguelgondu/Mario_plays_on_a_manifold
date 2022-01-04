"""
This script saves the arrays for all baseline experiments,
for several dimensions.

See table [ref] in the research log.
"""
from typing import List
from itertools import product

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical
from train_vae import load_data

from interpolations.geodesic_interpolation import GeodesicInterpolation

from diffusions.geometric_difussion import GeometricDifussion

# I need to train the hierarchical models first.
# Em, no. These are the final ones...
models = {
    2: "16388917374131331_mariovae_zdim_2_normal_final",
    8: "1638894528256156_mariovae_zdim_8_normal_final",
    32: "16388927503019269_mariovae_zdim_32_normal_final",
    64: "16388929591033669_mariovae_zdim_64_normal_final",
}


def get_random_pairs(
    points: t.Tensor, n_pairs: int = 20, seed: int = None
) -> List[t.Tensor]:
    if seed is not None:
        np.random.seed(seed)
    idx1 = np.random.choice(len(points), size=n_pairs, replace=False)
    idx2 = np.random.choice(len(points), size=n_pairs, replace=False)
    while np.any(idx1 == idx2):
        idx2 = np.random.choice(len(points), size=n_pairs, replace=False)

    pairs_1 = points[idx1]
    pairs_2 = points[idx2]

    return pairs_1, pairs_2


def save_arrays_for_model(model_name: str, z_dim: int, beta: float = -3.0) -> None:
    vae = VAEGeometryHierarchical(z_dim=z_dim)
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))
    vae.update_cluster_centers(beta=beta, n_clusters=500)

    train_, test_ = load_data(only_playable=True)
    levels = t.cat((train_, test_))
    encodings = vae.encode(levels).mean

    gi = GeodesicInterpolation(vae)
    zs_1, zs_2 = get_random_pairs(encodings, n_pairs=50)
    for line_i, (z1, z2) in enumerate(zip(zs_1, zs_2)):
        line = gi.interpolate(z1, z2)
        levels_in_line = vae.decode(line).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/geometric/{model_name}_beta_{int(beta*10)}_geodesic_interpolation_{line_i:03d}.npz",
            zs=line.detach().numpy(),
            levels=levels_in_line.detach().numpy(),
        )

    # Saving diffusions
    if z_dim == 2:
        gd = GeometricDifussion(10, scale=2.0)
    else:
        gd = GeometricDifussion(10, scale=1.0)

    for run_i in range(50):
        geometric_diffusion = gd.run(vae)
        levels_geometric = vae.decode(geometric_diffusion).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/geometric/{model_name}_beta_{int(beta*10)}_geometric_diffusion_{run_i:03d}.npz",
            zs=geometric_diffusion.detach().numpy(),
            levels=levels_geometric.detach().numpy(),
        )

    print(f"Arrays saved for model {model_name} (beta {beta})")


def inspect_z_dim_2():
    # Visual inspection for 2 dimensions
    model_2 = models[2]
    _, ax = plt.subplots(1, 1)

    vae = VAEMarioHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{model_2}.pt"))

    train_, test_ = load_data(only_playable=True)
    levels = t.cat((train_, test_))
    encodings = vae.encode(levels).mean.detach().numpy()

    ax.scatter(encodings[:, 0], encodings[:, 1], alpha=0.2)

    # Loading up the arrays:
    for line_i in range(50):
        line = np.load(
            f"./data/arrays/geometric/{model_2}_beta_-30_geodesic_interpolation_{line_i:03d}.npz"
        )
        zs = line["zs"]
        ax.scatter(zs[:, 0], zs[:, 1])

    # New fig. for the diffusions.
    _, ax2 = plt.subplots(1, 1)
    ax2.scatter(encodings[:, 0], encodings[:, 1], alpha=0.2)
    for run_i in range(50):
        geometric_d = np.load(
            f"./data/arrays/geometric/{model_2}_beta_-30_geometric_diffusion_{run_i:03d}.npz"
        )
        zs_g = geometric_d["zs"]

        ax2.scatter(zs_g[:, 0], zs_g[:, 1], c="red", alpha=0.5)

    plt.show()
    plt.close()


if __name__ == "__main__":
    # for (z_dim, model_name), beta in product(models.items(), [-2.0, -3.0]):
    #     save_arrays_for_model(model_name, z_dim, beta=beta)

    inspect_z_dim_2()
