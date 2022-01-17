"""
This script saves the arrays for all geometric experiments for GPC-built geometry,
for several dimensions.

See table [ref] in the research log.
"""
from itertools import product
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical

from interpolations.astar_gpc_interpolation import AStarGPCInterpolation
from diffusions.geometric_difussion import GeometricDifussion

from geometry_metrics_for_different_z_dims import get_random_pairs, models


def save_arrays_for_interpolation(trace_path: Path, gpc_kwargs: dict = {}):
    vae = VAEMarioHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{models[2]}.pt"))

    a = np.load(trace_path)
    zs, p = a["zs"], a["playabilities"]

    gpc = GaussianProcessClassifier(**gpc_kwargs)
    gpc.fit(zs, p)
    astar = AStarGPCInterpolation(10, gpc)

    # Saving interpolations
    # Only interpolate where playability is 1 according to
    # the AL exploration!
    zs_playable = t.from_numpy(zs[p == 1.0])
    zs1, zs2 = get_random_pairs(zs_playable, n_pairs=50)

    _, ax = plt.subplots(1, 1)

    for line_i, (z1, z2) in enumerate(zip(zs1, zs2)):
        line = astar.interpolate(z1, z2)
        ax.scatter(line[:, 0], line[:, 1], c="red")
        levels_in_line = vae.decode(line).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/geometric_gpc/{models[2]}_astar_gpc_interpolation_{line_i:03d}.npz",
            zs=line.detach().numpy(),
            levels=levels_in_line.detach().numpy(),
        )

    ax.imshow(astar.grid, extent=[-5, 5, -5, 5], cmap="Blues")

    plt.show()


def save_arrays_for_diffusion(trace_path: Path, gpc_kwargs: dict = {}):
    vae = VAEGeometryHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{models[2]}.pt"))

    a = np.load(trace_path)
    zs, p = a["zs"], a["playabilities"]

    gpc = GaussianProcessClassifier(**gpc_kwargs)
    gpc.fit(zs, p)

    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])

    res = gpc.predict_proba(bigger_grid)
    decision_boundary = 0.8

    predictions = np.array([0 if p[1] < decision_boundary else 1.0 for p in res])

    vae.update_cluster_centers(
        cluster_centers=t.from_numpy(bigger_grid[predictions == 1.0])
    )

    _, ax = plt.subplots(1, 1)

    gd = GeometricDifussion(10, scale=10.0)
    for run_i in range(50):
        geometric_diffusion = gd.run(vae)

        # This is using the reweighted decode.
        levels_geometric = vae.decode(geometric_diffusion, reweight=False).probs.argmax(
            dim=-1
        )
        np.savez(
            f"./data/arrays/geometric_gpc/{models[2]}_geometric_diffusion_gpc_{run_i:03d}.npz",
            zs=geometric_diffusion.detach().numpy(),
            levels=levels_geometric.detach().numpy(),
        )
        ax.scatter(
            geometric_diffusion.detach().numpy()[:, 0],
            geometric_diffusion.detach().numpy()[:, 1],
            c="red",
        )

    vae.plot_latent_space(ax=ax, plot_points=False)
    plt.show()


if __name__ == "__main__":
    trace_path = "./data/evolution_traces/bigger_trace.npz"
    save_arrays_for_interpolation(trace_path)
    save_arrays_for_diffusion(trace_path)
