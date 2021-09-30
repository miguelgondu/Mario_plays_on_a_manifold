import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry import VAEGeometry
from astar_interpolation import AStarInterpolation
from geodesic_interpolation import GeodesicInterpolation
from linear_interpolation import LinearInterpolation
from staying_within_playable_levels import (
    get_playable_points,
    plot_grid_reweight,
    local_KL,
    plot_column,
)
from simulator import test_level_from_z

from geoml.discretized_manifold import DiscretizedManifold


model_name = "mariovae_z_dim_2_overfitting_epoch_480"

playable_points = get_playable_points(model_name, full_playable=True)
playable_points = torch.from_numpy(playable_points)

vae = VAEGeometry()
vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
print("Updating cluster centers")
beta = -5.2
vae.update_cluster_centers(
    model_name,
    False,
    beta=beta,
    n_clusters=playable_points.shape[0],
    encodings=playable_points,
    cluster_centers=playable_points,
)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 10))

x_lims = (-6, 6)
y_lims = (-6, 6)

# Load the table and process it
path = (
    f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv"
)
df = pd.read_csv(path, index_col=0)
plot_column(df, "marioStatus", ax=ax1)

# Defining the interpolations.
li = LinearInterpolation()
grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
Mx, My = torch.meshgrid(grid[0], grid[1])
grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)
DM = DiscretizedManifold(vae, grid2, use_diagonals=True)
gi = GeodesicInterpolation(DM, model_name)
asi = AStarInterpolation(10, model_name)

z1s = [torch.Tensor([5.0, -3.0]), torch.Tensor([-4.0, -5.0])]
z2s = [torch.Tensor([5.0, 3.0]), torch.Tensor([-1.0, 5.0])]

for i, z1, z2 in zip(range(len(z1s)), z1s, z2s):
    linear = li.interpolate(z1, z2).detach().numpy()
    geodesic = gi.interpolate(z1, z2).detach().numpy()
    astar = asi.interpolate(z1, z2).detach().numpy()

    res_linear = [
        test_level_from_z(torch.from_numpy(z).type(torch.float), vae)["marioStatus"]
        for z in linear
    ]
    res_geodesic = [
        test_level_from_z(torch.from_numpy(z).type(torch.float), vae)["marioStatus"]
        for z in geodesic
    ]
    res_astar = [
        test_level_from_z(torch.from_numpy(z).type(torch.float), vae)["marioStatus"]
        for z in astar
    ]
    # res_linear = "k"
    # res_geodesic = "k"
    # res_astar = "k"

    ax1.plot(
        linear[:, 0], linear[:, 1], color="r", linewidth=4, label="linear", zorder=1
    )
    ax1.plot(
        geodesic[:, 0],
        geodesic[:, 1],
        color="g",
        linewidth=4,
        label="geodesic",
        zorder=1,
    )
    ax1.plot(astar[:, 0], astar[:, 1], color="c", linewidth=4, label="astar", zorder=1)

    ax1.scatter(
        linear[:, 0],
        linear[:, 1],
        c=res_linear,
        cmap="Blues",
        zorder=2,
        vmin=0.0,
        vmax=1.0,
        edgecolor="k",
    )
    ax1.scatter(
        geodesic[:, 0],
        geodesic[:, 1],
        c=res_geodesic,
        cmap="Blues",
        zorder=2,
        vmin=0.0,
        vmax=1.0,
        edgecolor="k",
    )
    ax1.scatter(
        astar[:, 0],
        astar[:, 1],
        c=res_astar,
        cmap="Blues",
        zorder=2,
        vmin=0.0,
        vmax=1.0,
        edgecolor="k",
    )

    if i == 0:
        ax1.legend()

print("Plotting Local KL approximation")
n_x, n_y = 50, 50
z1 = torch.linspace(*x_lims, n_x)
z2 = torch.linspace(*y_lims, n_x)

KL_image = np.zeros((n_y, n_x))
zs = torch.Tensor([[x, y] for x in z1 for y in z2])
positions = {
    (x.item(), y.item()): (i, j)
    for j, x in enumerate(z1)
    for i, y in enumerate(reversed(z2))
}

KLs = local_KL(vae, zs, eps=0.05)
for l, (x, y) in enumerate(zs):
    i, j = positions[(x.item(), y.item())]
    KL_image[i, j] = KLs[l]

# ax2.scatter(vae.cluster_centers[:, 0], vae.cluster_centers[:, 1], marker="x", c="k")
plot = ax2.imshow(KL_image, extent=[*x_lims, *y_lims], cmap="viridis")
plt.colorbar(plot, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()

plt.savefig(
    f"./data/plots/comparison_interpolations_beta_{str(beta).replace('.', '_')}.png"
)
plt.show()
