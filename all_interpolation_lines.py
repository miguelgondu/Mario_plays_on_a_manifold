"""
This script imports the Linear, AStar and Geodesic
interpolations and computes for a given set of random
pairs in latent space. These pairs are selected from
the training codes of the same VAE.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from vae_geometry import VAEGeometry
from geoml.discretized_manifold import DiscretizedManifold

from linear_interpolation import LinearInterpolation
from geodesic_interpolation import GeodesicInterpolation


def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


if __name__ == "__main__":
    n_lines = 10
    n_points_in_line = 10

    # Getting the training codes
    model_name = "mariovae_z_dim_2_overfitting_epoch_480"

    playable_points = get_playable_points(model_name)
    playable_points = torch.from_numpy(playable_points)
    vae = VAEGeometry()
    vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    vae.update_cluster_centers(
        model_name,
        False,
        beta=-4.5,
        n_clusters=playable_points.shape[0],
        encodings=playable_points,
        cluster_centers=playable_points,
    )
    training_codes = vae.encodings

    # Selecting random lines from them.
    np.random.seed(17)
    print(training_codes)
    print(training_codes.shape)
    z1_idxs = np.random.choice(len(training_codes), size=n_lines, replace=False)
    z2_idxs = np.random.choice(len(training_codes), size=n_lines, replace=False)
    z1s = [training_codes[idx, :] for idx in z1_idxs]
    z2s = [training_codes[idx, :] for idx in z2_idxs]

    # Linear interpolation
    li = LinearInterpolation(n_points=n_points_in_line)

    # Geodesic interpolation
    grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
    Mx, My = torch.meshgrid(grid[0], grid[1])
    grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)
    DM = DiscretizedManifold(vae, grid2, use_diagonals=True)

    gi = GeodesicInterpolation(
        DM,
        model_name,
        n_points=n_points_in_line,
    )

    _, ax = plt.subplots(1, 1)
    for z1, z2 in zip(z1s, z2s):
        line_li = li.interpolate(z1, z2).detach().numpy()
        line_gi = gi.interpolate(z1, z2).detach().numpy()
        ax.scatter(line_li[1:-1, 0], line_li[1:-1, 1], c="r")
        ax.scatter(line_gi[1:-1, 0], line_gi[1:-1, 1], c="g")
        # print(f"line between {z1} and {z2}: {li.interpolate(z1, z2)}")
        # print(f"geodesic between {z1} and {z2}: {gi.interpolate(z1, z2)}")

    plt.show()
    plt.close()
