"""
This script imports the Linear, AStar and Geodesic
interpolations and computes for a given set of random
pairs in latent space. These pairs are selected from
the training codes of the same VAE.
"""
import numpy as np
import pandas as pd
import torch

from vae_geometry import VAEGeometry
from linear_interpolation import LinearInterpolation


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
    np.random.seed(17)
    n_lines = 50

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

    training_codes = vae.encodings.detach().numpy()
    print(training_codes)
    print(training_codes.shape)
    z1_idxs = np.random.choice(len(training_codes), size=n_lines, replace=False)
    z2_idxs = np.random.choice(len(training_codes), size=n_lines, replace=False)
    z1s = [training_codes[idx, :] for idx in z1_idxs]
    z2s = [training_codes[idx, :] for idx in z2_idxs]

    li = LinearInterpolation()
    for z1, z2 in zip(z1s, z2s):
        print(f"line between {z1} and {z2}: {li.interpolate(z1, z2)}")
