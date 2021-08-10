"""
I need to circle back to experiment design,
because there are 2600 ish training levels,
and it turns out that doing a gridsearch
with 50x50 in the latent space is not that expensive
(2500 levels). Is this a sensible thing? If I did the gridsearch,
why not use it directly?
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from vae_geometry import VAEGeometry

# Types
Tensor = torch.Tensor


# Getting the playable levels
# Table was created in analyse_solvability_experiment.py
def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}.csv", index_col=0
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


def get_non_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}.csv", index_col=0
    )
    non_playable_points = df.loc[df["marioStatus"] == 0, ["z1", "z2"]]
    non_playable_points.drop_duplicates(inplace=True)
    non_playable_points = non_playable_points.values

    return non_playable_points


if __name__ == "__main__":
    model_name = "mariovae_z_dim_2_overfitting_epoch_480_playability_experiment"
    playable_points = get_playable_points(model_name)
    non_playable_points = get_non_playable_points(model_name)

    X = np.vstack((playable_points, non_playable_points))
    y = np.concatenate(
        (
            np.ones((playable_points.shape[0],)),
            np.zeros((non_playable_points.shape[0],)),
        )
    )

    x_lims = y_lims = [-6, 6]

    k_means = KMeans(n_clusters=50)
    k_means.fit(playable_points)

    kernel = 1.0 * RBF(length_scale=[1.0, 1.0])
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y)

    n_x, n_y = 50, 50
    z1 = torch.linspace(*x_lims, n_x)
    z2 = torch.linspace(*y_lims, n_x)

    class_image = np.zeros((n_y, n_x))
    zs = np.array([[x, y] for x in z1 for y in z2])
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    classes = gpc.predict(zs)
    for l, (x, y) in enumerate(zs):
        i, j = positions[(x.item(), y.item())]
        class_image[i, j] = classes[l]

    _, ax = plt.subplots(1, 1)
    ax.imshow(class_image, extent=[*x_lims, *y_lims], cmap="Blues")
    ax.scatter(playable_points[:, 0], playable_points[:, 1], marker="o", c="y")
    ax.scatter(non_playable_points[:, 0], non_playable_points[:, 1], marker="o", c="r")
    plt.show()
