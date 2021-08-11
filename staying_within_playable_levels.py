"""
I need to circle back to experiment design,
because there are 2600 ish training levels,
and it turns out that doing a gridsearch
with 50x50 in the latent space is not that expensive
(2500 levels). Is this a sensible thing? If I did the gridsearch,
why not use it directly?
"""
import json
from pathlib import Path
from sklearn import cluster

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from train_vae import load_data
from mario_utils.levels import tensor_to_sim_level
from approximate_metric import local_KL, plot_grid_reweight
from vae_geometry import VAEGeometry

# Types
Tensor = torch.Tensor


def plot_column(df, column_name, x_lims=None, y_lims=None, ax=None):
    col = df.groupby(["z1", "z2"]).mean()[column_name]
    # print(col)
    # print(col.index)
    zs = np.array([z for z in col.index])
    # print(zs)
    z1 = sorted(list(set(zs[:, 0])))
    z2 = sorted(list(set(zs[:, 1])), reverse=True)

    n_x = len(z1)
    n_y = len(z2)

    M = np.zeros((n_y, n_x))
    col: pd.Series
    for (z1i, z2i), m in col.iteritems():
        i, j = z2.index(z2i), z1.index(z1i)
        M[i, j] = m

    # print(M)
    # print(zs.shape)
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if x_lims is None:
        x_lims = [min(z1), max(z1)]

    if y_lims is None:
        y_lims = [min(z2), max(z2)]

    plot = ax.imshow(M, extent=[*x_lims, *y_lims], cmap="Blues")
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)

    if ax is None:
        plt.show()
        plt.close()


# Getting the playable levels
# Table was created in analyse_solvability_experiment.py
def create_table_training_levels():
    all_results = Path("./data/testing_training_levels").glob("*.json")
    training_tensors, test_tensors = load_data()
    all_levels = torch.cat((training_tensors, test_tensors))

    rows = []
    for result in all_results:
        with open(result) as fp:
            data = json.load(fp)

        idx = int(result.name.split("_")[1])
        iteration = int(result.name.split("_")[2].replace(".json", ""))
        level = all_levels[idx].view(-1, 11, 14, 14).detach().numpy()

        row = {"idx": idx, "iteration": iteration, "level": level, **data}

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("./data/processed/training_levels_results.csv")

    return df


def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


def get_non_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    non_playable_points = df.loc[df["marioStatus"] == 0, ["z1", "z2"]]
    non_playable_points.drop_duplicates(inplace=True)
    non_playable_points = non_playable_points.values

    return non_playable_points


def geodesics_in_grid(model_name):
    model_name = "mariovae_z_dim_2_overfitting_epoch_480"

    playable_points = get_playable_points(model_name)
    playable_points = torch.from_numpy(playable_points)

    vae = VAEGeometry()
    vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    print("Updating cluster centers")
    vae.update_cluster_centers(
        model_name,
        False,
        beta=-4.5,
        n_clusters=playable_points.shape[0],
        encodings=playable_points,
        cluster_centers=playable_points,
    )

    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7 * 4, 7))

    print("Plotting grid of levels")
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    plot_grid_reweight(vae, ax1, x_lims, y_lims, n_rows=12, n_cols=12)

    print("Plotting geodesics and latent space")
    vae.plot_w_geodesics(ax=ax2, plot_points=False)

    # Load the table and process it
    path = f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv"
    print("Printing simulation results")
    df = pd.read_csv(path, index_col=0)
    print(df)
    plot_column(df, "marioStatus", ax=ax3)

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

    # ax4.scatter(vae.cluster_centers[:, 0], vae.cluster_centers[:, 1], marker="x", c="k")
    plot = ax4.imshow(KL_image, extent=[*x_lims, *y_lims], cmap="viridis")
    plt.colorbar(plot, ax=ax4, fraction=0.046, pad=0.04)

    ax1.set_title("Decoded levels")
    ax2.set_title("Latent space and geodesics")
    ax3.set_title("Playability in simulation")
    ax4.set_title("Estimated metric volume")

    plt.tight_layout()
    plt.savefig("data/plots/geodesics_gridsearch.png")

    plt.show()


if __name__ == "__main__":
    # create_table_training_levels()

    model_name = "mariovae_z_dim_2_overfitting_epoch_480"
    geodesics_in_grid(model_name)

    # df = pd.read_csv("./data/processed/training_levels_results.csv")

    # # Reduce the df to only include the mean marioStatus
    # playability = df.groupby("idx").mean()["marioStatus"]
    # playable_idxs = playability[playability > 0.0].index.values
    # non_playable_idxs = playability[playability == 0.0].index.values
    # print(playable_idxs)
    # print(non_playable_idxs)

    # training_tensors, test_tensors = load_data()
    # all_levels = torch.cat((training_tensors, test_tensors))

    # playable_levels = all_levels[playable_idxs]
    # non_playable_levels = all_levels[non_playable_idxs]

    # vae = VAEGeometry()
    # vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    # # print("Updating cluster centers")
    # # vae.update_cluster_centers(model_name, False, beta=-1.5)

    # zs_playable = vae.encode(playable_levels)[0]
    # zs_non_playable = vae.encode(non_playable_levels)[0]

    # print(zs_playable)
    # print(zs_non_playable)

    # zs_p_numpy = zs_playable.detach().numpy()
    # zs_np_numpy = zs_non_playable.detach().numpy()

    # # _, ax = plt.subplots(1, 1)
    # # ax.scatter(zs_p_numpy[:, 0], zs_p_numpy[:, 1], marker="x", c="g")
    # # ax.scatter(zs_np_numpy[:, 0], zs_np_numpy[:, 1], marker="x", c="r")
    # # plt.show()

    # X = np.vstack((zs_p_numpy, zs_np_numpy))
    # y = np.concatenate((np.ones(zs_p_numpy.shape[0]), np.zeros(zs_np_numpy.shape[0])))

    # # X = np.vstack((playable_points, non_playable_points))
    # # y = np.concatenate(
    # #     (
    # #         np.ones((playable_points.shape[0],)),
    # #         np.zeros((non_playable_points.shape[0],)),
    # #     )
    # # )

    # x_lims = y_lims = [-6, 6]

    # k_means = KMeans(n_clusters=50)
    # k_means.fit(zs_p_numpy)

    # kernel = 1.0 * RBF(length_scale=[1.0, 1.0])
    # gpc = GaussianProcessClassifier(kernel=kernel)
    # gpc.fit(X, y)

    # n_x, n_y = 50, 50
    # z1 = torch.linspace(*x_lims, n_x)
    # z2 = torch.linspace(*y_lims, n_x)

    # class_image = np.zeros((n_y, n_x))
    # zs = np.array([[x, y] for x in z1 for y in z2])
    # positions = {
    #     (x.item(), y.item()): (i, j)
    #     for j, x in enumerate(z1)
    #     for i, y in enumerate(reversed(z2))
    # }

    # classes = gpc.predict(zs)
    # for l, (x, y) in enumerate(zs):
    #     i, j = positions[(x.item(), y.item())]
    #     class_image[i, j] = classes[l]

    # _, ax = plt.subplots(1, 1)
    # ax.imshow(class_image, extent=[*x_lims, *y_lims], cmap="Blues")
    # ax.scatter(zs_p_numpy[:, 0], zs_p_numpy[:, 1], marker="o", c="y")
    # ax.scatter(zs_np_numpy[:, 0], zs_np_numpy[:, 1], marker="o", c="r")
    # plt.show()
