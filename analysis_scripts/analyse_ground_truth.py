import json
from pathlib import Path
from mario_utils.plotting import plot_level_from_array

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical
from train_hierarchical_vae import load_data
from mario_utils.levels import onehot_to_levels

# print(list(all_experiments))
def create_table(model_name, path):
    experiment_path = Path(f"./data/ground_truth/{model_name}")
    print(experiment_path)
    all_experiments = list(experiment_path.glob("*.json"))
    print(all_experiments)

    rows = []
    for experiment in all_experiments:
        with open(experiment) as fp:
            row = json.load(fp)
            clean_row = {"z1": row["z"][0], "z2": row["z"][1], **row}
            del clean_row["z"]
        rows.append(clean_row)

    df = pd.DataFrame(rows)
    # print(df)

    df.to_csv(path)


def plot_column(df, column_name, ax=None):
    z1 = sorted(df["z1"].unique().tolist())
    z2 = sorted(df["z2"].unique().tolist(), reverse=True)

    n_x = len(z1)
    n_y = len(z2)

    M = np.zeros((n_y, n_x))
    for idx in df.index:
        z1i, z2i, m = df.loc[idx, ["z1", "z2", column_name]]
        i, j = z2.index(z2i), z1.index(z1i)
        # print(f"for {i,j}: ({z1i}, {z2i}): {m}")
        M[i, j] = m

    if ax is None:
        _, ax = plt.subplots(1, 1)

    plot = ax.imshow(M, extent=[min(z1), max(z1), min(z2), max(z2)], cmap="Blues")
    plt.colorbar(plot, ax=ax)

    if ax is None:
        plt.show()
        plt.close()


def plot_latent_space(model, training_data, ax, x_lim, y_lim):
    # Loading up the training points
    zs_training, _ = model.encode(training_data)

    print("zs for data")
    print(zs_training)
    print(zs_training.shape)

    zs_training = zs_training.detach().numpy()

    ax.scatter(zs_training[:, 0], zs_training[:, 1], label="Training encodings")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend()


if __name__ == "__main__":
    cwd = Path(".")

    model_name = "hierarchical_final_playable_final"
    # Creates the table if they don't exist.
    processed_s_exp_path = cwd / "data" / "processed" / "ground_truth"
    processed_s_exp_path.mkdir(exist_ok=True)

    path = processed_s_exp_path / f"{model_name}_ground_truth.csv"
    # create_table("hierarchical_final_playable", path)

    # Load the table and process it
    df = pd.read_csv(path, index_col=0)
    print(df.columns)
    interesting = lambda col: (
        col not in ["z1", "z2", "model_name"]
        and not col.startswith("total")
        and not col.startswith("agent")
        and not col.startswith("iteration")
        and not "Phys" in col
    )
    interesting_columns = [col for col in df.columns if interesting(col)]

    vae = VAEGeometryHierarchical()
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.update_cluster_centers(beta=-3.0, n_clusters=500)
    vae.eval()

    training_tensors, _ = load_data(only_playable=True)

    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
    all_axes = [*axes[0, 1:], *axes[1, :]]
    # plot_latent_space(vae, training_tensors, axes[0, 0], x_lim=[-6, 6], y_lim=[-6, 6])
    vae.plot_latent_space(ax=axes[0, 0])
    for ax, col_name in zip(all_axes, interesting_columns):
        ax.set_title(col_name)
        plot_column(df, col_name, ax)

    # plt.show()
    plots_path = cwd / "data" / "plots"
    plots_path.mkdir(exist_ok=True)
    fig.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(plots_path / f"sim_results_{model_name}.png")
    plt.close()
