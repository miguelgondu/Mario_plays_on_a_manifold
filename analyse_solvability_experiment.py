import json
from pathlib import Path
from mario_utils.plotting import plot_level_from_array

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vae_mario import VAEMario
from train_vae import load_data
from mario_utils.levels import onehot_to_levels

# print(list(all_experiments))
def create_table(model_name, path):
    experiment_path = Path(f"./data/playability_experiment/{model_name}")
    all_experiments = list(experiment_path.glob("*.json"))
    # print(all_experiments)

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


def get_playable_points(df: pd.DataFrame) -> np.ndarray:
    """
    Gets all the zs where marioStatus == 1.
    """
    winning = df[df["marioStatus"] == 1]
    z1 = winning["z1"]
    z2 = winning["z2"]
    return np.vstack((z1, z2)).T


if __name__ == "__main__":
    cwd = Path(".")
    model_names = (cwd / "data" / "playability_experiment").glob("mariovae_*")
    model_names = list(model_names)
    # This works.
    # for model_name in model_names:
    #     print(model_name.name)

    # Creates the table if they don't exist.
    processed_s_exp_path = cwd / "data" / "processed" / "playability_experiment"
    processed_s_exp_path.mkdir(exist_ok=True)

    for model_name in model_names:
        model_name = model_name.name
        path = processed_s_exp_path / f"{model_name}_playability_experiment.csv"
        if not path.exists():
            # print(f"Creating the table for {model_name} at {path}")
            create_table(model_name, path)

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

        clusters = get_playable_points(df)
        print(clusters)
        np.savez(
            f"./data/processed/playable_clusters_{model_name}.npz", clusters=clusters
        )
        continue

        vae = VAEMario(14, 14, z_dim=2)
        vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
        vae.eval()

        training_tensors, _ = load_data(only_playable="only_playable" in model_name)

        fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
        all_axes = [*axes[0, 1:], *axes[1, :]]
        plot_latent_space(
            vae, training_tensors, axes[0, 0], x_lim=[-6, 6], y_lim=[-6, 6]
        )
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
