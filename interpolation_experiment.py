"""
This experiment gets random interpolations on
the latent space and simulates them, measuring
playability in each one of them.
"""
from base_interpolation import BaseInterpolation
from typing import List
import pandas as pd
import torch
import click
import numpy as np

from vae_mario import VAEMario
from vae_geometry import VAEGeometry
from train_vae import load_data

from simulator import test_level_from_z
from linear_interpolation import LinearInterpolation
from astar_interpolation import AStarInterpolation
from geodesic_interpolation import GeodesicInterpolation

Tensor = torch.Tensor


def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


def load_model(model_name: str) -> VAEGeometry:
    # Get playable points and lines
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

    return vae


def get_random_pairs(
    encodings: Tensor, n_pairs: int = 100, seed: int = 17
) -> List[Tensor]:
    np.random.seed(seed)
    idx1 = np.random.choice(len(encodings), size=n_pairs, replace=False)
    idx2 = np.random.choice(len(encodings), size=n_pairs, replace=False)
    while np.any(idx1 == idx2):
        idx2 = np.random.choice(len(encodings), size=n_pairs, replace=False)

    pairs_1 = encodings[idx1]
    pairs_2 = encodings[idx2]

    return pairs_1, pairs_2


@click.command()
@click.argument(
    "model_name", default="mariovae_z_dim_2_overfitting_epoch_480", type=str
)
@click.option("--interpolation", default="linear", type=str)
@click.option("--n-lines", default=20, type=int)
@click.option("--n-points-in-line", default=10, type=int)
def experiment(model_name, interpolation, n_lines, n_points_in_line):
    vae = load_model(model_name)

    playable_tensors, _ = load_data(only_playable=True)
    playable_encodings, _ = vae.encode(playable_tensors)

    z1s, z2s = get_random_pairs(playable_encodings, n_lines)

    if interpolation == "linear":
        inter = LinearInterpolation(n_points=n_points_in_line)
