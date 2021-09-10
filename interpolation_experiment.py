"""
This experiment gets random interpolations on
the latent space and simulates them, measuring
playability in each one of them.
"""
from pathlib import Path
from typing import List
import pandas as pd
import torch
import click
import numpy as np
import json
import multiprocessing as mp
from itertools import repeat

from vae_mario import VAEMario
from vae_geometry import VAEGeometry
from train_vae import load_data
from geoml.discretized_manifold import DiscretizedManifold

from simulator import test_level_from_z
from base_interpolation import BaseInterpolation
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


def simulate_line(
    vae: VAEGeometry,
    line: List[Tensor],
    path_to_exp: Path = None,
    experiment_name: str = None,
):
    for i, z in enumerate(line):
        result = test_level_from_z(z, vae)
        result = {
            "experiment_name": experiment_name,
            "line_idx": i,
            "line": [z.tolist() for z in line],
            **result,
        }

        with open(path_to_exp / f"{experiment_name}_{i:05d}.json") as fp:
            json.dump(result, fp)


@click.command()
@click.argument(
    "model_name", default="mariovae_z_dim_2_overfitting_epoch_480", type=str
)
@click.option("--interpolation", default="linear", type=str)
@click.option("--n-lines", default=20, type=int)
@click.option("--n-points-in-line", default=10, type=int)
@click.option("--processes", default=10, type=int)
def experiment(model_name, interpolation, n_lines, n_points_in_line, processes):
    vae = load_model(model_name)

    playable_tensors, _ = load_data(only_playable=True)
    playable_encodings, _ = vae.encode(playable_tensors)

    z1s, z2s = get_random_pairs(playable_encodings, n_lines)

    if interpolation == "linear":
        inter = LinearInterpolation(n_points=n_points_in_line)
    elif interpolation == "geodesic":
        grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
        Mx, My = torch.meshgrid(grid[0], grid[1])
        grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)
        DM = DiscretizedManifold(vae, grid2, use_diagonals=True)
        inter = GeodesicInterpolation(DM, model_name, n_points_in_line=n_points_in_line)
    elif interpolation == "astar":
        inter = AStarInterpolation(n_points_in_line, model_name)

    # Core of the experiment: run the jewels
    experiment_name = f"interpolation_{model_name}_{interpolation}"

    filepath = Path(__file__).parent.resolve()
    path_to_exp = filepath / "data" / "interpolation_experiment" / experiment_name
    path_to_exp.mkdir(exist_ok=True)

    lines = [inter.interpolate(z1, z2) for z1, z2 in zip(z1s, z2s)]
    with mp.Pool(processes=processes) as p:
        p.starmap(
            simulate_line,
            zip(repeat(vae), lines, repeat(path_to_exp), repeat(experiment_name)),
        )
