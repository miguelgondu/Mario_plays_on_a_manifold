"""
This experiment gets random interpolations on
the latent space and simulates them, measuring
playability in each one of them.
"""
from pathlib import Path
from typing import List
from vae_mario_hierarchical import VAEMarioHierarchical
import pandas as pd
import torch
import click
import numpy as np
import json
import torch.multiprocessing as mp
from itertools import repeat

from vae_geometry_hierarchical import VAEGeometryHierarchical
from geoml.discretized_manifold import DiscretizedManifold

from simulator import test_level_from_z
from interpolations.base_interpolation import BaseInterpolation
from interpolations.linear_interpolation import LinearInterpolation
from interpolations.astar_interpolation import AStarInterpolation
from interpolations.geodesic_interpolation import GeodesicInterpolation

Tensor = torch.Tensor


def get_playable_points(model_name: str) -> np.ndarray:
    playable_points = np.load(
        f"data/processed/all_playable_encodings_{model_name}.npz"
    )["encodings"]

    return playable_points


def load_model(model_name: str, only_playable=False) -> VAEGeometryHierarchical:
    # Get playable points and lines
    if only_playable:
        playable_points = get_playable_points(model_name)
        encodings = torch.from_numpy(playable_points)
    else:
        encodings = None

    vae = VAEGeometryHierarchical()
    vae.load_state_dict(torch.load(f"models/{model_name}.pt", map_location="cpu"))
    print("Updating cluster centers")
    vae.update_cluster_centers(
        model_name,
        False,
        beta=-2.5,
        n_clusters=300,
        encodings=encodings,
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
    model_name: VAEGeometryHierarchical,
    line_id: int,
    line: List[Tensor],
    path_to_exp: Path = None,
    experiment_name: str = None,
    only_playable: bool = False,
):
    # Load the model
    if only_playable:
        playable_points = get_playable_points(model_name)
        encodings = torch.from_numpy(playable_points)
    else:
        encodings = None

    vae = VAEGeometryHierarchical(device="cpu")
    vae.load_state_dict(torch.load(f"models/{model_name}.pt", map_location="cpu"))
    # print("Updating cluster centers")
    vae.update_cluster_centers(
        model_name,
        False,
        beta=-2.5,
        n_clusters=300,
        encodings=encodings,
    )
    vae.eval()

    for i, z in enumerate(line):
        print(f"Evaluating line {line_id} (z {i}).")
        result = test_level_from_z(z, vae)
        result = {
            "experiment_name": experiment_name,
            "line_id": line_id,
            "line": [z.tolist() for z in line],
            "z_idx": i,
            "z": z.tolist(),
            **result,
        }

        with open(
            path_to_exp / f"{experiment_name}_{line_id:05d}_{i:05d}.json", "w"
        ) as fp:
            json.dump(result, fp)


# @click.command()
# @click.argument("model_name", default="final_overfitted_nnj_epoch_300", type=str)
# @click.option("--n-lines", default=100, type=int)
# @click.option("--n-points-in-line", default=20, type=int)
# def save_lines(model_name, n_lines, n_points_in_line):
#     vae_simple = VAEMarioHierarchical()
#     vae_simple.load_state_dict(
#         torch.load(f"./models/{model_name}.pt", map_location="cpu")
#     )
#     all_playable_levels = np.load("./data/processed/all_playable_levels_onehot.npz")[
#         "levels"
#     ]
#     playable_encodings = (
#         vae_simple.encode(torch.from_numpy(all_playable_levels).type(torch.float))
#         .mean.detach()
#         .numpy()
#     )
#     np.savez(
#         f"./data/processed/all_playable_encodings_{model_name}.npz",
#         encodings=playable_encodings,
#     )

#     vae = load_model(model_name)

#     playable_encodings = torch.from_numpy(playable_encodings).type(torch.float)

#     z1s, z2s = get_random_pairs(playable_encodings, n_lines)

#     for interpolation in ["linear", "geodesic", "astar"]:
#         if interpolation == "linear":
#             inter = LinearInterpolation(n_points_in_line=n_points_in_line)
#         elif interpolation == "geodesic":
#             grid = [torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50)]
#             Mx, My = torch.meshgrid(grid[0], grid[1])
#             grid2 = torch.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)
#             DM = DiscretizedManifold(vae, grid2, use_diagonals=True)
#             inter = GeodesicInterpolation(
#                 DM, model_name, n_points_in_line=n_points_in_line
#             )
#         elif interpolation == "astar":
#             continue
#             # inter = AStarInterpolation(n_points_in_line, model_name)

#         # Core of the experiment:
#         experiment_name = f"interpolation_{model_name}_{interpolation}"
#         filepath = Path(__file__).parent.resolve()
#         path_to_lines = filepath / "data" / "interpolation_experiment" / experiment_name
#         path_to_lines.mkdir(exist_ok=True)

#         lines = np.array(
#             [inter.interpolate(z1, z2).detach().numpy() for z1, z2 in zip(z1s, z2s)]
#         )  # n_lines x n_points_in_line x 2.
#         print(lines)
#         print(lines.shape)

#         np.savez(path_to_lines / f"{interpolation}_lines.npz", lines=lines)


@click.command()
@click.argument("model_name", default="final_overfitted_nnj_epoch_300", type=str)
@click.option("--interpolation", default="linear", type=str)
@click.option("--processes", default=10, type=int)
def experiment(model_name, interpolation, processes):
    # Core of the experiment:
    experiment_name = f"interpolation_{model_name}_{interpolation}"

    filepath = Path(__file__).parent.resolve()
    path_to_exp = filepath / "data" / "interpolation_experiment" / experiment_name

    # Loading lines
    lines = np.load(path_to_exp / f"{interpolation}_lines.npz", allow_pickle=True)[
        "lines"
    ]
    # lines = torch.from_numpy(lines).detach()

    if processes > 1:
        with mp.Pool(processes=processes) as p:
            p.starmap(
                simulate_line,
                zip(
                    repeat(model_name),
                    range(len(lines)),
                    [torch.from_numpy(l).type(torch.float).detach() for l in lines],
                    repeat(path_to_exp),
                    repeat(experiment_name),
                ),
            )
    else:
        for line in lines:
            simulate_line(model_name, line, path_to_exp, experiment_name)


if __name__ == "__main__":
    experiment()
    # save_lines()
