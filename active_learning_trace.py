"""
This script gets a 1000 AL queries trace
for a given model name.
"""
from typing import Tuple
import json
from itertools import product
import multiprocessing as mp
from pathlib import Path

import click
import torch as t
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from vae_mario_hierarchical import VAEMarioHierarchical
from simulate_array import _simulate_array
from simulator import test_level_from_decoded_tensor


def load_vae(model_name) -> VAEMarioHierarchical:
    vae = VAEMarioHierarchical()
    device = vae.device
    vae.load_state_dict(t.load(f"./models/{model_name}.pt", map_location=device))

    return vae


def get_initial_data(model_name):
    vae = load_vae(model_name)

    initial_data_path = Path("./data/arrays/five_vaes/initial_data_AL")
    initial_data_path.mkdir(exist_ok=True)

    array_path = initial_data_path / f"{model_name}_100_points_in_a_grid.npz"

    zs = t.Tensor(
        [[z1, z2] for z1, z2 in product(t.linspace(-5, 5, 10), t.linspace(-5, 5, 10))]
    )
    levels = vae.decode(zs).probs.argmax(dim=-1)

    np.savez(
        array_path,
        zs=zs.detach().numpy(),
        levels=levels.detach().numpy(),
    )

    _simulate_array(array_path, 5, 5, exp_folder="five_vaes/initial_data_AL")


def load_initial_data(model_name) -> Tuple[np.ndarray]:
    results_path = Path("./data/array_simulation_results/five_vaes/initial_data_AL")
    df = pd.read_csv(
        results_path / f"{model_name}_100_points_in_a_grid.csv",
        index_col=0,
    )
    by_z = df.groupby("z")["marioStatus"].mean()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    playabilities = by_z.values
    playabilities[playabilities > 0.0] = 1.0

    return zs, playabilities


def query(gpc: GaussianProcessClassifier) -> Tuple[np.ndarray, float]:
    """
    In AL literature, they refer to max uncertainty as the points
    that are the closest to the a 0.5 probability prediction
    """
    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
    probs = gpc.predict_proba(bigger_grid)
    close_to_uncertainty = np.abs(probs[:, 0] - 0.5)
    next_point = bigger_grid[np.argmin(close_to_uncertainty)]

    return next_point, np.min(close_to_uncertainty)


def simulate_level(level: t.Tensor, processes: int, repeats: int) -> int:
    levels = np.repeat(level.cpu().reshape(1, 14, 14), repeats, axis=0)
    with mp.Pool(processes) as p:
        results = p.map(test_level_from_decoded_tensor, levels)

    playabilities = [r["marioStatus"] for r in results]
    res = np.mean(playabilities)
    if res > 0.0:
        res = 1.0

    return int(res)


@click.command()
@click.argument("model-name", type=str)
def run(model_name):
    """
    Runs 1000 AL queries for this model.
    """

    # Hyperparameters
    kernel = 1.0 * Matern(nu=3 / 2) + 1.0 * WhiteKernel()
    n_iterations = 1000

    # Results keeping
    results_path = Path("./data/evolution_traces/five_vaes")
    results_path.mkdir(exist_ok=True)

    # Loading models
    vae = load_vae(model_name)
    gpc = GaussianProcessClassifier(kernel=kernel)

    # Bootstrapping with initial data
    zs, playabilities = load_initial_data()
    gpc.fit(zs, playabilities)

    for _ in range(n_iterations):
        # Get next point to query
        next_point, _ = query(gpc)

        next_level = vae.decode(t.Tensor(next_point)).probs.argmax(dim=-1)
        p = simulate_level(next_level, 5, 5)

        # print(f"Tested {next_point}. p={p}. ALS={als} ({i+1}/{n_iterations})")

        zs = np.vstack((zs, next_point))
        playabilities = np.concatenate((playabilities, np.array([p])))

        np.savez(
            results_path / f"{model_name}_AL_trace_{n_iterations}.npz",
            zs=zs,
            playabilities=playabilities,
        )
        gpc = GaussianProcessClassifier(kernel=kernel)
        gpc.fit(zs, playabilities)


if __name__ == "__main__":
    run()
