"""
This script gets a 500 AL queries trace
for a given model name.
"""
from typing import Tuple
import json
from itertools import product
import multiprocessing as mp
from pathlib import Path
from time import time

import click
import torch as t
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from utils.simulator.simulate_array import _simulate_array
from utils.simulator.interface import test_level_from_decoded_tensor


def load_vae(model_name) -> VAEMarioHierarchical:
    vae = VAEMarioHierarchical()
    device = vae.device
    vae.load_state_dict(
        t.load(f"./models/ten_vaes/{model_name}.pt", map_location=device)
    )

    return vae


def get_initial_data(model_name):
    vae = load_vae(model_name)

    initial_data_path = Path("./data/arrays/ten_vaes/initial_data_AL")
    initial_data_path.mkdir(exist_ok=True, parents=True)

    array_path = initial_data_path / f"{model_name}.npz"

    zs = t.Tensor(
        [[z1, z2] for z1, z2 in product(t.linspace(-5, 5, 10), t.linspace(-5, 5, 10))]
    )
    levels = vae.decode(zs).probs.argmax(dim=-1)

    np.savez(
        array_path,
        zs=zs.detach().numpy(),
        levels=levels.cpu().detach().numpy(),
    )

    _simulate_array(array_path, 40, 5, exp_folder="ten_vaes/initial_data_AL")


def load_initial_data(model_name) -> Tuple[np.ndarray]:
    results_path = Path("./data/array_simulation_results/ten_vaes/initial_data_AL")
    df = pd.read_csv(
        results_path / f"{model_name}.csv",
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
@click.option("--overwrite/--not-overwrite", default=False)
def run(model_name, overwrite):
    """
    Runs 500 AL queries for this model.
    """

    # Hyperparameters
    kernel = 1.0 * Matern(nu=3 / 2) + 1.0 * WhiteKernel()
    n_iterations = 500

    # Results keeping
    results_path = Path("./data/evolution_traces/ten_vaes")
    results_path.mkdir(exist_ok=True, parents=True)
    trace_path = results_path / f"{model_name}.npz"

    # Getting initial data
    initial_data_path = Path("./data/array_simulation_results/ten_vaes/initial_data_AL")

    # Checking if we have already ran something before
    if trace_path.exists():
        a = np.load(results_path / f"{model_name}.npz")
        zs = a["zs"]
        playabilities = a["playabilities"]

        if len(zs) > n_iterations + 100 and not overwrite:
            # We don't need to run anything
            print(f"We already have a trace at {trace_path}. Queries: {len(zs) - 100}")
            return

        to_run = n_iterations - (len(zs) - 100)
    else:
        to_run = n_iterations
        if (initial_data_path / f"{model_name}.csv").exists():
            zs, playabilities = load_initial_data(model_name)
        else:
            get_initial_data(model_name)
            zs, playabilities = load_initial_data(model_name)

    if overwrite:
        to_run = n_iterations
        zs = zs[:100]
        playabilities = playabilities[:100]

    # Loading models
    it = time()
    vae = load_vae(model_name)
    gpc = GaussianProcessClassifier(kernel=kernel)
    print(f"Loaded the models: {time() - it:.2f}")

    # Bootstrapping with initial data
    it = time()
    gpc.fit(zs, playabilities)
    print(f"Fitted the initial gpc: {time() - it:.2f}")

    for i in range(to_run):
        it = time()
        # Get next point to query
        next_point, als = query(gpc)
        print(f"Queried the next point: {time() - it:.2f}")

        it = time()
        next_level = vae.decode(t.Tensor(next_point)).probs.argmax(dim=-1)
        p = simulate_level(next_level, 5, 5)
        print(f"Simulated the level: {time() - it:.2f}")

        print(
            f"Tested {next_point}. p={p:.1f}. ALS={als:1.4f} ({len(zs)-100}/{n_iterations})"
        )

        zs = np.vstack((zs, next_point))
        playabilities = np.concatenate((playabilities, np.array([p])))

        if ((i + 1) % 10) or (i + 1 == n_iterations):
            np.savez(
                results_path / f"{model_name}.npz",
                zs=zs,
                playabilities=playabilities,
            )

        it = time()
        kernel = 1.0 * Matern(nu=3 / 2) + 1.0 * WhiteKernel()
        gpc = GaussianProcessClassifier(kernel=kernel)
        gpc.fit(zs, playabilities)
        print(f"Fitted the GPC (on {len(zs)}): {time() - it:.2f}")


if __name__ == "__main__":
    run()
