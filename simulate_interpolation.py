"""
This script grabs a list of levels
(presented in the scheme of List[dict["z", "level"]])
and runs it with the simulator.
"""
from typing import List
import os
import json
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

# from random_interpolation import random_interpolation_experiment
from simulator import test_level_from_z
from vae_mario import VAEMario
from vae_geometry import VAEGeometry

MODELS_PATH = "./models_experiment"


def random_interpolation_experiment(
    n_pairs: int,
    x_lims: List[float],
    y_lims: List[float],
    n_points_in_line: int = 20,
):
    """
    This function selects n_pairs of points
    between x_lims and y_lims, interpolates
    them with a line and returns a list of levels
    in between.
    """
    # If you load this here, torch locks in multiprocessing
    # afterward. SO WEIRD.
    # vae = VAEGeometry()
    # vae.load_state_dict(torch.load(f"models_experiment/{model_name}.pt"))
    # vae.eval()

    lower = torch.Tensor([x_lims[0], y_lims[0]])
    upper = torch.Tensor([x_lims[1], y_lims[1]])

    lines_of_zs = []

    for n_pair in range(n_pairs):
        z1 = lower + (upper - lower) * torch.rand((1, 2))
        z2 = lower + (upper - lower) * torch.rand((1, 2))

        # print(z1, z2)
        t = torch.linspace(0, 1, n_points_in_line).unsqueeze(-1)
        line = (1 - t) * z1 + t * z2
        # levels = vae.decode(line)
        for z in line:
            lines_of_zs.append({"n_line": n_pair, "z": z})

    return lines_of_zs


def process(i, z, n_line, model_name):
    """
    asdf
    """
    print(f"Processing line {n_line}, point {i} ({z})")
    print(f"Loading model {model_name}")
    vae = VAEGeometry()
    vae.load_state_dict(torch.load(f"{MODELS_PATH}/{model_name}.pt"))
    vae.eval()
    print(f"Model loaded!")
    # print(f"Processing line {n_line}, point {i}")

    # Create the folder for the data
    cwd = Path(".")
    path_to_exp_folder = cwd / "data" / "interpolation_experiment" / model_name
    path_to_exp_folder.mkdir(parents=True, exist_ok=True)

    # Test the level 5 times
    for iteration in range(5):
        res = test_level_from_z(z, vae)
        thing_to_save = {
            "model_name": model_name,
            "n_line": n_line,
            "z": tuple([float(zi) for zi in z.detach().numpy()]),
            "iteration": iteration,
            **res,
        }

        path_to_exp = (
            path_to_exp_folder / f"z_line_{n_line}_point_{i:03d}_{iteration}.json"
        )
        with open(path_to_exp, "w") as fp:
            json.dump(thing_to_save, fp)

    print(f"Processed z {z} for model {model_name}.")


@click.command()
@click.option(
    "--model-name", type=str, default="mariovae_z_dim_2_overfitting_epoch_480"
)
@click.option("--processors", type=int, default=None)
def main(model_name, processors):
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    n_points_in_line = 10
    lines = random_interpolation_experiment(
        10, x_lims, y_lims, n_points_in_line=n_points_in_line
    )

    zs = torch.vstack([doc["z"] for doc in lines]).type(torch.float)
    print(zs)
    n_lines = [doc["n_line"] for doc in lines]
    # print(n_lines)
    is_ = [i % n_points_in_line for i in range(len(zs))]
    # print(is_)

    if processors != None:
        print(f"Starting the Pool for {model_name}.")

        with Pool(processors) as p:
            p.starmap(
                process,
                zip(is_, zs, n_lines, repeat(model_name)),
            )
    else:
        for tuple_ in zip(is_, zs, n_lines, repeat(model_name)):
            process(*tuple_)


if __name__ == "__main__":
    main()
