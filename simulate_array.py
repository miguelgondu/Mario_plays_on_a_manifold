from pathlib import Path
import multiprocessing as mp
import json
from typing import Dict, List
from itertools import repeat

import click
import numpy as np
import pandas as pd

from simulator import test_level_from_int_array


def test_level(
    i: int,
    z: np.ndarray,
    level: np.ndarray,
    array_name: str,
    exp_folder: str = None,
    visualize: bool = False,
    verbose: bool = False
):
    if verbose:
        print(f"Processing level at index {i} (z={z})")

    if exp_folder is not None:
        res_path = Path("./data/array_simulation_jsons") / exp_folder
        res_path.mkdir(exist_ok=True, parents=True)
        saving_path = res_path / f"{array_name}_{i:08d}.json"
    else:
        saving_path = (
            Path("./data/array_simulation_jsons") / f"{array_name}_{i:08d}.json"
        )

    # Check if we have already simulated it, and if so, load it instead.
    if saving_path.exists():
        with open(saving_path) as fp:
            res = json.load(fp)
            assert res["z"] == z.tolist()
            assert (np.array(json.loads(res["level"])) == level).all()
            return res

    res = test_level_from_int_array(level, visualize=visualize)
    res = {"z": z.tolist(), "level": level.tolist(), **res}
    if verbose:
        print(f"Processed {i} (z={z}, p={res['marioStatus']})")

    with open(saving_path, "w") as fp:
        json.dump(res, fp)

    return res


def _simulate_array(
    array_path, processes, repetitions_per_level, exp_folder=None, visualize=False
):
    array_path = Path(array_path)
    array_name = array_path.stem

    array = np.load(array_path)
    levels = array["levels"]
    zs = array["zs"]

    assert levels.shape[0] == zs.shape[0]
    print(f"Simulating {array_path}")
    print(
        f"Will process {levels.shape[0]} levels ({levels.shape[0] * repetitions_per_level} simulations)."
    )

    # Check if we've already simulated it
    results_path = Path("./data/array_simulation_results")
    if exp_folder is not None:
        (results_path / exp_folder).mkdir(exist_ok=True, parents=True)
        saving_path = results_path / exp_folder / f"{array_name}.csv"
    else:
        saving_path = results_path / f"{array_name}.csv"

    if saving_path.exists():
        print(f"Skipping this array, since we found something in {saving_path}")
        return

    # Repeat
    levels = np.repeat(levels, repetitions_per_level, axis=0)
    zs = np.repeat(zs, repetitions_per_level, axis=0)

    with mp.Pool(processes) as p:
        results = p.starmap(
            test_level,
            zip(
                range(len(zs)),
                zs,
                levels,
                repeat(array_name),
                repeat(exp_folder),
                repeat(visualize),
            ),
        )

    rows = []
    for z, level, result in zip(zs, levels, results):
        row = {"z": z.tolist(), "level": level.tolist(), **result}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(saving_path)


@click.command()
@click.argument(
    "array_path", type=str, default="./data/arrays/samples_for_playability.npz"
)
@click.option("--processes", type=int, default=5)
@click.option("--repetitions_per_level", type=int, default=1)
@click.option("--exp_folder", type=str, default=None)
def simulate_array(array_path, processes, repetitions_per_level, exp_folder):
    """
    Takes an array stored as an .npz with
    the keys "zs" and "levels" and simulates it,
    storing the results in a csv with the same name
    as the array in ./data/array_simulation_results.

    In the process, saves each individual result in
    ./data/array_simulation_jsons.
    """
    _simulate_array(array_path, processes, repetitions_per_level, exp_folder=exp_folder)


if __name__ == "__main__":
    simulate_array()
