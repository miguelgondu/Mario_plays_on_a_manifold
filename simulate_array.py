from pathlib import Path
import multiprocessing as mp
import json
from typing import Dict, List
from itertools import repeat

import click
import numpy as np
import pandas as pd

from simulator import test_level_from_int_array


def test_level(i: int, z: np.ndarray, level: np.ndarray, array_name: str):
    print(f"Processing level at index {i} (z={z})")
    res = test_level_from_int_array(level)
    res = {"z": z.tolist(), "level": level.tolist(), **res}

    with open(f"./data/array_simulation_jsons/{array_name}_{i:08d}.json", "w") as fp:
        json.dump(res, fp)

    return res


@click.command()
@click.argument(
    "array_path", type=str, default="./data/arrays/samples_for_playability.npz"
)
@click.option("--processes", type=int, default=5)
@click.option("--repetitions_per_level", type=int, default=1)
def simulate_array(array_path, processes, repetitions_per_level):
    """
    Takes an array stored as an .npz with
    the keys "zs" and "levels" and simulates it,
    storing the results in a csv with the same name
    as the array in ./data/array_simulation_results.

    In the process, saves each individual result in
    ./data/array_simulation_jsons.
    """
    array_path = Path(array_path)
    array_name = array_path.name.replace(".npz", "")

    array = np.load(array_path)
    levels = array["levels"]
    zs = array["zs"]

    assert levels.shape[0] == zs.shape[0]

    # Repeat
    levels = np.repeat(levels, repetitions_per_level, axis=0)
    zs = np.repeat(zs, repetitions_per_level, axis=0)

    print(f"Will process {levels.shape[0]} levels.")
    with mp.Pool(processes) as p:
        results = p.starmap(
            test_level, zip(range(len(zs)), zs, levels, repeat(array_name))
        )

    rows = []
    for z, level, result in zip(zs, levels, results):
        row = {"z": z.tolist(), "level": level.tolist(), **result}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"./data/array_simulation_results/{array_name}.csv")


if __name__ == "__main__":
    # simulate_array()
    simulate_array()
