from pathlib import Path
import multiprocessing as mp
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from simulator import test_level_from_int_array


@click.command()
@click.argument("array_path", type=str)
@click.option("--processes", type=int, default=5)
@click.option("--repetitions_per_level", type=int, default=1)
def simulate_array(array_path, processes, repetitions_per_level):
    """
    Takes an array stored as an .npz with
    the keys "zs" and "levels" and simulates it,
    storing the results in a csv with the same name
    as the array in ./data/array_simulation_results
    """
    array_path = Path(array_path)
    array_name = array_path.name.replace(".npz", "")

    array = np.load(array_path)
    levels = array["levels"]
    zs = array["zs"]

    # Repeat
    levels = np.repeat(levels, repetitions_per_level, axis=0)
    zs = np.repeat(zs, repetitions_per_level, axis=0)

    with mp.Pool(processes) as p:
        results = p.map(test_level_from_int_array, levels)

    rows = []
    for z, level, result in zip(zs, levels, results):
        row = {"z": z.tolist(), "level": level.tolist(), **result}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"./data/array_simulation_results/{array_name}.csv")


if __name__ == "__main__":
    simulate_array()
