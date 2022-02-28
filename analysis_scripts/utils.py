from itertools import repeat
from typing import Tuple, List
import json
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit


def zs_and_playabilities(filepath: str) -> Tuple[np.ndarray]:
    """
    Loads a csv in filepath (expected to be the result
    of simulating an array), and returns zs and playabilities.
    """
    df = pd.read_csv(filepath)
    by_z = df.groupby("z")["marioStatus"].mean()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    playabilities = by_z.values
    playabilities[playabilities > 0.0] = 1.0

    return zs, playabilities


def zs_and_levels(filepath: str) -> Tuple[np.ndarray]:
    """
    Loads a csv in filepath (expected to be the result
    of simulating an array), and returns zs and levels.
    """
    df = pd.read_csv(filepath)
    by_z = df.groupby("z")["levels"].unique()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    levels = []
    for unique_levels in by_z.values:
        for l in unique_levels:
            levels.append(json.loads(l))

    return zs, np.array(levels)


def get_mean_playability_of_csv(path: Path) -> float:
    return get_mean_column_of_csv(path, "marioStatus")


def get_mean_column_of_csv(path: Path, column: str) -> float:
    df = pd.read_csv(path)
    return df[column].mean()


def get_levels_of_csv(path: Path) -> List[np.ndarray]:
    df = pd.read_csv(path, index_col=0)

    by_z = df.groupby("z")
    levels = by_z["level"].unique().values
    levels = [
        np.array(json.loads(level))
        for unique_levels in levels
        for level in unique_levels
    ]

    return levels


def get_mean(
    experiment: List[Path],
    column: str,
    return_std: bool = False,
    plot_hist: bool = True,
) -> float:
    means = []
    for p in experiment:
        means.append(get_mean_column_of_csv(p, column))

    if plot_hist:
        name = f"{p.parent.stem}_{p.parent.parent.stem}"
        fig, ax = plt.subplots(1, 1)
        ax.hist(means)
        ax.set_title(name)
        fig.savefig(f"./data/plots/ten_vaes/mean_hists/{name}.png")
        plt.close(fig)

    if return_std:
        return np.mean(means), np.std(means)
    else:
        return np.mean(means)


def get_proportion_w_jump(experiment: List[Path]) -> Tuple[float]:
    means = []
    for i in range(10):
        exps_in_id = filter(lambda x: f"_id_{i}_" in x.name, experiment)
        dfs = [pd.read_csv(p) for p in exps_in_id]
        df = pd.concat(dfs)
        prop = len(df[df["jumpActionsPerformed"] > 0.0]) / len(df)
        means.append(prop)

    return np.mean(means), np.std(means)


def get_mean_w_conditioning(
    experiment: List[Path], column: str = "jumpActionsPerformed"
) -> Tuple[float]:
    # Get all arrays and concatenate them together
    dfs = [pd.read_csv(p) for p in experiment]
    df = pd.concat(dfs)
    # maybe do a grouping by z (?)

    # Slice w. the condition of playability
    df_playable = df[df["marioStatus"] == 1.0]

    # compute means and stds.
    mean = df_playable[column].mean()
    std = df_playable[column].std()

    return mean, std


def get_mean_playability(
    experiment: List[Path], processes: int = None, return_std: bool = False
) -> float:
    if processes is not None:
        with mp.Pool(processes) as pool:
            means = pool.map(get_mean_playability_of_csv, experiment)
    else:
        means = []
        for p in experiment:
            means.append(get_mean_playability_of_csv(p))

    # assert len(means) in [20, 50]
    if return_std:
        return np.mean(means), np.std(means)
    else:
        return np.mean(means)


def get_all_levels(experiment: List[Path], processes: int = None) -> List[np.ndarray]:
    if processes is not None:
        with mp.Pool(processes) as pool:
            levels_packed = pool.map(get_levels_of_csv, experiment)
        levels_as_arrays = [lvl for lvl_array in levels_packed for lvl in lvl_array]
    else:
        levels_as_arrays = []
        for p in experiment:
            levels = get_levels_of_csv(p)
            levels_as_arrays += levels

    return levels_as_arrays


@jit(nopython=True)
def similarity(level_a: np.ndarray, level_b: np.ndarray) -> float:
    w, h = level_a.shape
    coincide = np.count_nonzero(level_a == level_b)

    return (1 / (w * h)) * coincide


def get_mean_diversities_of_levels(levels: List[np.ndarray]) -> float:
    similarities = []
    for a, level in enumerate(levels):
        for another_level in levels[a + 1 :]:
            sim_ = similarity(level, another_level)
            similarities.append(sim_)

    similarities = np.array(similarities)

    return np.mean(1 - similarities)


def get_mean_diversities(experiment: List[Path], processes: int = None) -> float:
    levels = get_all_levels(experiment, processes=processes)
    return get_mean_diversities_of_levels(levels)


def load_experiment_csv_paths(exp_name: str) -> Tuple[List[Path], List[Path]]:
    """
    Loads up interpolations and diffusions, and returns them
    as lists of paths
    """
    res_path = Path("./data/array_simulation_results/ten_vaes/")
    interp_path = res_path / "interpolations" / exp_name
    if interp_path.exists():
        interps = list(interp_path.glob("*.csv"))
    else:
        print(f"Couldn't find interpolations for {exp_name}")
        interps = []

    diff_path = res_path / "diffusions" / exp_name
    if diff_path.exists():
        diffs = list(diff_path.glob("*.csv"))
    else:
        print(f"Couldn't find diffusions for {exp_name}")
        diffs = []

    return interps, diffs
