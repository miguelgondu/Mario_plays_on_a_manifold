"""
This script loads and analyses all 'baselines',
which are in the ../data/array_simulation_results/baselines/*.csv
"""
import random
from typing import List
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mario_utils.plotting import get_img_from_level


def get_mean_playability(experiment: List[Path]) -> float:
    """
    TODO: should I reduce by level first? We do it on the other
    computations I think.

    Which other computations?! This are the basis!
    """
    means = []
    for p in experiment:
        df = pd.read_csv(p)
        means.append(df["marioStatus"].mean())

    assert len(means) == 20
    return np.mean(means)


def get_all_levels(experiment: List[Path]) -> List[np.ndarray]:
    levels_as_arrays = []
    for p in experiment:
        df = pd.read_csv(p, index_col=0)
        levels = df["level"].values
        levels = [np.array(json.loads(level)) for level in levels]
        levels_as_arrays += levels

    return levels_as_arrays


def summarize(res: List[Path]):
    """
    Grabs a list of all the paths related to one dimension,
    and analyses them, returning mean playability for the
    linear interpolations and the two baseline diffusions.
    """
    row = {}
    linear_interpolations = filter(lambda x: "_linear_interpolation_" in x.name, res)
    row["linear interpolation mean"] = get_mean_playability(linear_interpolations)

    normal_diffusions = filter(lambda x: "_normal_diffusion_" in x.name, res)
    row["normal diffusions mean"] = get_mean_playability(normal_diffusions)

    baseline_diffusions = filter(lambda x: "_baseline_diffusion_" in x.name, res)
    row["baseline diffusions mean"] = get_mean_playability(baseline_diffusions)

    print(row)
    return row


def plot_random_levels(axes, levels):
    flat_axes = axes.flatten()
    for ax, lvl in zip(flat_axes, random.sample(levels, len(flat_axes))):
        img = get_img_from_level(lvl)
        ax.imshow(img)
        ax.axis("off")


def print_levels(res: List[Path], comment: str):
    linear_interpolations = filter(lambda x: "_linear_interpolation_" in x.name, res)
    levels_l = get_all_levels(linear_interpolations)

    normal_diffusions = filter(lambda x: "_normal_diffusion_" in x.name, res)
    levels_nd = get_all_levels(normal_diffusions)

    baseline_diffusions = filter(lambda x: "_baseline_diffusion_" in x.name, res)
    levels_bd = get_all_levels(baseline_diffusions)

    fig1, axes1 = plt.subplots(8, 8, figsize=(8 * 5, 8 * 5))
    plot_random_levels(axes1, levels_l)
    fig1.tight_layout()
    fig1.savefig(
        f"./data/plots/random_levels_in_baselines/{comment}_random_linear_interpolation.jpg"
    )

    fig2, axes2 = plt.subplots(8, 8, figsize=(8 * 5, 8 * 5))
    plot_random_levels(axes2, levels_nd)
    fig2.tight_layout()
    fig2.savefig(
        f"./data/plots/random_levels_in_baselines/{comment}_random_normal_diffusion.jpg"
    )

    fig3, axes3 = plt.subplots(8, 8, figsize=(8 * 5, 8 * 5))
    plot_random_levels(axes3, levels_bd)
    fig3.tight_layout()
    fig3.savefig(
        f"./data/plots/random_levels_in_baselines/{comment}_random_baseline_diffusion.jpg"
    )

    plt.close()


if __name__ == "__main__":
    results_ = list(Path("./data/array_simulation_results/baselines/").glob("*.csv"))
    res_2 = list(filter(lambda x: "_zdim_2_" in x.name, results_))
    res_8 = list(filter(lambda x: "_zdim_8_" in x.name, results_))
    res_32 = list(filter(lambda x: "_zdim_32_" in x.name, results_))
    res_64 = list(filter(lambda x: "_zdim_64_" in x.name, results_))

    rows = [
        {"z dim": 2, **summarize(res_2)},
        {"z dim": 8, **summarize(res_8)},
        {"z dim": 32, **summarize(res_32)},
        {"z dim": 64, **summarize(res_64)},
    ]

    print(pd.DataFrame(rows))

    print_levels(res_2, "zdim_2")
    print_levels(res_8, "zdim_8")
    print_levels(res_32, "zdim_32")
    print_levels(res_64, "zdim_64")
