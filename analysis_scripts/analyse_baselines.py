"""
This script loads and analyses all 'baselines',
which are in the ../data/array_simulation_results/baselines/*.csv
"""
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np


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
