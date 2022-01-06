from typing import Tuple
import json

import numpy as np
import pandas as pd


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
