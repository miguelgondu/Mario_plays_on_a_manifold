"""
Some utils for computing all the arrays in experiment.py
"""
from pathlib import Path
from typing import Dict, Tuple, List
from itertools import product
import json

import torch as t
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Matern


def load_csv_as_arrays(path: Path) -> Tuple[np.ndarray]:
    """
    Takes path to csv (result of simulating an array) and
    returns the playability map.
    """
    df = pd.read_csv(path, index_col=0)
    by_z = df.groupby("z")["marioStatus"].mean()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    playabilities = by_z.values
    playabilities[playabilities > 0.0] = 1.0

    return zs, playabilities


def load_trace_as_arrays(path_to_trace: Path, n_iterations: int) -> Tuple[np.ndarray]:
    """
    Loads a trace, passes it through the GPC,
    predicts for the [-5, 5] square and returns
    the points and predictions.
    """
    a = np.load(path_to_trace)
    zs = a["zs"][: n_iterations + 100]
    p = a["playabilities"][: n_iterations + 100]

    kernel = 1.0 * Matern(nu=3 / 2) + 1.0 * WhiteKernel()
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(zs, p)

    z1s = np.linspace(-5, 5, 50)
    z2s = np.linspace(-5, 5, 50)

    bigger_grid = np.array([[z1, z2] for z1, z2 in product(z1s, z2s)])
    res = gpc.predict_proba(bigger_grid)

    decision_boundary = 0.75
    p = np.array([1.0 if r > decision_boundary else 0.0 for r in res[:, 1]])

    return bigger_grid, p


def load_arrays_as_map(zs: np.ndarray, p: np.ndarray) -> Dict[tuple, float]:
    return {(z[0], z[1]): p for z, p in zip(zs, p)}


def load_trace_as_map(path_to_trace: Path, n_iterations: int) -> Dict[tuple, float]:
    zs, p = load_trace_as_arrays(path_to_trace, n_iterations)
    return load_arrays_as_map(zs, p)


def load_csv_as_map(path: Path):
    zs, p = load_csv_as_arrays(path)
    return load_arrays_as_map(zs, p)


def positions_from_map(p_map: Dict[tuple, float]) -> Dict[tuple, tuple]:
    zs = np.array([z for z in p_map.keys()])
    z1s = np.array(sorted(list(set([z[0] for z in zs]))))
    z2s = np.array(sorted(list(set([z[1] for z in zs]))))

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(reversed(z2s))
    }

    return positions


def grid_from_map(p_map: Dict[tuple, float]) -> np.ndarray:
    zs = np.array([z for z in p_map.keys()])
    z1s = np.array(sorted(list(set([z[0] for z in zs]))))
    z2s = np.array(sorted(list(set([z[1] for z in zs]))))

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1s) for i, y in enumerate(reversed(z2s))
    }

    grid = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        grid[i, j] = p_map[z]

    return grid


def get_random_pairs(
    points: t.Tensor, n_pairs: int = 20, seed: int = None
) -> List[t.Tensor]:
    if seed is not None:
        np.random.seed(seed)
    idx1 = np.random.choice(len(points), size=n_pairs, replace=False)
    idx2 = np.random.choice(len(points), size=n_pairs, replace=False)
    while np.any(idx1 == idx2):
        idx2 = np.random.choice(len(points), size=n_pairs, replace=False)

    pairs_1 = points[idx1]
    pairs_2 = points[idx2]

    return pairs_1, pairs_2
