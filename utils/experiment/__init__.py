"""
Some utils for computing all the arrays in experiment.py
"""
from pathlib import Path
from typing import Dict, Tuple, List, Union
from itertools import product
import json

import torch as t
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from geoml.discretized_manifold import DiscretizedManifold

from vae_models.vae_mario_obstacles import VAEWithObstacles
from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from vae_models.vae_zelda_hierachical import VAEZeldaHierarchical


def load_csv_as_arrays(
    path: Path, column="marioStatus"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes path to csv (result of simulating an array) and
    returns the zs alongside the mean of {column}. By default,
    it returns mean playability.
    """
    df = pd.read_csv(path, index_col=0)
    by_z = df.groupby("z")[column].mean()
    zs = np.array([json.loads(z) for z in by_z.index.values])
    vals = by_z.values

    return zs, vals


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

    decision_boundary = 0.5
    p = np.array([1.0 if r > decision_boundary else 0.0 for r in res[:, 1]])

    return bigger_grid, p


def load_arrays_as_map(zs: np.ndarray, p: np.ndarray) -> Dict[tuple, float]:
    return {(z[0], z[1]): p for z, p in zip(zs, p)}


def load_trace_as_map(path_to_trace: Path, n_iterations: int) -> Dict[tuple, float]:
    zs, p = load_trace_as_arrays(path_to_trace, n_iterations)
    return load_arrays_as_map(zs, p)


def load_csv_as_map(path: Path, column="marioStatus"):
    zs, p = load_csv_as_arrays(path, column=column)
    return load_arrays_as_map(zs, p)


def load_csv_as_grid(path: Path, column="marioStatus"):
    """..."""
    # first as map
    p_map = load_csv_as_map(path, column=column)
    return grid_from_map(p_map)


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


def build_discretized_manifold(
    p_map: Dict[tuple, float], vae_path: Path
) -> DiscretizedManifold:
    """
    Loads the VAE, adds obstacles to it
    according to the p_map and returns a discretized manifold.
    """
    vae = VAEWithObstacles()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))

    zs = np.array([z for z in p_map.keys()])
    p = np.array([p_ for p_ in p_map.values()])

    obstacles = t.from_numpy(zs[p != 1.0]).type(t.float)
    vae.update_obstacles(obstacles)
    grid = [t.linspace(-5, 5, 50), t.linspace(-5, 5, 50)]
    Mx, My = t.meshgrid(grid[0], grid[1])
    grid2 = t.cat((Mx.unsqueeze(0), My.unsqueeze(0)), dim=0)

    return DiscretizedManifold(vae, grid2, use_diagonals=True)


def load_experiment(path_to_array: Path, path_to_csv: Path) -> Tuple[np.ndarray]:
    """
    Loads zs, levels and in the appropiate order.
    """
    zs, p = load_csv_as_arrays(path_to_csv)

    original_array = np.load(path_to_array)
    zs_original = original_array["zs"]
    levels_original = original_array["levels"]

    # re-ordering the playabilities
    p_original = [p[zs.tolist().index(z.tolist())] for z in zs_original]
    return zs_original, p_original, levels_original


def intersection(
    char_1: Dict[tuple, float], char_2: Dict[tuple, float]
) -> Dict[tuple, float]:
    """
    Returns the intersection between two characteristic maps,
    assuming their keys are the same.
    """
    assert set(list(char_1.keys())) == set(list(char_2.keys()))

    res = {}
    for z, v1, v2 in zip(char_1.keys(), char_1.values(), char_2.values()):
        if v1 == v2 == 1.0:
            res[z] = 1.0
        else:
            res[z] = 0.0

    return res


def load_model(model_id: int = 0) -> VAEMarioHierarchical:
    model_name = f"vae_mario_hierarchical_id_{model_id}"
    vae = VAEMarioHierarchical()
    vae.load_state_dict(
        t.load(f"./trained_models/ten_vaes/{model_name}.pt", map_location=vae.device)
    )
    vae.eval()

    return vae


def load_model_from_path(
    vae_path: Path,
) -> Union[VAEMarioHierarchical, VAEZeldaHierarchical]:
    if "mario" in vae_path.stem:
        vae = VAEMarioHierarchical()
        vae.load_state_dict(t.load(vae_path, map_location=vae.device))
        vae.eval()
    elif "zelda" in vae_path.stem:
        vae = VAEZeldaHierarchical()
        vae.load_state_dict(t.load(vae_path, map_location=vae.device))
        vae.eval()
    else:
        raise ValueError(
            f"Unexpected vae path (w.o. mario or zelda) in {vae_path.stem}"
        )

    return vae
