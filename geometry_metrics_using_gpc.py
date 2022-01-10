"""
This script saves the arrays for all geometric experiments,
for several dimensions.

See table [ref] in the research log.
"""
from typing import List
from itertools import product
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

from vae_mario_hierarchical import VAEMarioHierarchical
from train_vae import load_data

from interpolations.astar_gpc_interpolation import AStarGPCInterpolation
from diffusions.geometric_difussion import GeometricDifussion


from geometry_metrics_for_different_z_dims import get_random_pairs, models


def save_arrays_for_interpolation(trace_path: Path, gpc_kwargs: dict = {}):
    vae = VAEMarioHierarchical(z_dim=2)
    vae.load_state_dict(t.load(f"./models/{models[2]}.pt"))

    a = np.load(trace_path)
    zs, p = a["zs"], a["playability"]

    gpc = GaussianProcessClassifier(**gpc_kwargs)
    gpc.fit(zs, p)
    astar = AStarGPCInterpolation(10, gpc)

    # Saving interpolations
    # Only interpolate where playability is 1 according to
    # the AL exploration!
    zs_playable = t.from_numpy(zs[p == 1.0])
    zs1, zs2 = get_random_pairs(zs_playable, n_pairs=50)
    for line_i, (z1, z2) in enumerate(zip(zs1, zs2)):
        line = astar.interpolate(z1, z2)
        levels_in_line = vae.decode(line).probs.argmax(dim=-1)
        np.savez(
            f"./data/arrays/geometric/{models[2]}_astar_gpc_interpolation_{line_i:03d}.npz",
            zs=line.detach().numpy(),
            levels=levels_in_line.detach().numpy(),
        )


def save_arrays_for_diffusion(trace_path: Path, gpc_kwargs: dict = {}):
    # TODO: implement this
    pass
