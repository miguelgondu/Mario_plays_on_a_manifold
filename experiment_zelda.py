"""
Computes the levels for Zelda, measures expected
grammar passing.
"""
from pathlib import Path

import torch as t
import numpy as np

from vae_zelda_hierachical import VAEZeldaHierarchical

from geometry import DiscretizedGeometry, Geometry
from grammar_zelda import grammar_check


def load_discretized_geometry(vae_path) -> DiscretizedGeometry:
    # vae_path = Path("./models/zelda/zelda_hierarchical_final.pt")
    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path))
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    n_rows = n_cols = 100
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    # zs = np.array([[a, b] for a, b in product(z1, z2)])
    # images_dist = vae.decode(t.from_numpy(zs).type(t.float))
    # images = images_dist.probs.argmax(dim=-1)
    # for level in images:
    #     print(grammar_check(level.detach().numpy()))

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }
    zs_in_positions = t.Tensor([z for z in positions.keys()]).type(t.float)
    levels = vae.decode(zs_in_positions).probs.argmax(dim=-1)
    p_map = {z: grammar_check(level) for z, level in zip(positions.keys(), levels)}

    ddg = DiscretizedGeometry(
        p_map,
        "zelda_discretized_grammar_gt",
        vae_path,
        beta=-5.5,
        n_grid=100,
        inner_steps_diff=30,
        x_lims=x_lims,
        y_lims=y_lims,
    )

    return ddg


if __name__ == "__main__":
    vae_path = Path("./models/zelda/zelda_hierarchical_final.pt")
    ddg = load_discretized_geometry(vae_path)

    ddg._get_arrays_for_interpolation()
