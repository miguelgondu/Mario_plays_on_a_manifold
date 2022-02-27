"""
Computes the levels for Zelda, measures expected
grammar passing.
"""
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from experiment_utils import load_arrays_as_map

from vae_zelda_hierachical import VAEZeldaHierarchical

from geometry import DiscretizedGeometry, Geometry
from grammar_zelda import grammar_check


def load_discretized_geometry(vae_path) -> DiscretizedGeometry:
    # This should be with Zelda obstacles? No, that's
    # being taken care of by the ddg itself.
    x_lims = (-4, 4)
    y_lims = (-4, 4)

    # TODO: replace this one for the grammar array
    array = np.load(f"./data/processed/zelda/grammar_checks/{vae_path.stem}.npz")
    zs = array["zs"]
    ps = array["playabilities"]
    p_map = load_arrays_as_map(zs, ps)
    # p_map = {tuple(z.tolist()): p for z, p in zip(zs, ps)}

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
    # for path_ in Path("./models/zelda").glob("zelda_hierarchical_final_*.pt"):
    for _id in [0, 3, 5, 6]:
        path_ = Path(f"./models/zelda/zelda_hierarchical_final_{_id}.pt")
        ddg = load_discretized_geometry(path_)
        interps = ddg._get_arrays_for_interpolation()
        # Plot grid and interps.
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        grid = ddg.grid
        ax.imshow(grid, extent=[-4, 4, -4, 4], cmap="Blues")

        means_interps = []
        for (interp, levels) in interps.values():
            ps = [grammar_check(level) for level in levels]
            means_interps.append(np.mean(ps))
            ax.plot(interp[:, 0], interp[:, 1])

        ax.axis("off")
        print(
            f"means for interpolations: {np.mean(means_interps)}, {np.std(means_interps)}"
        )

        plt.show()
        plt.close(fig)
