"""
Computes the levels for Zelda, measures expected
grammar passing.
"""
from pathlib import Path
from typing import List, Dict
import pandas as pd

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment import load_arrays_as_map

from vae_models.vae_zelda_hierachical import VAEZeldaHierarchical

from geometries import BaselineGeometry, DiscretizedGeometry, Geometry, NormalGeometry
from utils.zelda.grammar import grammar_check


def load_grammar_p_map(vae_path) -> Dict[tuple, float]:
    # This should be with Zelda obstacles? No, that's
    # being taken care of by the ddg itself.
    x_lims = (-4, 4)
    y_lims = (-4, 4)

    # TODO: replace this one for the grammar array
    array = np.load(f"./data/processed/zelda/grammar_checks/{vae_path.stem}.npz")
    zs = array["zs"]
    ps = array["playabilities"]
    p_map = load_arrays_as_map(zs, ps)

    return p_map


def load_discretized_geometry(
    vae_path: Path, beta: float = -5.0, force: bool = False
) -> DiscretizedGeometry:
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
        exp_folder="zelda",
        beta=beta,
        n_grid=100,
        inner_steps_diff=30,
        x_lims=x_lims,
        y_lims=y_lims,
        force=force,
    )

    return ddg


def load_baseline_geometry(vae_path: Path) -> BaselineGeometry:
    p_map = load_grammar_p_map(vae_path)
    bg = BaselineGeometry(
        p_map, "zelda_baseline_grammar_gt", vae_path, exp_folder="zelda"
    )

    return bg


def load_normal_geometry(vae_path: Path) -> NormalGeometry:
    p_map = load_grammar_p_map(vae_path)
    ng = NormalGeometry(p_map, "zelda_normal_grammar_gt", vae_path, exp_folder="zelda")

    return ng


def experiment(geometry: Geometry, force: bool = False) -> None:
    """
    Saves results for a given geometry
    """
    geometry.save_arrays(force=force)
    interp_res_path = (
        Path("./data/array_simulation_results/zelda/interpolations") / geometry.exp_name
    )
    diff_res_path = (
        Path("./data/array_simulation_results/zelda/diffusions") / geometry.exp_name
    )

    interp_res_path.mkdir(exist_ok=True, parents=True)
    diff_res_path.mkdir(exist_ok=True, parents=True)

    for path_ in geometry.interpolation_path.glob("*.npz"):
        # Load and check for playability and diversity.
        array = np.load(path_)
        zs = array["zs"]
        levels = array["levels"]
        ps = np.array([grammar_check(level) for level in levels]).astype(int)

        np.savez(interp_res_path / f"{path_.stem}", zs=zs, levels=levels, ps=ps)

    for path_ in geometry.diffusion_path.glob("*.npz"):
        # Load and check for playability and diversity.
        array = np.load(path_)
        zs = array["zs"]
        levels = array["levels"]
        ps = np.array([grammar_check(level) for level in levels]).astype(int)

        np.savez(diff_res_path / f"{path_.stem}", zs=zs, levels=levels, ps=ps)


def discrete_geometry_experiment() -> pd.DataFrame:
    beta = -5.0
    for _id in [0, 3, 5, 6]:
        path_ = Path(f"./models/zelda/zelda_hierarchical_final_{_id}.pt")
        ddg = load_discretized_geometry(path_, beta=-5.0, force=True)
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

        fig.savefig(f"./data/plots/zelda/different_betas/{path_.stem}_beta_{beta}.png")
        # plt.show()
        plt.close(fig)


def baseline_experiment():
    for _id in [0, 3, 5, 6]:
        vae_path = Path(f"./models/zelda/zelda_hierarchical_final_{_id}.pt")
        bg = load_baseline_geometry(vae_path)
        interps = bg._get_arrays_for_interpolation()
        diffs = bg._get_arrays_for_diffusion()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 7, 1 * 7))
        grid = bg.grid
        ax1.imshow(grid, extent=[-4, 4, -4, 4], cmap="Blues")
        ax2.imshow(grid, extent=[-4, 4, -4, 4], cmap="Blues")

        means_interps = []
        for (interp, levels) in interps.values():
            ps = [grammar_check(level) for level in levels]
            means_interps.append(np.mean(ps))
            ax1.plot(interp[:, 0], interp[:, 1])

        print(
            f"means for interpolations: {np.mean(means_interps)}, {np.std(means_interps)}"
        )

        means_diffs = []
        for (diff, levels) in diffs.values():
            ps = [grammar_check(level) for level in levels]
            means_diffs.append(np.mean(ps))
            ax2.scatter(diff[:, 0], diff[:, 1], c=ps)

        print(f"means for diffusions: {np.mean(means_diffs)}, {np.std(means_diffs)}")
        fig.suptitle(f"{vae_path.stem} - baseline")


def normal_experiment():
    for _id in [0, 3, 5, 6]:
        vae_path = Path(f"./models/zelda/zelda_hierarchical_final_{_id}.pt")
        ng = load_normal_geometry(vae_path)
        interps = ng._get_arrays_for_interpolation()
        diffs = ng._get_arrays_for_diffusion()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 7, 1 * 7))
        grid = ng.grid
        ax1.imshow(grid, extent=[-4, 4, -4, 4], cmap="Blues")
        ax2.imshow(grid, extent=[-4, 4, -4, 4], cmap="Blues")

        means_interps = []
        for (interp, levels) in interps.values():
            ps = [grammar_check(level) for level in levels]
            means_interps.append(np.mean(ps))
            ax1.plot(interp[:, 0], interp[:, 1])

        print(
            f"means for interpolations: {np.mean(means_interps)}, {np.std(means_interps)}"
        )

        means_diffs = []
        for (diff, levels) in diffs.values():
            ps = [grammar_check(level) for level in levels]
            means_diffs.append(np.mean(ps))
            ax2.scatter(diff[:, 0], diff[:, 1], c=ps)

        print(f"means for diffusions: {np.mean(means_diffs)}, {np.std(means_diffs)}")
        fig.suptitle(f"{vae_path.stem} - baseline")


if __name__ == "__main__":
    # discrete_geometry_experiment()
    # baseline_experiment()
    # normal_experiment()
    # plt.show()
    for _id in [0, 3, 5, 6]:
        vae_path = Path(f"./models/zelda/zelda_hierarchical_final_{_id}.pt")
        ddg = load_discretized_geometry(vae_path)
        bg = load_baseline_geometry(vae_path)
        ng = load_normal_geometry(vae_path)

        experiment(ddg)
        experiment(bg)
        experiment(ng)
