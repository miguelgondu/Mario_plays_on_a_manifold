"""
Plots the latent space w. data from
the jumping experiment.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from utils.experiment import intersection, load_csv_as_map
from geometries import DiscretizedGeometry, NormalGeometry, BaselineGeometry
from plotting_scripts.utils import load_and_order_results

if __name__ == "__main__":
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    playable_map = load_csv_as_map(path_to_gt)
    strict_playability = {z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()}
    jump_map = load_csv_as_map(path_to_gt, column="jumpActionsPerformed")
    strict_jump_map = {z: 1.0 if jumps > 0.0 else 0.0 for z, jumps in jump_map.items()}
    p_map = intersection(strict_playability, strict_jump_map)

    ddg = DiscretizedGeometry(p_map, "discretized_jumping_plots", vae_path)
    bg = BaselineGeometry(p_map, "baseline_jumping_plots", vae_path)
    ng = NormalGeometry(p_map, "baseline_jumping_plots", vae_path)

    # Plotting the difference between the grid in one and the other
    fig1, (ax_b, ax_n, ax_dd) = plt.subplots(1, 3, figsize=(3 * 7, 1 * 7))

    ax_b.imshow(bg.grid, cmap="Blues")
    ax_n.imshow(ng.grid, cmap="Blues")
    ax_dd.imshow(ddg.grid, cmap="Blues")

    # Plotting some of the interpolations and diffusions, colored by jumps
    fig2, (ax_b, ax_n, ax_dd) = plt.subplots(1, 3, figsize=(3 * 7, 1 * 7))

    # csvs and arrays for baseline
    res_path_interps = Path(
        "./data/array_simulation_results/ten_vaes/interpolations/discretized_force_jump_gt"
    )
    array_path_interps = Path(
        "./data/arrays/ten_vaes/interpolations/discretized_force_jump_gt"
    )
    for p in filter(lambda x: "id_0" in x.name, res_path_interps.glob("*.csv")):
        model_name = p.stem
        array_p = array_path_interps / f"{model_name}.npz"

        try:
            zs, jumps, levels = load_and_order_results(
                p, array_p, "jumpActionsPerformed"
            )
            ax_b.plot(zs[:, 0], zs[:, 1], "-k")
            ax_b.scatter(zs[:, 0], zs[:, 1], c=jumps, cmap="inferno")
        except:
            print("Couldn't")

    plt.show()
