"""
TODO: For each experiment get
- a p_map plot
- the approximated manifold (using ddg)
- example interpolations
- example diffusions.
"""
from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from geometries import BaselineGeometry, DiscretizedGeometry
from plotting_scripts.plot_before_and_after_calibrating import BIGGER_SIZE

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_mario_obstacles import VAEWithObstacles
from utils.experiment import grid_from_map, load_arrays_as_map, load_csv_as_map
from vae_zelda_hierachical import VAEZeldaHierarchical


def plot_metric_volume_in_3d(with_obstacles=True):
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)
    obstacles = grid_from_map(p_map)

    dg = DiscretizedGeometry(
        p_map,
        "geometry_for_plotting_animations",
        vae_path,
        with_obstacles=with_obstacles,
        force=not with_obstacles,
    )

    vae = VAEMarioHierarchical()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))
    vae.eval()

    zs_mv = dg.zs_of_metric_volumes
    mv = dg.metric_volumes
    map_mv = {tuple(z.tolist()): mv for z, mv in zip(zs_mv, mv)}
    calibrated = grid_from_map(map_mv)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, calibrated)
    ax.axis("off")

    name = "obstacles" if with_obstacles else "normal"

    for angle in range(0, 360):
        print(f"Plotting {name} in {angle}/360")
        ax.view_init(40, angle)
        fig.tight_layout()
        fig.savefig(f"./data/plots/metric_volume_surfaces/{name}_{angle:03d}.png")

    # rotate the axes and update
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(0.001)

    # ax.axis("off")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_metric_volume_in_3d(with_obstacles=True)
    plot_metric_volume_in_3d(with_obstacles=False)


# ffmpeg -framerate 18 -i obstacles_%03d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -crf 5 video_obstacles.mp4
# ffmpeg -framerate 18 -i normal_%03d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -crf 5 video_normal.mp4
