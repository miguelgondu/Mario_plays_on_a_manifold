"""
Loads one of the discretized geometries, builds
a discrete graph from it and runs graph-based
Bayesian Optimization.
"""

from pathlib import Path
from geometries.discretized_geometry import DiscretizedGeometry
from utils.experiment import load_csv_as_map


def load_graph():
    # Hyperparameters
    vae_path = Path("./trained_models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)

    dg = DiscretizedGeometry(p_map, "geometry_for_plotting_banner", vae_path)

    return dg.to_graph()


if __name__ == "__main__":
    G = load_graph()
    print(G.edges)
    print("done!")
