from pathlib import Path
from typing import Tuple

import torch as t
import numpy as np
import networkx as nx

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

from geometries import DiscretizedGeometry

from utils.simulator.interface import test_level_from_int_tensor
from utils.experiment import load_csv_as_map

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# Run first samples
def run_first_samples(
    vae: VAEMarioHierarchical, n_samples: int = 10, force: bool = False
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    data_path = (
        ROOT_DIR
        / "data"
        / "bayesian_optimization"
        / "initial_traces"
        / "playability_and_jumps.npz"
    )
    if not force and data_path.exists():
        array = np.load(data_path)
        latent_codes = t.from_numpy(array["zs"])
        playability = t.from_numpy(array["playability"])
        jumps = t.from_numpy(array["jumps"])

        return latent_codes, playability, jumps

    latent_codes = 5.0 * vae.p_z.sample((n_samples,))
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for level in levels:
        results = test_level_from_int_tensor(level, visualize=True)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])

    # Saving the array
    np.savez(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps.npz",
        zs=latent_codes.detach().numpy(),
        playability=np.array(playability),
        jumps=np.array(jumps),
    )

    # Returning.
    return latent_codes, t.Tensor(playability), t.Tensor(jumps)


def run_first_samples_from_graph(
    vae: VAEMarioHierarchical, graph: nx.Graph, n_samples: int = 50, force: bool = False
):
    # Sample {n_samples} from graph indices, transform them to
    # latent coordinates and play them.

    ...


def load_geometry():
    """
    Loads a discretized geometry as a graph.
    """
    vae_path = Path("./trained_models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)

    dg = DiscretizedGeometry(p_map, "geometry_for_plotting_banner", vae_path)

    return dg
