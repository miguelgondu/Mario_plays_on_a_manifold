from pathlib import Path
from typing import Tuple

import torch as t
from torch.distributions import Uniform
import numpy as np
import networkx as nx

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical

from geometries import DiscretizedGeometry

from utils.simulator.interface import test_level_from_int_tensor
from utils.experiment import load_csv_as_map

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# Run first samples
def run_first_samples(
    vae: VAEMarioHierarchical,
    n_samples: int = 50,
    force: bool = False,
    save_path: Path = None,
    model_id: int = 0
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    if save_path is None:
        save_path = (
            ROOT_DIR
            / "data"
            / "bayesian_optimization"
            / "initial_traces"
            / f"playabilities_and_jumps_{model_id}.npz"
        )
    if not force and save_path.exists():
        array = np.load(save_path)
        latent_codes = t.from_numpy(array["zs"])
        playability = t.from_numpy(array["playability"])
        jumps = t.from_numpy(array["jumps"])

        return latent_codes, playability, jumps

    latent_codes = Uniform(t.Tensor([-5.0, -5.0]), t.Tensor([5.0, 5.0])).sample_n(
        n_samples
    )
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for i, level in enumerate(levels):
        results = test_level_from_int_tensor(level, visualize=True)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])
        print(
            "i:",
            i,
            "p:",
            results["marioStatus"],
            "jumps:",
            results["jumpActionsPerformed"],
        )

    # Saving the array
    np.savez(
        save_path,
        zs=latent_codes.cpu().detach().numpy(),
        playability=np.array(playability),
        jumps=np.array(jumps),
    )

    # Returning.
    return latent_codes, t.Tensor(playability), t.Tensor(jumps)


def run_first_samples_from_graph(
    vae: VAEMarioHierarchical,
    discretized_geometry: DiscretizedGeometry,
    n_samples: int = 50,
    force: bool = False,
    model_id: int = 0,
):
    data_path = (
        ROOT_DIR
        / "data"
        / "bayesian_optimization"
        / "initial_traces"
        / f"playability_and_jumps_from_graph_{model_id}.npz"
    )
    if not force and data_path.exists():
        array = np.load(data_path)
        latent_codes = t.from_numpy(array["zs"])
        playability = t.from_numpy(array["playability"])
        jumps = t.from_numpy(array["jumps"])

        return latent_codes, playability, jumps

    graph = discretized_geometry.to_graph()
    random_indexes = np.random.permutation(len(graph.nodes()))[:n_samples]
    random_nodes = [discretized_geometry.graph_nodes[idx] for idx in random_indexes]
    latent_codes = t.Tensor(
        [discretized_geometry.inverse_positions[node] for node in random_nodes]
    )
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for level in levels:
        results = test_level_from_int_tensor(level, visualize=True)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])

    # Saving the array
    np.savez(
        data_path,
        zs=latent_codes.cpu().detach().numpy(),
        playability=np.array(playability),
        jumps=np.array(jumps),
    )

    # Returning.
    return (
        latent_codes.to(vae.device),
        t.Tensor(playability).to(vae.device),
        t.Tensor(jumps).to(vae.device),
    )


def load_geometry(
    beta: float = -5.5,
    mean_scale: float = 1.0,
    name="geometry_for_plotting_banner",
    model_id: int = 0,
):
    """
    Loads a discretized geometry as a graph.
    """
    vae_path = Path(
        f"./trained_models/ten_vaes/vae_mario_hierarchical_id_{model_id}.pt"
    )
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)

    dg = DiscretizedGeometry(p_map, name, vae_path, beta=beta, mean_scale=mean_scale)

    return dg


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mean_scales = [0.5, 0.6, 0.7, 0.8, 1.0]
    _, axes = plt.subplots(1, len(mean_scales))
    for ax, mean_scale in zip(axes, mean_scales):
        dg = load_geometry(mean_scale=mean_scale, name=f"geometry_with_-5.5")
        ax.imshow(dg.grid)
        ax.set_title(r"$\lambda = " + f"{mean_scale}$")

    plt.show()
