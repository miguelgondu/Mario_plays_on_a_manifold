"""
To check whether all the indexing is working, this script
loads and checks the discretized geometry's inner graph,
making sure we are properly building the graph.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from utils.experiment import load_model
from utils.experiment.bayesian_optimization import load_geometry
from utils.gp_models.graph_gp import GraphBasedGP

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
path_to_graph_laplacians = ROOT_DIR / "data" / "graph_laplacians"
path_to_graph_laplacians.mkdir(exist_ok=True)


def main():
    dg = load_geometry()

    graph = dg.to_graph()
    print(type(graph))

    latent_codes = t.cat(
        [dg.from_graph_node_to_latent_code(n).unsqueeze(0) for n in graph.nodes]
    )

    gp = GraphBasedGP(
        t.Tensor([100, 200]),
        t.Tensor([1.0, 1.0]),
        graph,
        path_to_laplacian=path_to_graph_laplacians / "example.npz",
        force_compute_laplacian=False,
    )
    node_idx = t.arange(len(dg.graph_nodes))

    gp.eval()
    predictions = gp(node_idx).mean

    _, (ax, ax2) = plt.subplots(1, 2)
    ax.scatter(
        latent_codes[:, 0].detach().numpy(),
        latent_codes[:, 1].detach().numpy(),
        c=predictions.detach().numpy(),
    )

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
