"""
To check whether all the indexing is working, this script
loads and checks the discretized geometry's inner graph,
making sure we are properly building the graph.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t

import gpytorch

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

    optimizer = t.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.VariationalELBO(gp.likelihood, gp, gp.train_targets)

    # Training the success model
    # with settings.lazily_evaluate_kernels(False):
    for i in range(10):
        optimizer.zero_grad()
        output = gp(gp.train_inputs[0].type(t.long))
        loss = -mll(output, gp.train_targets).mean()
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, 50, loss.item()))
        optimizer.step()

    gp.eval()
    # gp.kernel._set_lengthscale(t.Tensor([10.0]))
    dist_ = gp(node_idx)
    predictions = dist_.mean
    stds = dist_.stddev

    _, (ax1, ax2) = plt.subplots(1, 2)
    plot = ax1.scatter(
        latent_codes[:, 0].detach().numpy(),
        latent_codes[:, 1].detach().numpy(),
        c=predictions.detach().numpy(),
    )
    ax2.scatter(
        latent_codes[:, 0].detach().numpy(),
        latent_codes[:, 1].detach().numpy(),
        c=stds.detach().numpy(),
    )
    plt.colorbar(plot)

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
