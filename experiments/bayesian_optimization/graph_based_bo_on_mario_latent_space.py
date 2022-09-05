"""
Loads one of the discretized geometries, builds
a discrete graph from it and runs graph-based
Bayesian Optimization.
"""
from pathlib import Path

import torch as t
import networkx as nx

import gpytorch

from botorch.acquisition import ExpectedImprovement

from geometries.discretized_geometry import DiscretizedGeometry

from utils.experiment import load_csv_as_map, load_model
from utils.experiment.bayesian_optimization import (
    load_geometry,
    run_first_samples,
    run_first_samples_from_graph,
)
from utils.gp_models.graph_gp import GraphBasedGP
from utils.simulator.interface import test_level_from_int_tensor

t.set_default_dtype(t.float64)


def bayesian_optimization_iteration(
    nodes: t.Tensor, jumps: t.Tensor, discretized_geometry: DiscretizedGeometry
):
    vae = load_model()
    graph = discretized_geometry.to_graph()
    adjacency_matrix = nx.adjacency_matrix(graph).todense().astype(float)
    graph_idxs = (
        t.Tensor(
            [
                discretized_geometry.graph_nodes.index((int(i.item()), int(j.item())))
                for (i, j) in nodes
            ]
        )
        .type(t.long)
        .unsqueeze(1)
    )

    # TODO: these nodes should be ints... so we need to implement
    # a transformation from R2 to the graph.
    graph_gp = GraphBasedGP(graph_idxs, jumps, adjacency_matrix)

    # Training the GP
    optimizer = t.optim.Adam(graph_gp.parameters(), lr=0.1)
    mll_graph = gpytorch.mlls.ExactMarginalLogLikelihood(graph_gp.likelihood, graph_gp)

    # Training the success model
    for i in range(10):
        optimizer.zero_grad()
        output = graph_gp(graph_idxs)
        loss = -mll_graph(output, jumps).mean()
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, 50, loss.item()))
        optimizer.step()

    graph_gp.eval()
    # There's something weird with the graph gp.
    graph_gp.num_outputs = 1
    graph_gp.posterior = graph_gp.__call__

    EI = ExpectedImprovement(graph_gp, max(jumps))

    # ooooh, how do we optimize the acquisition function?
    # In this case we don't have any combinatorial explotion.
    # We could just do a "grid search", evaluate in all nodes.
    all_nodes = t.Tensor(list(range(len(graph.nodes())))).view(-1, 1, 1).type(t.long)
    grid_search = EI(all_nodes)
    candidate = all_nodes[t.argmax(grid_search)]

    # candidate, _ = optimize_acqf(
    #     cEI,
    #     bounds=bounds,
    #     q=1,
    #     num_restarts=5,
    #     raw_samples=20,
    # )

    level = vae.decode(candidate).probs.argmax(dim=-1)
    results = test_level_from_int_tensor(level, visualize=True)

    # TODO: implement a visualization util that
    # swallows the graph nodes and arranges them on the grid.
    # if plot_latent_space:
    #     fig, ax = plt.subplots(1, 1)
    #     plot_prediction(model, ax)

    #     ax.scatter(latent_codes[:, 0], latent_codes[:, 1], c="black", marker="x")
    #     ax.scatter(candidate[:, 0], candidate[:, 1], c="red", marker="d")

    #     plt.show()
    #     plt.close(fig)

    return (
        candidate,
        t.Tensor([[results["jumpActionsPerformed"]]]),
        t.Tensor([results["marioStatus"]]),
    )


def run_experiment():
    print("Loading the model and geometry")
    vae = load_model()
    discretized_geoemtry = load_geometry()

    print("Populating the graph")
    discretized_geoemtry.to_graph()

    # Get some first samples and save them.
    print("Getting the first samples")
    latent_codes, playability, jumps = run_first_samples_from_graph(
        vae, discretized_geoemtry
    )
    jumps = jumps.type(t.float64).unsqueeze(1)

    print("Querying to graph ids:")
    nodes = discretized_geoemtry.from_latent_code_to_graph_node(latent_codes)

    # Initialize the GPR model for the predicted number
    # of jumps.
    print("BO iterations")
    for iteration in range(20):
        if (iteration + 1) % 5 == 0:
            plot_latent_space = True
        else:
            plot_latent_space = False

        candidate, jump, p = bayesian_optimization_iteration(
            nodes, jumps, discretized_geoemtry
        )
        print(f"tested {candidate} and got {jump}")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playability = t.hstack((playability, p))


if __name__ == "__main__":
    run_experiment()
