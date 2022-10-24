"""
This is a toy example in which, in a massive graph,
we try to find the node that has the highest degree
using GP based B.O.
"""

from matplotlib import pyplot as plt
import torch as t
import numpy as np
import networkx as nx

import gpytorch
from gpytorch import settings

from utils.gp_models.graph_gp import GraphBasedGP
from utils.visualization.graphs import visualize_graph_gp_in_ax

t.set_default_dtype(t.float64)


def bayesian_optimization_iteration(
    graph_idxs: t.Tensor, degrees: t.Tensor, graph: nx.Graph
):
    graph_based_gp = GraphBasedGP(
        graph_idxs,
        degrees,
        nx.adjacency_matrix(graph).toarray().astype(float),
    )

    # print(graph_based_gp)

    optimizer = t.optim.Adam(graph_based_gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.VariationalELBO(
        graph_based_gp.likelihood, graph_based_gp, graph_based_gp.train_targets
    )

    # Training the success model
    with settings.lazily_evaluate_kernels(False):
        for i in range(50):
            optimizer.zero_grad()
            output = graph_based_gp(*graph_based_gp.train_inputs)
            loss = -mll(output, graph_based_gp.train_targets).mean()
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, 50, loss.item()))
            optimizer.step()

        _, ax = plt.subplots(1, 1)
        visualize_graph_gp_in_ax(ax, graph_based_gp, graph)
        plt.show()


def run_experiment():
    graph = nx.erdos_renyi_graph(10, 0.5)
    graph_idxs = t.from_numpy(np.array(graph.nodes)).type(t.float64)
    function_to_predict = t.randn((len(graph_idxs),)).type(t.float64)
