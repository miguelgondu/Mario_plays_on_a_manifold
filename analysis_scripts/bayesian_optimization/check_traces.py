"""
Given traces of Vanilla, Constrained and Graph-based Bayesian Optimization,
this script loads them and checks metrics regarding nr. of playable levels
and best playable level.
"""

import numpy as np

if __name__ == "__main__":
    initial_trace = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps.npz"
    )["jumps"]
    initial_trace_graph = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps_from_graph.npz"
    )["jumps"]

    trace_names = [
        # "vanilla_bo",
        # "vanilla_bo_third",
        # "vanilla_bo_fourth",
        # "constrained_bo_third",
        # "restricted_bo",
        # "restricted_bo_third",
        # "restricted_bo_fourth",
        "restricted_bo_fifth",
        "vanilla_bo_EI"
    ]
    for trace_name in trace_names:
        if "restricted" in trace_name:
            initial_amount = len(initial_trace_graph)
        else:
            initial_amount = len(initial_trace)

        print("-" * 30, trace_name, "-" * 30)
        arr = np.load(f"./data/bayesian_optimization/traces/{trace_name}.npz")
        p = arr["playability"][initial_amount:]
        print(f"Playable percentage: {sum(p)}/{len(p)} ({sum(p)/len(p)})")
        valid_jumps = arr["jumps"][initial_amount:][p == 1]
        print(f"valid jumps max: {max(valid_jumps)}")
