import pandas as pd
import numpy as np


def build_table():
    initial_trace = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps.npz"
    )["jumps"]
    initial_trace_graph = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps_from_graph.npz"
    )["jumps"]

    print(f"initial trace length {len(initial_trace)}")
    print(f"initial trace graph length {len(initial_trace_graph)}")


if __name__ == "__main__":
    build_table()
