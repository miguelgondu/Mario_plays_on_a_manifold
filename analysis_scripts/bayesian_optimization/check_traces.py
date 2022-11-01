"""
Given traces of Vanilla, Constrained and Graph-based Bayesian Optimization,
this script loads them and checks metrics regarding nr. of playable levels
and best playable level.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    initial_trace = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps.npz"
    )["jumps"]
    initial_trace_graph = np.load(
        "./data/bayesian_optimization/initial_traces/playability_and_jumps_from_graph.npz"
    )["jumps"]

    # trace_names = [
    #     # "vanilla_bo",
    #     # "vanilla_bo_third",
    #     # "vanilla_bo_fourth",
    #     # "constrained_bo_third",
    #     # "restricted_bo",
    #     # "restricted_bo_third",
    #     # "restricted_bo_fourth",
    #     "restricted_bo_fifth",
    #     "restricted_bo_sixth",
    #     "restricted_bo_ucf",
    #     "vanilla_bo_EI",
    # ]
    trace_names = (
        []
        # [f"vanilla_bo_{i}" for i in range(10)]
        # + [f"constrained_bo_{i}" for i in range(10)]
        + [f"random_samples_1_{i}" for i in range(10)]
        + [f"vanilla_bo_1_{i}" for i in range(10)]
        + [f"restricted_bo_1_{i}" for i in range(10)]
    )
    rows = []
    rows_all_jumps = []
    for trace_name in trace_names:
        if "restricted" in trace_name:
            initial_amount = len(initial_trace_graph)
            name = "restricted"
        elif "random" in trace_name:
            initial_amount = 0
            name = "random"
        else:
            initial_amount = len(initial_trace)
            name = "vanilla"

        print("-" * 30, trace_name, "-" * 30)
        arr = np.load(f"./data/bayesian_optimization/traces/{trace_name}.npz")
        p = arr["playability"][initial_amount:]
        print(f"Playable percentage: {sum(p)}/{len(p)} ({sum(p)/len(p)})")
        valid_jumps = arr["jumps"][initial_amount:][p == 1]
        print(f"valid jumps max: {max(valid_jumps)}, position: {valid_jumps.argmax()}")

        for j in valid_jumps:
            rows_all_jumps.append({"experiment": name, "jumps": j})

        row = {
            "experiment": name,
            "percentage playable": sum(p) / len(p),
            "max valid jump": max(valid_jumps),
            "position of max": valid_jumps.argmax(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="experiment", y="max valid jump")
    plt.show()

    # df2 = pd.DataFrame(rows_all_jumps)
    # sns.violinplot(data=df2, x="experiment", y="jumps")
    # plt.show()
