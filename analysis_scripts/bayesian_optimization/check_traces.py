"""
Given traces of Vanilla, Constrained and Graph-based Bayesian Optimization,
this script loads them and checks metrics regarding nr. of playable levels
and best playable level.
"""
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.bayesian_optimization.restricted_domain_bo_on_mario_latent_space import (
    fitness_function,
)

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def load_traces(model_id: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the Bayesian optimization traces and returns
    a tuple of dataframes with all the content.

    Returns a tuple dataframe [max jumps, all jumps]
    """
    initial_trace_path = ROOT_DIR / "data" / "bayesian_optimization" / "initial_traces"
    initial_trace = np.load(
        initial_trace_path / f"playabilities_and_jumps_{model_id}.npz"
    )["jumps"]

    trace_names = (
        [f"random_samples_1_{i}" for i in range(20)]
        + [f"vanilla_bo_1_{i}" for i in range(20)]
        + [f"restricted_bo_1_{i}" for i in range(20)]
    )

    rows = []
    rows_all_jumps = []
    for trace_name in trace_names:
        if "restricted" in trace_name:
            initial_amount = len(initial_trace)
            name = "Restricted B.O."
        elif "random" in trace_name:
            initial_amount = 0
            name = "Random"
        else:
            initial_amount = len(initial_trace)
            name = "B.O."

        print("-" * 30, trace_name, "-" * 30)
        arr = np.load(f"./data/bayesian_optimization/traces/{trace_name}.npz")
        p = arr["playability"][initial_amount:]
        print(f"Playable percentage: {sum(p)}/{len(p)} ({sum(p)/len(p)})")
        valid_jumps_fitness_func = fitness_function(
            arr["jumps"][initial_amount:][p == 1]
        )
        print(
            f"valid jumps max: {max(valid_jumps_fitness_func)}, position: {valid_jumps_fitness_func.argmax()}"
        )

        for j in valid_jumps_fitness_func:
            rows_all_jumps.append({"experiment": name, "jumps": j})

        row = {
            "Experiment": name,
            "Avg. playabilities": p.sum() / len(p),
            "Max. jump": int(max(valid_jumps_fitness_func) * 10) // 2,
            "position of max": valid_jumps_fitness_func.argmax(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df_jumps = pd.DataFrame(rows_all_jumps)

    return (df, df_jumps)


if __name__ == "__main__":
    model_id = 1

    initial_trace_path = ROOT_DIR / "data" / "bayesian_optimization" / "initial_traces"
    initial_trace = np.load(
        initial_trace_path / f"playabilities_and_jumps_{model_id}.npz"
    )["jumps"]
    # initial_trace_graph = np.load(
    #     initial_trace_path / f"playability_and_jumps_from_graph_{model_id}.npz"
    # )["jumps"]

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
    #     "vanilla_bo_EI",  rm vanilla_bo_10.npz vanilla_bo_1_11.npz vanilla_bo_1_12.npz vanilla_bo_1_13.npz vanilla_bo_1_14.npz vanilla_bo_1_15.npz vanilla_bo_1_16.npz vanilla_bo_1_17.npz vanilla_bo_1_18.npz vanilla_bo_1_19.npz
    # ]
    trace_names = (
        []
        # [f"vanilla_bo_{i}" for i in range(10)]
        # + [f"constrained_bo_{i}" for i in range(10)]
        + [f"random_samples_1_{i}" for i in range(20)]
        + [f"vanilla_bo_1_{i}" for i in range(20)]
        + [f"restricted_bo_1_{i}" for i in range(20)]
    )
    rows = []
    rows_all_jumps = []
    for trace_name in trace_names:
        if "restricted" in trace_name:
            initial_amount = len(initial_trace)
            name = "Restricted B.O."
        elif "random" in trace_name:
            initial_amount = 0
            name = "Random"
        else:
            initial_amount = len(initial_trace)
            name = "Vanilla B.O."

        print("-" * 30, trace_name, "-" * 30)
        arr = np.load(f"./data/bayesian_optimization/traces/{trace_name}.npz")
        p = arr["playability"][initial_amount:]
        print(f"Playable percentage: {sum(p)}/{len(p)} ({sum(p)/len(p)})")
        valid_jumps_fitness_func = fitness_function(
            arr["jumps"][initial_amount:][p == 1]
        )
        print(
            f"valid jumps max: {max(valid_jumps_fitness_func)}, position: {valid_jumps_fitness_func.argmax()}"
        )

        for j in valid_jumps_fitness_func:
            rows_all_jumps.append({"Experiment": name, "jumps": j})

        row = {
            "Experiment": name,
            "Avg. playabilities": p.sum() / len(p),
            "Max. jump": int(max(valid_jumps_fitness_func) * 10) // 2,
            "position of max": valid_jumps_fitness_func.argmax(),
        }
        rows.append(row)

    _, (ax_jumps, ax_safety) = plt.subplots(1, 2)
    df = pd.DataFrame(rows)
    sns.stripplot(data=df, x="Experiment", y="Max. jump", ax=ax_jumps, size=10)
    sns.stripplot(
        data=df, x="Experiment", y="Avg. playabilities", ax=ax_safety, size=10
    )
    plt.show()

    # df2 = pd.DataFrame(rows_all_jumps)
    # sns.violinplot(data=df2, x="experiment", y="jumps")
    # plt.show()
