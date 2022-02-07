"""
This script builds the table [ref: table:results:comparing_geometries]
"""
from typing import Tuple

import numpy as np
import pandas as pd

from analysis_scripts.utils import (
    get_mean_playability,
    get_mean_diversities,
    get_mean,
    load_experiment,
)


def build_table_layout() -> pd.DataFrame:
    geometries = ["discrete geometry", "continuous geometry", "baselines"]

    index = []
    for geometry in geometries:
        if geometry != "baselines":
            for m in [100, 200, 300, 400, 500]:
                index.append((geometry, m))

        index.append((geometry, "full"))
    index = pd.MultiIndex.from_tuples(index, names=["Geometry", "AL queries"])

    columns = [
        (r"$\mathbb{E}[\text{playability}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{playability}]$", "Random Walks"),
        (r"$\mathbb{E}[\text{jumps}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{jumps}]$", "Random Walks"),
    ]
    columns = pd.MultiIndex.from_tuples(columns)
    data = np.zeros((len(index), 4))
    table = pd.DataFrame(
        data,
        index=index,
        columns=columns,
    )

    print("Table layout:")
    print(table.to_latex(escape=False, float_format="%1.3f"))
    return table


def process_experiment(exp_name: str, processes: int = None):
    interps, diffs = load_experiment(exp_name)
    print(f"# of interpolations: {len(interps)}")
    print(f"# of diffusions: {len(diffs)}")

    mp_interps, sp_interps = get_mean_playability(
        interps, processes=processes, return_std=True
    )
    mp_diffs, sp_diffs = get_mean_playability(
        diffs, processes=processes, return_std=True
    )

    mj_interps, sj_interps = get_mean(interps, "jumpActionsPerformed", return_std=True)
    mj_diffs, sj_diffs = get_mean(diffs, "jumpActionsPerformed", return_std=True)

    return {
        "i-playability": f"{mp_interps:.2f}" + r"$\pm$" + f"{sp_interps:.2f}",
        "d-playability": f"{mp_diffs:.2f}" + r"$\pm$" + f"{sp_diffs:.2f}",
        "i-jumps": f"{mj_interps:.2f}" + r"$\pm$" + f"{sj_interps:.2f}",
        "d-jumps": f"{mj_diffs:.2f}" + r"$\pm$" + f"{sj_diffs:.2f}",
    }


def parse_exp_name(exp_name: str) -> Tuple[str, str]:
    if "baseline" in exp_name:
        first = "baselines"
    elif "discrete" in exp_name:
        first = "discrete geometry"
    elif "continuous" in exp_name:
        first = "continuous geometry"
    else:
        raise ValueError(f"Unkown experiment {exp_name}")

    if "gt" in exp_name:
        second = "full"
    elif "AL" in exp_name:
        m = int(exp_name.split("_")[-1])
        second = m
    else:
        raise ValueError(f"Unkown experiment {exp_name}")

    return (first, second)


def fill_out_experiment(table: pd.DataFrame, exp_name: str, processes: int = None):
    E_playability = r"$\mathbb{E}[\text{playability}]$"
    E_jumps = r"$\mathbb{E}[\text{jumps}]$"

    # Filling out baselines
    update = process_experiment(exp_name, processes=processes)
    multiindex = parse_exp_name(exp_name)
    table.loc[multiindex, (E_playability, "Interpolation")] = update["i-playability"]
    table.loc[multiindex, (E_jumps, "Interpolation")] = update["i-jumps"]
    table.loc[multiindex, (E_playability, "Random Walks")] = update["d-playability"]
    table.loc[multiindex, (E_jumps, "Random Walks")] = update["d-jumps"]

    # return table


def process():
    table = build_table_layout()

    print(table)
    print("baseline_jump_gt")
    fill_out_experiment(table, "baseline_jump_gt", processes=None)
    # print(table)
    print("discrete_jump_gt")
    fill_out_experiment(table, "discrete_jump_gt", processes=None)
    # print(table)
    # print("continuous_gt")
    # fill_out_experiment(table, "continuous_gt", processes=None)
    # print(table)
    # for m in [100, 200, 300, 400, 500]:
    #     print(f"discrete_AL_{m}")
    #     fill_out_experiment(table, f"discrete_AL_{m}", processes=None)

    # for m in [100, 200, 300, 400, 500]:
    #     print(f"continuous_AL_{m}")
    #     fill_out_experiment(table, f"continuous_AL_{m}", processes=None)

    print(table.to_latex(escape=False))


if __name__ == "__main__":
    process()
