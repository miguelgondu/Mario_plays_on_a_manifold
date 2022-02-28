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
    get_mean_w_conditioning,
    load_experiment_csv_paths,
)


def build_table_layout() -> pd.DataFrame:
    geometries = ["Ours", "Baseline", "Normal"]

    # index = []
    # for geometry in geometries:
    #     if geometry not in ["baselines", "normal"]:
    #         for m in [100, 200, 300, 400, 500]:
    #             index.append((geometry, m))

    #     index.append((geometry, "full"))
    # index = pd.MultiIndex.from_tuples(index, names=["Geometry", "AL queries"])
    index = geometries

    columns = [
        (r"$\mathbb{E}[\text{jumps}\,|\,l\text{ is playable}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{jumps}\,|\,l\text{ is playable}]$", "Random Walks"),
    ]
    columns = pd.MultiIndex.from_tuples(columns)
    data = np.zeros((len(index), len(columns)))
    table = pd.DataFrame(
        data,
        index=index,
        columns=columns,
    )

    print("Table layout:")
    print(table.to_latex(escape=False, float_format="%1.3f"))
    return table


def process_experiment(exp_name: str, processes: int = None):
    interps, diffs = load_experiment_csv_paths(exp_name)
    print(f"# of interpolations: {len(interps)}")
    print(f"# of diffusions: {len(diffs)}")

    mj_interps, sj_interps = get_mean_w_conditioning(interps)
    mj_diffs, sj_diffs = get_mean_w_conditioning(diffs)

    return {
        "i-jumps": f"{mj_interps:.2f}" + r"$\pm$" + f"{sj_interps:.2f}",
        "d-jumps": f"{mj_diffs:.2f}" + r"$\pm$" + f"{sj_diffs:.2f}",
    }


def parse_exp_name(exp_name: str) -> Tuple[str, str]:
    if "baseline" in exp_name:
        first = "Baseline"
    elif "discretized" in exp_name:
        first = "Ours"
    elif "normal" in exp_name:
        first = "Normal"
    else:
        raise ValueError(f"Unkown experiment {exp_name}")

    return first


def fill_out_experiment(table: pd.DataFrame, exp_name: str, processes: int = None):
    E_jumps = r"$\mathbb{E}[\text{jumps}\,|\,l\text{ is playable}]$"

    # Filling out baselines
    update = process_experiment(exp_name, processes=processes)
    multiindex = parse_exp_name(exp_name)
    table.loc[multiindex, (E_jumps, "Interpolation")] = update["i-jumps"]
    table.loc[multiindex, (E_jumps, "Random Walks")] = update["d-jumps"]

    # return table


def process():
    table = build_table_layout()

    print(table)
    print("baseline_force_jump_gt")
    fill_out_experiment(table, "baseline_force_jump_gt", processes=None)

    print("discrized_force_jump_gt")
    fill_out_experiment(table, "discretized_force_jump_gt", processes=None)

    print("normal_force_jump_gt")
    fill_out_experiment(table, "normal_force_jump_gt", processes=None)
    # print(table)
    # print("continuous_jump_gt")
    # fill_out_experiment(table, "continuous_jump_gt", processes=None)
    # print(table)
    # for m in [100, 200, 300, 400, 500]:
    #     print(f"discrete_AL_{m}")
    #     fill_out_experiment(table, f"discretized_force_jump_AL_{m}", processes=None)

    # for m in [100, 200, 300, 400, 500]:
    #     print(f"continuous_AL_{m}")
    #     fill_out_experiment(table, f"continuous_AL_{m}", processes=None)

    print(table.to_latex(escape=False))


if __name__ == "__main__":
    process()
