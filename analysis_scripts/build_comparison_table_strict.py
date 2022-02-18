"""
This script builds the table [ref: table:results:comparing_geometries]
"""
from typing import Tuple

import numpy as np
import pandas as pd

from analysis_scripts.utils import (
    get_mean_playability,
    get_mean_diversities,
    load_experiment_csv_paths,
)


def build_table_layout() -> pd.DataFrame:
    geometries = ["discrete geometry", "continuous geometry", "baseline", "normal"]

    index = []
    for geometry in geometries:
        if geometry != "baselines" and geometry != "normal":
            for m in [100, 200, 300, 400, 500]:
                index.append((geometry, m))

        index.append((geometry, "full"))
    index = pd.MultiIndex.from_tuples(index, names=["Geometry", "AL queries"])

    columns = [
        (r"$\mathbb{E}[\text{playability}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{playability}]$", "Random Walks"),
        (r"$\mathbb{E}[\text{diversity}]$", "Interpolation"),
        (r"$\mathbb{E}[\text{diversity}]$", "Random Walks"),
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
    interps, diffs = load_experiment_csv_paths(exp_name)
    print(f"# of interpolations: {len(interps)}")
    print(f"# of diffusions: {len(diffs)}")

    mp_interps, sp_interps = get_mean_playability(
        interps, processes=processes, return_std=True
    )
    mp_diffs, sp_diffs = get_mean_playability(
        diffs, processes=processes, return_std=True
    )

    # TODO: filter by ID and get the stats there.
    # get interps and diffs one by one (maybe with .split("_"), maybe not necessary)
    # and then diversities = apply(get_all_diversities, each_one)
    # compute the means and vars from there.
    means_interp = []
    means_diff = []
    for i in range(10):
        interps_in_id = filter(lambda x: f"_id_{i}_" in x.name, interps)
        diffs_in_id = filter(lambda x: f"_id_{i}_" in x.name, diffs)

        diversities_in_interps = [
            get_mean_diversities([path]) for path in interps_in_id
        ]
        diversities_in_diffs = [get_mean_diversities([path]) for path in diffs_in_id]

        means_interp.append(np.mean(diversities_in_interps))
        means_diff.append(np.mean(diversities_in_diffs))

    md_interps, sd_interps = np.mean(means_interp), np.std(means_interp)
    md_diffs, sd_diffs = np.mean(means_diff), np.std(means_diff)

    # md_interps = get_mean_diversities(interps, processes=processes)
    # md_diffs = get_mean_diversities(diffs, processes=processes)

    return {
        "i-playability": f"{mp_interps:.2f}" + r"$\pm$" + f"{sp_interps:.2f}",
        "d-playability": f"{mp_diffs:.2f}" + r"$\pm$" + f"{sp_diffs:.2f}",
        "i-diversity": f"{md_interps:.2f}" + r"$\pm$" + f"{sd_interps:.2f}",
        "d-diversity": f"{md_diffs:.2f}" + r"$\pm$" + f"{sd_diffs:.2f}",
    }


def parse_exp_name(exp_name: str) -> Tuple[str, str]:
    if "baseline" in exp_name:
        first = "baselines"
    elif "discrete" in exp_name:
        first = "discrete geometry"
    elif "continuous" in exp_name:
        first = "continuous geometry"
    elif "normal" in exp_name:
        first = "normal"
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
    E_diversity = r"$\mathbb{E}[\text{diversity}]$"

    # Filling out baselines
    update = process_experiment(exp_name, processes=processes)
    multiindex = parse_exp_name(exp_name)
    table.loc[multiindex, (E_playability, "Interpolation")] = update["i-playability"]
    table.loc[multiindex, (E_diversity, "Interpolation")] = update["i-diversity"]
    table.loc[multiindex, (E_playability, "Random Walks")] = update["d-playability"]
    table.loc[multiindex, (E_diversity, "Random Walks")] = update["d-diversity"]

    # return table


def process():
    table = build_table_layout()

    print(table)
    print("baseline_strict_gt")
    fill_out_experiment(table, "baseline_strict_gt", processes=None)
    # print(table)
    print("discrete_strict_gt")
    fill_out_experiment(table, "discrete_strict_gt", processes=None)
    # print(table)
    print("continuous_strict_gt")
    fill_out_experiment(table, "continuous_strict_gt", processes=None)
    
    print("normal_strict_gt")
    fill_out_experiment(table, "normal_strict_gt", processes=None)
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
