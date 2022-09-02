"""
Runs all arrays in the experiment
"""
from pathlib import Path

from utils.simulator.simulate_array import _simulate_array


def run_exp(exp_name: str, processes: int = 50):
    """
    Runs all arrays in ./data/arrays/ten_vaes/{diffusions and interpolations}/{exp_name}
    """
    arrays_path = Path("./data/arrays/ten_vaes/")
    arrays_diff = arrays_path / "diffusions" / exp_name
    arrays_interp = arrays_path / "interpolations" / exp_name

    for path in arrays_diff.glob("*.npz"):
        saving_path = Path(
            f"./data/array_simulation_results/ten_vaes/diffusions/{exp_name}"
        )
        if not (saving_path / f"{path.stem}.csv").exists():
            _simulate_array(
                path, processes, 5, exp_folder=f"ten_vaes/diffusions/{exp_name}"
            )
        else:
            print(f"There's already data at {saving_path}/{path.stem}.csv")
            continue

    for path in arrays_interp.glob("*.npz"):
        saving_path = Path(
            f"./data/array_simulation_results/ten_vaes/interpolations/{exp_name}"
        )
        if not (saving_path / f"{path.stem}.csv").exists():
            _simulate_array(
                path, processes, 5, exp_folder=f"ten_vaes/interpolations/{exp_name}"
            )
        else:
            print(f"There's already data at {saving_path}/{path.stem}.csv")
            continue


def run():
    # Running the ground truths
    # run_exp("baseline_strict_gt", processes=90)
    # run_exp("normal_strict_gt", processes=90)
    # run_exp("discrete_strict_gt", processes=90)
    # run_exp("continuous_strict_gt", processes=90)
    # run_exp("discretized_strict_gt", processes=60)

    # Discrete AL
    # for m in [100, 200, 300, 400, 500]:
    #     run_exp(f"discretized_strict_AL_{m}", processes=60)

    # Cont AL
    # for m in [100, 200, 300, 400, 500]:
    #     run_exp(f"continuous_AL_{m}", processes=90)

    # Discrete jump
    # run_exp("discrete_force_jump_gt", processes=90)
    # run_exp("discretized_force_jump_2_gt", processes=60)
    run_exp("baseline_force_jump_2_gt", processes=60)
    # run_exp("normal_force_jump_2_gt", processes=60)

    # baseline jump
    # run_exp("baseline_force_jump_gt", processes=90)

    # Continuous jump
    # run_exp("continuous_jump_gt", processes=90)


if __name__ == "__main__":
    run()
