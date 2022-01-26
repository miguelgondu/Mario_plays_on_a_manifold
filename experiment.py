"""
Gets all arrays once ground truth and AL traces are computed.
"""

from pathlib import Path
from typing import Type

from experiment_utils import load_csv_as_map, load_trace_as_map
from geometry import (
    Geometry,
    BaselineGeometry,
    NormalGeometry,
    DiscreteGeometry,
    ContinuousGeometry,
)


def save_all_arrays(
    exp_name: str, GeometryType: Type[Geometry], with_AL=True, force=False
):
    """
    Saves all arrays for a certain geometry
    """
    all_vae_paths = Path("./models/ten_vaes").glob("*.pt")
    for vae_path in all_vae_paths:
        model_name = vae_path.stem
        path_to_gt = (
            Path("./data/array_simulation_results/ten_vaes/ground_truth")
            / f"{model_name}.csv"
        )
        path_to_AL_trace = (
            Path("./data/evolution_traces/ten_vaes") / f"{model_name}.npz"
        )

        # For ground truth
        if path_to_gt.exists():
            p_map = load_csv_as_map(path_to_gt)
            gt_geometry = GeometryType(p_map, f"{exp_name}_gt", vae_path)
            gt_geometry.save_arrays(force=force)

        # For multiple iterations in the AL trace
        if with_AL:
            if path_to_AL_trace.exists():
                for m in [100, 200, 300, 400, 500]:
                    p_map_m = load_trace_as_map(path_to_AL_trace, m)
                    AL_geometry_m = GeometryType(
                        p_map_m, f"{exp_name}_AL_{m}", vae_path
                    )
                    AL_geometry_m.save_arrays(force=force)


if __name__ == "__main__":
    # For the baseline
    save_all_arrays("baseline", BaselineGeometry, with_AL=False)
    save_all_arrays("normal", NormalGeometry, with_AL=False)

    # Discrete
    save_all_arrays("discrete", DiscreteGeometry, force=True)

    # Continuous
    # save_all_arrays("continuous", ContinuousGeometry)
