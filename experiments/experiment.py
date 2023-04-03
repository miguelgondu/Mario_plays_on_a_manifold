"""
Gets all arrays once ground truth and AL traces are computed.
"""

from pathlib import Path
from typing import Type

from utils.experiment import (
    build_discretized_manifold,
    load_csv_as_map,
    load_trace_as_map,
)
from geometries import (
    Geometry,
    BaselineGeometry,
    NormalGeometry,
    DiscreteGeometry,
    DiscretizedGeometry,
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
            mean_p_map = load_csv_as_map(path_to_gt)
            strict_p_map = {z: 1.0 if p == 1.0 else 0.0 for z, p in mean_p_map.items()}
            if GeometryType == ContinuousGeometry:
                manifold = build_discretized_manifold(strict_p_map, vae_path)
                gt_geometry = GeometryType(
                    strict_p_map, f"{exp_name}_gt", vae_path, manifold=manifold
                )
            else:
                gt_geometry = GeometryType(strict_p_map, f"{exp_name}_gt", vae_path)

            gt_geometry.save_arrays(force=force)

        # For multiple iterations in the AL trace
        if with_AL:
            if path_to_AL_trace.exists():
                for m in [100, 200, 300, 400, 500]:
                    p_map_m = load_trace_as_map(path_to_AL_trace, m)
                    strict_p_map_m = {
                        z: 1.0 if p == 1.0 else 0.0 for z, p in p_map_m.items()
                    }
                    if GeometryType == ContinuousGeometry:
                        manifold = build_discretized_manifold(p_map_m, vae_path)
                        AL_geometry_m = GeometryType(
                            strict_p_map_m,
                            f"{exp_name}_AL_{m}",
                            vae_path,
                            manifold=manifold,
                        )
                    else:
                        AL_geometry_m = GeometryType(
                            strict_p_map_m, f"{exp_name}_AL_{m}", vae_path
                        )
                    AL_geometry_m.save_arrays(force=force)


if __name__ == "__main__":
    # # For the baseline
    # save_all_arrays("baseline_strict", BaselineGeometry, with_AL=False)
    # save_all_arrays("normal_strict", NormalGeometry, with_AL=False)

    # # Discrete
    # save_all_arrays("discrete_strict", DiscreteGeometry, with_AL=False)

    # # Discretized
    save_all_arrays("discretized_strict", DiscretizedGeometry, with_AL=True)

    # Continuous
    # save_all_arrays("continuous_strict", ContinuousGeometry, with_AL=False)
