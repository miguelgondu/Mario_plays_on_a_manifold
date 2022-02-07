"""
Gets all arrays once ground truth and AL traces are computed.
"""

from pathlib import Path
from typing import Dict, Type

from experiment_utils import (
    build_discretized_manifold,
    load_csv_as_map,
    load_trace_as_map,
    intersection,
)
from geometry import (
    Geometry,
    BaselineGeometry,
    NormalGeometry,
    DiscreteGeometry,
    ContinuousGeometry,
)


def from_jumps_to_p(val_map: Dict[tuple, float]) -> Dict[tuple, float]:
    """
    Returns a characteristic map given by all the points
    where Mario doesn't jump.
    """
    return {z: 1.0 if jumps == 0 else 0.0 for z, jumps in val_map.items()}


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
            playable_map = load_csv_as_map(path_to_gt)
            strict_playability = {
                z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()
            }
            jump_map = load_csv_as_map(path_to_gt, column="jumpActionsPerformed")
            no_jump_map = {
                z: 1.0 if jumps == 0.0 else 0.0 for z, jumps in jump_map.items()
            }
            p_map = intersection(strict_playability, no_jump_map)
            if GeometryType == ContinuousGeometry:
                manifold = build_discretized_manifold(p_map, vae_path)
                gt_geometry = GeometryType(
                    p_map, f"{exp_name}_gt", vae_path, manifold=manifold
                )
            else:
                gt_geometry = GeometryType(p_map, f"{exp_name}_gt", vae_path)

            gt_geometry.save_arrays(force=force)

        # For multiple iterations in the AL trace
        if with_AL:
            if path_to_AL_trace.exists():
                for m in [100, 200, 300, 400, 500]:
                    p_map_m = load_trace_as_map(path_to_AL_trace, m)
                    if GeometryType == ContinuousGeometry:
                        manifold = build_discretized_manifold(p_map_m, vae_path)
                        AL_geometry_m = GeometryType(
                            p_map_m, f"{exp_name}_AL_{m}", vae_path, manifold=manifold
                        )
                    else:
                        AL_geometry_m = GeometryType(
                            p_map_m, f"{exp_name}_AL_{m}", vae_path
                        )
                    AL_geometry_m.save_arrays(force=force)


if __name__ == "__main__":
    # # For the baseline
    save_all_arrays("baseline_jump", BaselineGeometry, with_AL=False)
    # save_all_arrays("normal", NormalGeometry, with_AL=False)

    # # Discrete
    save_all_arrays("discrete_jump", DiscreteGeometry, with_AL=False)

    # Continuous
    # save_all_arrays("continuous", ContinuousGeometry)
