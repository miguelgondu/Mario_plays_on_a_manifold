"""
Runs all arrays in the experiment
"""
from pathlib import Path

from simulate_array import _simulate_array


def run(exp_name: str):
    """
    Runs all arrays in ./data/arrays/ten_vaes/{diffusions and interpolations}/{exp_name}
    """
    arrays_path = Path("./data/arrays/ten_vaes/")
    arrays_diff = arrays_path / "diffusions" / exp_name
    arrays_interp = arrays_path / "interpolations" / exp_name

    # TODO: finish implementing these
