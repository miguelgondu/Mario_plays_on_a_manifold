from pathlib import Path
from typing import Tuple

import numpy as np

from experiment_utils import load_csv_as_arrays


def load_and_order_results(csv_path: Path, array_path: Path, column: str) -> Tuple[np.ndarray]:
    """
    Gets the csvs and returns points, results and levels in the right order
    """
    zs, v = load_csv_as_arrays(csv_path, column=column)

    array = np.load(array_path)
    zs_in_order = array["zs"]
    levels = array["zs"]

    # Reordering v according to the original order in the arrays
    v = [v[zs.tolist().index(z.tolist())] for z in zs_in_order]

    return (zs_in_order, v, levels)
