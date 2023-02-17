"""
This script generates plots of completely
random levels for Chap. 7 of the dissertation.
"""

from pathlib import Path

import numpy as np

from utils.mario.plotting import save_level_from_array

DISSERTATION_PATH = Path("/Users/migd/Projects/dissertation")
FIG_PATH = DISSERTATION_PATH / "Figures" / "Chapter_7" / "manifold_hypothesis"

level = np.random.dirichlet([1.0] * (11), size=(14, 14)).argmax(axis=-1)
print(level)
print(level.shape)

save_level_from_array(FIG_PATH / "random_level.png", level)
