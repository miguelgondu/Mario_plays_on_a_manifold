"""
This script builds the table [ref: table:results:comparing_geometries]
"""
import numpy as np
import pandas as pd

index = ["discrete geometry full", "continuous geometry full", "baselines"]
data = np.zeros((len(index), 4))
table = pd.DataFrame(
    data,
    index=index,
    columns=[
        "E.Play. interpolation",
        "E.Play diffusion",
        "E.Div. interpolation",
        "E.Div. diffusion",
    ],
)

print(table.to_latex())
