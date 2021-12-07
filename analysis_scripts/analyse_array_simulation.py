import json
from pathlib import Path

import pandas as pd

all_levels = list(
    Path("./data/array_simulation_jsons").glob("ground_truth_another_vae_final_*.json")
)

rows = []
for i, path in enumerate(all_levels):
    print(f"{i}/{len(all_levels)}\r", flush=True)
    with open(path) as fp:
        row = json.load(fp)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("./data/array_simulation_results/ground_truth_another_vae.csv")
print(df)
