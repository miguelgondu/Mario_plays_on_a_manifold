import pandas as pd

from sampling_n_points_at_random import models

if __name__ == "__main__":
    rows = []
    for z_dim, model_name in models.items():
        print(f"model name: {model_name}")
        df = pd.read_csv(
            f"./data/array_simulation_results/{model_name}.csv", index_col=0
        )
        p = df.groupby("z")["marioStatus"].mean()
        p[p > 0.0] = 1.0
        print(p.mean())
        row = {"z dim": z_dim, "mean playability (n=1000)": p.mean()}
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_latex(index=False))
