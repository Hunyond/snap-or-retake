import os
import sys

import numpy as np
import pandas as pd


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((df["pred"] - df["usability"]) ** 2)))


def main():
    folder = sys.argv[1]
    files = sorted(f for f in os.listdir(folder) if f.endswith("_val.csv"))
    if len(files) != 4:
        raise ValueError(f"Expected 4 validation files in {folder}, found {len(files)}")

    folds = []
    for f in files:
        data = pd.read_csv(os.path.join(folder, f))
        data = data[data["eval_ok"] == 1]
        folds.append(data)
        print(f"{f}: RMSE {rmse(data):.4f} on {len(data)}")

    all_folds = pd.concat(folds, ignore_index=True)
    print(f"out-of-fold RMSE {rmse(all_folds):.4f} on {len(all_folds)} photographs")


if __name__ == "__main__":
    main()