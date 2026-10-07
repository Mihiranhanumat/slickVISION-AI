"""Experiment bookkeeping: one row per experiment in artifacts/reports/experiment_matrix.csv."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def upsert_experiment(csv_path, name: str, values: Dict[str, Any]) -> pd.DataFrame:
    """Insert or update the row for experiment `name`, keeping columns written by other scripts."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if len(df) and "name" in df.columns and name in set(df["name"]):
        i = df.index[df["name"] == name][0]
        for k, v in values.items():
            if k not in df.columns:
                df[k] = pd.Series(dtype="object")
            df.loc[i, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([{"name": name, **values}])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df
