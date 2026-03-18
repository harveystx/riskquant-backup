# src/riskquant/config/proxy_config/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_proxy_config_xlsx_into_df(
    *,
    path: Union[str, Path],
    sheet_name: Optional[str] = "proxy_config",
) -> pd.DataFrame:
    """
    Load proxy configuration from Excel and return active rows only.
    """

    path_str = str(path)

    with pd.ExcelFile(path_str, engine="openpyxl") as xls:
        available = xls.sheet_names

        if sheet_name is not None and sheet_name not in available:
            raise ValueError(
                f"Missing required sheet '{sheet_name}' in '{path_str}'. "
                f"Available sheets: {available}"
            )

        df = pd.read_excel(xls, sheet_name=sheet_name)

    df = df.convert_dtypes()

    if "is_active" in df.columns:
        mask = df["is_active"].fillna(False).astype(bool)
        if mask.any():
            df = df[mask]
            df = df.drop(columns=["is_active"])
            df = df.reset_index(drop=True)

    return df
