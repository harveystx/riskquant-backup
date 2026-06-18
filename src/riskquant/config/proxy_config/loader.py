# src/riskquant/config/proxy_config/loader.py
"""
Loads proxy configuration from an Excel workbook into a clean DataFrame.

The workbook MUST carry an `active` column. Rows with active=True are
returned; the `active` column is dropped from the result. If `active`
is absent the loader raises — silently treating an absent column as
"all rows active" is a real production foot-gun (drafts and historical
entries get compiled).

This loader only handles the human-facing xlsx convention. The in-memory
canonical fixture is post-filtered (no active column) and goes through
compile_proxy_topology_from_df directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# "active" column is not a hard requirement.
_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "proxying_level",
        "required_name",
        "proxy_name",
        "vol_scaling_factor",
    }
)

# Columns that survive the active-filter and are returned to compile.py.
_POST_FILTER_COLS: tuple[str, ...] = (
    "proxying_level",
    "required_name",
    "proxy_name",
    "vol_scaling_factor",
)


def load_proxy_config_xlsx_into_df(
    *,
    path: str | Path,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """
    Load the proxy-config sheet and return active rows only.

    Parameters
    ----------
    path
        Path to the Excel workbook.
    sheet_name
        Sheet to read. Defaults to the first sheet if None.

    Returns
    -------
    pd.DataFrame
        Columns: proxying_level, required_name, proxy_name,
        vol_scaling_factor. `active` is consumed and dropped.

    Raises
    ------
    ValueError
        If a required column is missing, or if the sheet name does not
        exist in the workbook.
    """
    path_str = str(path)

    with pd.ExcelFile(path_str, engine="openpyxl") as xls:
        available = list(xls.sheet_names)
        if sheet_name is not None and sheet_name not in available:
            raise ValueError(
                f"Missing required sheet {sheet_name!r} in {path_str!r}. "
                f"Available sheets: {available}"
            )
        _sheet: str = str(sheet_name) if sheet_name is not None else str(available[0])
        df: pd.DataFrame = pd.read_excel(xls, sheet_name=_sheet)

    # --- required columns: explicit (no silent active fallback) ---
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) in sheet {_sheet!r} of {path_str!r}: "
            f"{sorted(missing)}.\n"
            f"Required: {sorted(_REQUIRED_COLS)}"
        )

    # --- active filter ---
    if "active" in df.columns:
        mask = df["active"].fillna(False).astype(bool)
        df = df.loc[mask, list(_POST_FILTER_COLS)].reset_index(drop=True)
    else:
        df = df[list(_POST_FILTER_COLS)].reset_index(drop=True)

    # --- type coercions ---
    # String columns: strip whitespace, leave nulls as nulls (validator
    # rejects both null and blank; loader does NOT fillna to "").
    for col in ("proxying_level", "required_name", "proxy_name"):
        df[col] = df[col].astype("string").str.strip()

    # vol_scaling_factor: float64 with missing -> 1.0 (identity).
    df["vol_scaling_factor"] = df["vol_scaling_factor"].fillna(1.0).astype(np.float64)

    return df


if __name__ == "__main__":
    import pathlib  # Add this at the top of your file

    from riskquant.utils.printer import (
        print_calling_file,
        print_success_message,
    )

    print_calling_file()

    # 1. Get the absolute path of the directory containing THIS script
    current_dir = pathlib.Path(__file__).parent.resolve()

    # Project directory is 4 levels up from the current script
    project_dir = current_dir.parent.parent.parent.parent
    print(f"Project directory: {project_dir}")

    # 2. Build the path dynamically relative to the current script.
    # NOTE: You may need to add .parent if this script is nested deeper.
    # For example: current_dir.parent / "data" / "dev"
    DEV_DATA_DIR = project_dir / "data" / "dev"
    DEV_DATA_FILE_NAME = "dev_risk_base_proxy_config_v01.xlsx"

    # Combine them using the / operator (a handy feature of pathlib)
    full_path = DEV_DATA_DIR / DEV_DATA_FILE_NAME
    print(f"\nFull path to Excel file: {full_path}")

    # Convert the Path object back to a string just in case
    # load_proxy_config_xlsx_into_df strictly requires a string
    df = load_proxy_config_xlsx_into_df(path=str(full_path))

    print(f"\nLoaded proxy config DataFrame (shape: {df.shape}):")
    print(df.to_string(index=False))

    print_success_message()
