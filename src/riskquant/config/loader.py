# riskquant/config/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, Mapping

import pandas as pd


def load_risk_static_data_config_xlsx(
    *,
    path: Union[str, Path],
    sheet_names: Optional[Iterable[str]] = None,
) -> Mapping[str, pd.DataFrame]:
    """
    Load selected sheets from a config workbook.

    Notebook-safe:
      - Ensures the Excel file handle is closed so the workbook can be saved/edited after load.
    Memory/CPU efficient:
      - Reads selected sheets in one call where possible.
      - Avoids unnecessary copies.
    """
    path_str = str(path)

    # Use a context manager to guarantee the file handle is closed
    with pd.ExcelFile(path_str, engine="openpyxl") as xls:
        available = xls.sheet_names

        if sheet_names is None:
            sheets = available  # reuse list
        else:
            # materialize once; also ensures list-like -> dict return
            sheets = tuple(sheet_names)

        missing = tuple(s for s in sheets if s not in available)
        if missing:
            raise ValueError(f"Missing required sheet(s) in '{path_str}': {missing}")

        # One read call for all requested sheets
        dict_of_dfs = pd.read_excel(xls, sheet_name=sheets)

    # Validate that dfs is a mapping of sheet_name -> DataFrame
    if not isinstance(dict_of_dfs, Mapping):
        raise ValueError(
            "Expected multiple sheets to return a Mapping[str, DataFrame]."
        )

    # Return
    return dict_of_dfs
