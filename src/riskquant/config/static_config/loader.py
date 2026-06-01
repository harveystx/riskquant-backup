# src/riskquant/config/static_config/loader.py
"""
Thin xlsx loader for the static-config workbook.

The loader does not validate anything beyond Excel parseability; all
schema / semantic validation lives in `validate_static_config_dfs`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskquant.config.static_config.validate import REQUIRED_SHEETS


def load_risk_static_config_xlsx_into_dict(
    *,
    path: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Read every required sheet from an .xlsx workbook into a dict of DataFrames.

    Parameters
    ----------
    path
        Path to the Excel workbook.

    Returns
    -------
    dict[str, pd.DataFrame]
        Sheet-name -> DataFrame for every name in `REQUIRED_SHEETS`.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.
    ValueError
        If a required sheet is missing from the workbook.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workbook not found: {p}")

    excel = pd.ExcelFile(p)
    available: set[str] = {str(name) for name in excel.sheet_names}
    missing = REQUIRED_SHEETS - available
    if missing:
        raise ValueError(
            f"Workbook {p.name} is missing required sheet(s): {sorted(missing)}.\n"
            f"Sheets present: {sorted(available)}."
        )

    return {name: excel.parse(name) for name in REQUIRED_SHEETS}
