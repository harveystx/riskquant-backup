# src/riskquant/config/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Final, Iterable, Mapping, Optional, Union

import pandas as pd


def write_config_dict_to_llm_txt(
    *,
    data: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
) -> None:
    """
    Serialize config DataFrames into an LLM-friendly text format.

    Design goals:
        - Deterministic
        - Token efficient
        - Easily parsable by LLMs
        - No dataframe truncation
        - Explicit structure boundaries
    """
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:

        for sheet_name, df in data.items():

            # Ensure stable column order
            df = df.copy(deep=False)

            f.write(f"=== SHEET: {sheet_name} ===\n")
            f.write(f"ROWS: {len(df)}\n")
            f.write("COLUMNS: " + ", ".join(map(str, df.columns)) + "\n\n")

            # CSV is extremely LLM-friendly.
            # Use na_rep="" instead of .fillna("") to safely handle strict dtypes like Int64.
            df.to_csv(
                f,
                index=False,
                lineterminator="\n",
                na_rep="",
            )

            f.write("\n=== END SHEET ===\n\n")


def load_risk_static_data_config_xlsx_into_dict(
    *,
    path: Union[str, Path],
    sheet_names: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
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
            raise ValueError(
                f"Missing required sheet(s) in '{path_str}': {missing}"
            )

        # One read call for all requested sheets
        dict_of_dfs = pd.read_excel(xls, sheet_name=sheets)

    # Validate that dfs is a mapping of sheet_name -> DataFrame
    if not isinstance(dict_of_dfs, Mapping):
        raise ValueError(
            "Expected multiple sheets to return a Mapping[str, DataFrame]."
        )

    # Clean the DataFrames to ensure consistent dtypes and memory layouts.
    clean_dict_of_dfs = {}
    for sheet_name, df in dict_of_dfs.items():
        # Convert dtypes: floats with NaNs become nullable Int64, objects become strings.
        # This prevents precision leaks and enforces tight memory layouts.
        clean_dict_of_dfs[sheet_name] = df.convert_dtypes()

    # Return a regular dict to avoid potential issues with pandas' internal dict subclass
    return clean_dict_of_dfs
