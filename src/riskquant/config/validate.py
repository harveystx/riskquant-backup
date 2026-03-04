# src/riskquant/config/validate.py
"""
High-performance vectorized validation for risk static config data.

Design goals:
  - Fail-fast before memory allocation of runtime objects.
  - Zero native Python loops over dataframe rows.
  - Utilize NumPy/Pandas C-level vectorization for constraints (Unique, FK).
"""

from typing import Dict, Final

import numpy as np
import pandas as pd

from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict
from riskquant.utils.printer import (
    print_success_message,
)


def validate_static_config_dfs(dict_of_dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Executes vectorized schema checks (uniqueness and foreign keys) across the
    raw configuration dataframes.

    Args:
        dict_of_dfs: The raw loaded dataframes mapped by sheet name.

    Raises:
        ValueError: On the first violation of schema constraints.
    """

    # --------------------------------------------------------- #
    # 1. UNIQUENESS CONSTRAINTS (Primary Keys / Alt Keys)
    # --------------------------------------------------------- #
    # Structure: sheet_name -> tuple of column names that must be unique
    unique_constraints = {
        "container_meta": ("container_name", "container_name_id"),
        "risk_factors": ("risk_factor_name", "risk_factor_name_id"),
        "asset_classes": ("asset_class", "asset_class_id"),
        "asset_subclasses": ("asset_subclass", "asset_subclass_id"),
        "unit_types": ("unit_type_name", "unit_type_name_id"),
        "unit_scales": ("unit_scale_name", "unit_scale_name_id"),
        "ccy": ("ccy", "ccy_id"),
        "shift_types": ("shift_type", "shift_type_id"),
        "container_proxies": ("container_name",),
        "risk_factor_proxies": ("risk_factor_name",),
    }

    for sheet_name, cols in unique_constraints.items():
        df = dict_of_dfs.get(sheet_name)
        if df is None or df.empty:
            continue

        for col in cols:
            if col not in df.columns:
                continue  # Missing columns handled separately if strictly required

            # .is_unique is an O(N) C-level check.
            if not df[col].is_unique:
                # Only pay the cost of finding the exact duplicates if we fail
                duplicates = df[col][df[col].duplicated()].unique()
                raise ValueError(
                    f"Validation failed: '{sheet_name}.{col}' must be unique. "
                    f"Duplicates found: {duplicates.tolist()}"
                )

    # --------------------------------------------------------- #
    # 2. FOREIGN KEY CONSTRAINTS (Referential Integrity)
    # --------------------------------------------------------- #
    # Structure: tuple(child_sheet, child_col) -> tuple(parent_sheet, parent_col)
    fk_constraints = {
        ("risk_factors", "container_name"): (
            "container_meta",
            "container_name",
        ),
        ("asset_subclasses", "asset_class"): ("asset_classes", "asset_class"),
    }

    for (c_sheet, c_col), (p_sheet, p_col) in fk_constraints.items():
        c_df = dict_of_dfs.get(c_sheet)
        p_df = dict_of_dfs.get(p_sheet)

        if (
            c_df is None
            or p_df is None
            or c_col not in c_df.columns
            or p_col not in p_df.columns
        ):
            continue

        # Extract unique arrays directly. Dropping NA ensures we don't flag optional/blank FKs.
        child_vals = c_df[c_col].dropna().unique()
        parent_vals = p_df[p_col].dropna().unique()

        # np.setdiff1d is a highly optimized C-level array operation.
        # It is significantly faster and uses less memory than Python set(A) - set(B).
        missing_fks = np.setdiff1d(child_vals, parent_vals, assume_unique=True)

        if missing_fks.size > 0:
            raise ValueError(
                f"Validation failed: '{c_sheet}.{c_col}' contains values not present in "
                f"'{p_sheet}.{p_col}'. Missing: {missing_fks.tolist()}"
            )


if __name__ == "__main__":

    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # Path to the .xlsx workbook containing the static config sheets.
    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG

    # Load the config sheets into a dictionary of DataFrames.
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=xlsx_path)

    # Run the vectorized validations on the loaded dataframes.
    validate_static_config_dfs(dict_of_dfs)

    # If we reach this point without exceptions, all validations have passed.
    print_success_message()
