# src/riskquant/config/compile.py
"""
Main API for compiling configurations from source data (e.g. .xlsx workbooks)
into runtime config objects.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Union
from typing_extensions import Final

from riskquant.config.builder import RiskStaticConfigFrames
from riskquant.config.configs import RiskStaticConfig
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict
from riskquant.config.resolution import RiskFactorNameAbsToRelMapping
from riskquant.config.validate import validate_static_config_dfs
from riskquant.utils.dates import get_previous_business_date_as_date
from riskquant.utils.printer import (
    print_dict,
    print_success_message,
)


def compile_risk_static_config_from_xlsx(
    path: Union[str, Path],
) -> RiskStaticConfig:
    """
    Load, validate, and compile a risk static config .xlsx workbook into a
    RiskStaticConfig instance.

    Args:
        path: Path to the Excel (.xlsx) workbook containing the static config sheets.

    Returns:
        RiskStaticConfig: the compiled runtime configuration object.

    Raises:
        ValueError: if required sheets are missing, or validation fails on
            uniqueness/referential integrity constraints.
    """

    # 1. Load config dataframes into a dictionary (with optimized dtypes).
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=path)

    # 2. Vectorized fail-fast validation of raw DataFrames (C-level uniqueness & FK checks).
    validate_static_config_dfs(dict_of_dfs)

    # 3. Create the lightweight frames container (now stripped of heavy validation logic).
    frames_config = RiskStaticConfigFrames(**dict_of_dfs)

    # 4. Compile the DataFrames into our highly efficient, slot-based python structures.
    compiled_config = frames_config.compile()

    # 5. Return the compiled config object ready for use in runtime/model code.
    return compiled_config


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Setup paths
    # ---------------------------------------------------------
    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG

    # ---------------------------------------------------------
    # 2. Compile the Config
    # ---------------------------------------------------------
    print("\n--- Compiling Static Config ---")
    compiled_config = compile_risk_static_config_from_xlsx(path=xlsx_path)
    print(compiled_config.summary())

    # ---------------------------------------------------------
    # 3. Test Risk Factor Resolution for an As-Of Date
    # ---------------------------------------------------------
    as_of_date = get_previous_business_date_as_date()

    print(f"\n--- Resolving Risk Factors for As-Of Date: {as_of_date} ---")

    # Use the static method on the mapping class which calls the pure function in resolution.py
    mapping = RiskFactorNameAbsToRelMapping.from_config(
        config=compiled_config,
        as_of_date=as_of_date,
    )

    # Print a small sample of the resolved mapping to verify
    mapping_df = mapping.to_df()
    print(f"\nSuccessfully resolved {len(mapping_df)} absolute risk factors.")
    print("\nSample of resolved mappings (Absolute -> Relative):")
    print(mapping_df.to_string())

    # Property accessors for the same data.
    print("\n\nProperty accessors:")
    print("\n\nAbsolute to relative name mapping:")
    print_dict(mapping.abs_to_rf_name)
    print("\n\nAbsolute to relative ID mapping:")
    print_dict(mapping.abs_to_rf_id)
    print("\n\nRelative name to relative ID mapping:")
    print_dict(mapping.rf_name_to_rf_id)

    # ---------------------------------------------------------
    # 4. Success
    # ---------------------------------------------------------
    print_success_message()
