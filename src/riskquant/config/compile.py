# src/riskquant/config/compile.py
"""
Main API for compiling configurations from source data (e.g. .xlsx workbooks)
into runtime config objects.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path
from typing_extensions import Final

from riskquant.config.configs import RiskStaticConfig
from riskquant.config.builder import RiskStaticConfigFrames
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict
from riskquant.config.validate import validate_static_config_dfs
from riskquant.utils.printer import print_success_message


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

    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # Path to the .xlsx workbook containing the static config sheets.
    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG

    # Run the full compile process and print the resulting config object.
    compiled_config = compile_risk_static_config_from_xlsx(path=xlsx_path)

    # The compiled_config.summary() method should provide a concise overview of the config contents.
    print(compiled_config.summary())

    # Print success message and a summary of the compiled config.
    print_success_message()
