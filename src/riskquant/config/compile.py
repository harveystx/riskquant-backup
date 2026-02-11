# src/riskquant/config/compile.py
"""
Main API for compiling configurations from source data (e.g. .xlsx workbooks) 
into runtime config objects.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

from riskquant.config.configs import RiskStaticConfigFrames, RiskStaticConfig
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict


def compile_risk_static_config_from_xlsx(path: Union[str, Path]) -> RiskStaticConfig:
    """
    Load and compile a risk static config .xlsx workbook into a RiskStaticConfig 
    instance.

    Args:
        path: Path to the Excel (.xlsx) workbook containing the static config sheets.

    Returns:
        RiskStaticConfig: the compiled runtime configuration object.

    Raises:
        ValueError: if required sheets are missing or validation fails during
            construction/compilation of the config frames.
    """

    # Load config dataframes into tabs from the .xlsx file.
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=path)

    # Create the frames config.
    frames_config = RiskStaticConfigFrames(**dict_of_dfs)

    # Compile the config.
    compiled_config = frames_config.compile()

    # Return
    return compiled_config