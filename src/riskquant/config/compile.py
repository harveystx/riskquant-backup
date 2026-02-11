# src/riskquant/config/compile.py
"""Helpers to compile risk static configuration from Excel workbooks.

This module exposes a small convenience function used by notebooks and
scripts to load the risk static configuration workbook, construct the
`RiskStaticConfigFrames`, and compile it into a runtime
`RiskStaticConfig` instance.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

from riskquant.config.configs import RiskStaticConfigFrames, RiskStaticConfig
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict


def compile_risk_static_config_from_xlsx(path: Union[str, Path]) -> RiskStaticConfig:
    """Load and compile a risk static config workbook.

    This is a thin helper that mirrors the common notebook pattern:

        dfs = load_risk_static_data_config_xlsx_into_dict(path=...)
        frames = RiskStaticConfigFrames(**dfs)
        cfg = frames.compile()

    Args:
        path: Path to the Excel workbook containing the static config sheets.

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