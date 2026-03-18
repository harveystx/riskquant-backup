from typing import Dict, Final

import numpy as np
import pandas as pd

from riskquant.config.static_config.loader import (
    load_risk_static_data_config_xlsx_into_dict,
)
from riskquant.config.static_config.validate import validate_static_config_dfs
from riskquant.utils.printer import (
    print_success_message,
)


if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

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
