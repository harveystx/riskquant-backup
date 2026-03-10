from typing import Final

from riskquant.config.builder import RiskStaticConfigFrames
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict
from riskquant.config.validate import validate_static_config_dfs
from riskquant.utils.printer import print_success_message


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

    # 1. Load config dataframes into a dictionary (with optimized dtypes).
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=xlsx_path)

    # 2. Vectorized fail-fast validation of raw DataFrames (C-level uniqueness & FK checks).
    validate_static_config_dfs(dict_of_dfs)

    # 3. Create the lightweight frames container (now stripped of heavy validation logic).
    frames_config = RiskStaticConfigFrames(**dict_of_dfs)

    # 4. Compile the DataFrames into our highly efficient, slot-based python structures.
    compiled_config = frames_config.compile()

    # 5. Print summary of the compiled config to confirm successful compilation.
    print(compiled_config.summary())

    # 4. Print success message and a summary of the loaded frames config.
    print_success_message()
