from pathlib import Path
from typing import Final

from riskquant.config.static_config.loader import (
    load_risk_static_data_config_xlsx_into_dict,
    write_config_dict_to_llm_txt,
)
from riskquant.utils.printer import print_as_list_str, print_success_message

if __name__ == "__main__":
    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    DEV_DATA_DIR: Final[str] = "C:/Users/mharvey/Documents/code/riskquant/data/dev"

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # Path to the .xlsx workbook containing the static config sheets.
    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG

    # Load the config sheets into a dictionary of DataFrames.
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=xlsx_path)

    # Write the loaded config dict to an LLM-friendly text file for inspection/debugging.
    write_config_dict_to_llm_txt(
        data=dict_of_dfs,
        output_path=Path(DEV_DATA_DIR) / "dev_risk_static_data_config_llm.txt",
    )

    # Print the loaded sheet names to confirm successful load.
    print_as_list_str(dict_of_dfs.keys())

    # Print success message to confirm that the test ran without errors.
    print_success_message()
