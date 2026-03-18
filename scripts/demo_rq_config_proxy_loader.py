from typing import Final

from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_success_message,
)


if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # --- HEADING ---

    # Print the name of the calling file for clarity in the console output.
    print_calling_file()

    # --- LOAD THE PROXY CONFIG XLSX SHEET INTO A DATAFRAME ---

    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_CONFIG: Final[str] = (
        "dev_risk_base_proxy_config_v01.xlsx"
    )

    # Path to the .xlsx workbook containing the static config sheets.
    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_CONFIG

    # Load the config sheets into a dictionary of DataFrames.
    df = load_proxy_config_xlsx_into_df(path=xlsx_path)

    # Print.
    print_df(df)

    # --- FINISH ---

    # Print success message to confirm that the test ran without errors.
    print_success_message()
