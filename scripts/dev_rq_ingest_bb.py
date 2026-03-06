import pandas as pd
from typing import Final
from riskquant.utils.printer import print_df, print_success_message
from riskquant.utils.dates import get_previous_business_date_as_date
from riskquant.config.compile import compile_risk_static_config_from_xlsx
from riskquant.portfolio.ingest.batch_builder import build_ingest_batch
from riskquant.config.resolution import RiskFactorAbsToRelNameIdMap
from riskquant.portfolio.ingest.dispatcher import iter_trades_from_batch


def run_batch_builder_demo():

    # --- INPUTS ---

    # Paths to data files (Update these to match your local environment)

    # Path to dev directory containing dev input data
    dev_data_dir: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    # Path to config file
    dev_config_file: Final[str] = "dev_risk_static_data_config_v01.xlsx"
    xlsx_path_dev_config_file = f"{dev_data_dir}/{dev_config_file}"

    # Path to position file
    dev_pos_file: Final[str] = "dev_positions_v01.xlsx"
    xlsx_path_dev_pos_file = f"{dev_data_dir}/{dev_pos_file}"

    # Set an As-Of Date
    as_of_date = get_previous_business_date_as_date()

    # --- LOAD CONFIG DATAFRAMES INTO A RISK STATIC CONFIG OBJECT ---

    # Print
    print(
        f"\n--- Loading and compiling config from: {xlsx_path_dev_config_file} ---"
    )

    # This single call loads, validates, and compiles the Excel file into pure Python objects.
    config = compile_risk_static_config_from_xlsx(
        path=xlsx_path_dev_config_file
    )

    # --- GET RISK FACTOR MAPPING ---

    # Print
    print(f"\n--- Resolving risk factors for As-Of Date: {as_of_date} ---")

    # Generate the mapping
    rf_mapping = RiskFactorAbsToRelNameIdMap.from_config(
        config=config,
        as_of_date=as_of_date,
    )

    # --- LOAD POSITIONS DATAFRAME ---

    print(f"\n--- Loading raw positions from: {xlsx_path_dev_pos_file} ---")

    # In a real environment, you might specify a sheet name.
    raw_df = pd.read_excel(xlsx_path_dev_pos_file)

    # --- BUILD THE INGEST BATCH ---

    # Build the IngestBatch
    print("\n--- Validating, Translating, and Building IngestBatch ---")
    batch = build_ingest_batch(
        df=raw_df,
        rf_name_to_rf_id=rf_mapping.abs_to_rel_id,
        container_name_to_container_id=config.container_name_to_container_id,
    )

    # --- AUDIT THE OUTPUT BATCH ---

    # Audit the output
    print("\n--- Auditing internal arrays via .to_df() ---")

    # Use your print_df utility to display the reconstructed DataFrame nicely
    print_df(batch.df, float_format="_.2f", zero_tol=1e-8)

    # --- DISPATCHER ---

    # ... append to the very bottom of the script ...
    print("\n--- Dispatching IngestBatch to Trade Builders ---")
    trades = list(iter_trades_from_batch(batch))

    print(f"Successfully compiled {len(trades)} TradeRecord objects!")

    # Let's inspect a complex trade to prove the routing worked
    print("\nSample Compiled Trade (Row 23 - Vanilla Option Node):")
    print(trades[23])

    # --- FINISH ---
    print_success_message()


if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    run_batch_builder_demo()
