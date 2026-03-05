import datetime as dt
from pathlib import Path

# Config imports
from riskquant.config.compile import compile_risk_static_config_from_xlsx
from riskquant.config.loader import load_risk_static_data_config_xlsx_into_dict
from riskquant.config.validate import validate_static_config_dfs
from riskquant.config.builder import RiskStaticConfigFrames
from riskquant.config.resolution import RiskFactorNameAbsToRelMapping


# Utils imports
from riskquant.utils.dates import get_previous_business_date_as_date
from riskquant.utils.printer import (
    print_dict,
    print_success_message,
    print_wrapped_text_box,
)


def run_config_demo():
    # ---------------------------------------------------------
    # 0. Setup Paths (Update these to match your local environment)
    # ---------------------------------------------------------
    dev_data_dir = Path("C:/Users/mharvey/Documents/code/riskquant/data/dev")
    xlsx_path = dev_data_dir / "dev_risk_static_data_config_v01.xlsx"

    # Make sure the file exists before running the demo
    if not xlsx_path.exists():
        print(
            f"Warning: Could not find {xlsx_path}. "
            "Please update the path to run."
        )
        return

    # =========================================================
    # METHOD A: The High-Level "Easy Button" (compile.py)
    # =========================================================
    print_wrapped_text_box("METHOD A: High-Level Compilation")

    # This single call loads, validates, and compiles the Excel file into pure Python objects.
    config = compile_risk_static_config_from_xlsx(path=xlsx_path)

    print("\n--- Config Summary ---")
    print(config.summary())

    # =========================================================
    # METHOD B: The Granular Step-by-Step Approach
    # Useful if you ever load data from a database instead of Excel
    # =========================================================
    print_wrapped_text_box("METHOD B: Granular Step-by-Step Compilation")

    # 1. Load DataFrames
    print("1. Loading DataFrames...")
    dict_of_dfs = load_risk_static_data_config_xlsx_into_dict(path=xlsx_path)

    # 2. Vectorized Validation (Fail-fast uniqueness and FK checks)
    print("2. Validating DataFrames...")
    validate_static_config_dfs(dict_of_dfs)

    # 3. Create the transitional builder object
    print("3. Building Frames Container...")
    frames_config = RiskStaticConfigFrames(**dict_of_dfs)

    # 4. Compile into the lightweight runtime object
    print("4. Compiling Runtime Config...")
    config_from_steps = frames_config.compile()
    assert config_from_steps.summary() == config.summary()

    # =========================================================
    # DEMO 1: Accessing Runtime Configuration (configs.py)
    # =========================================================
    print_wrapped_text_box("DEMO 1: Interacting with the Compiled Config")

    # Let's pretend we are looking for a specific risk factor
    sample_rf_name = list(config.risk_factors_dict.keys())[0]
    rf_id = config.get_risk_factor_id(sample_rf_name)
    print(f"\nRisk Factor lookup: '{sample_rf_name}' -> ID: {rf_id}")

    # Accessing cached properties (computed lazily on first access)
    print("\nTriggering lazy caches...")
    total_cached_rf_names = len(config.rf_name_to_rf_id)
    total_cached_rf_objects = len(config.rf_id_to_risk_factor)
    print(f"Cached {total_cached_rf_names} name->ID mappings.")
    print(f"Cached {total_cached_rf_objects} ID->RiskFactor object mappings.")

    # Demonstrating lookup of container metadata
    sample_rf_obj = config.rf_id_to_risk_factor[rf_id]
    container_meta = config.container_meta_dict[sample_rf_obj.container_name]
    print(f"\nContainer Meta for '{sample_rf_obj.container_name}':")
    print(f"  - Asset Class: {container_meta.asset_class}")
    print(f"  - Dimension:   {container_meta.container_dim}")
    print(f"  - Shift Type:  {container_meta.shift_type_name}")

    # =========================================================
    # DEMO 2: Risk Factor Resolution (resolution.py)
    # Resolving relative names (e.g., "FWD_1M") to absolute names (e.g., "FWD_202604")
    # =========================================================
    print_wrapped_text_box("DEMO 2: Risk Factor Resolution (Spot Rolling)")

    # Set an As-Of Date
    as_of_date = get_previous_business_date_as_date()
    # Or hardcode for reproducible testing: as_of_date = dt.date(2026, 3, 5)

    print(f"Resolving risk factors for As-Of Date: {as_of_date}")

    # Generate the mapping
    mapping = RiskFactorNameAbsToRelMapping.from_config(
        config=config,
        as_of_date=as_of_date,
    )

    # Display sample of the mapping
    print(
        f"\nSample of absolute to relative name mapping for as_of_date={as_of_date}: \n "
    )
    print_dict(mapping.abs_to_rf_name_and_id, limit=100)

    # Get the mapping as a DataFrame for easy viewing
    mapping_df = mapping.to_df()

    print(f"\nSuccessfully resolved {len(mapping_df)} absolute risk factors.")
    print("\nSample of resolved mappings (Absolute -> Relative + ID):")
    # Print the first 10 rows to show how the spot rolling logic resolved the names
    print(mapping_df.head(10).to_string())

    # Example of accessing the dictionary mappings directly
    sample_abs_name = mapping_df.index[0]
    resolved_rel_name = mapping.abs_to_rf_name[sample_abs_name]
    resolved_id = mapping.abs_to_rf_id[sample_abs_name]

    print(f"\nDictionary Access Example:")
    print(f"  Absolute Name: {sample_abs_name}")
    print(f"  Maps to Rel:   {resolved_rel_name}")
    print(f"  Maps to ID:    {resolved_id}")

    # =========================================================
    # FINISH
    # =========================================================
    print_success_message()


if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    run_config_demo()
