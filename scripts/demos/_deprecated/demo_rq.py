# riskquant/scripts/demo_rq.py

from datetime import date
from typing import Final

import pandas as pd

from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_xlsx,
)
from riskquant.config.static_config.rf_resolution import (
    RiskFactorAbsToRelNameIdMap,
)
from riskquant.constants import MOCK_VOLATILITY
from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_xlsx,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_xlsx,
)
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_dict,
    print_success_message,
    print_wrapped_text_box,
)


def run_demo():

    # ------------------------------------------------------
    # --- TURN OFF WARNINGS ---
    # ------------------------------------------------------

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # ------------------------------------------------------
    # --- HEADING ---
    # ------------------------------------------------------

    # Print the name of the calling file for clarity in the console output.
    print_calling_file()

    # ------------------------------------------------------
    # --- PATHS AND FILE NAMES ---
    # ------------------------------------------------------

    # Path to directory with demo files
    DEV_DATA_DIR: Final[str] = "C:/Users/mharvey/Documents/code/riskquant/data/dev"

    # Demo risk static config xlsx
    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # Demo base proxy config xlsx
    DEV_DATA_FILE_NAME_PROXY_CONFIG: Final[str] = "dev_risk_base_proxy_config_v01.xlsx"

    # Demo risk factor levels xlsx (history)
    DEV_DATA_FILE_NAME_RF_LEVELS_HIST: Final[str] = "dev_risk_factor_levels_v01.xlsx"

    # Demo risk factor levels xlsx (snapshot)
    DEV_DATA_FILE_NAME_RF_LEVELS_SNAP: Final[str] = (
        "dev_risk_factor_levels_snapshot_v01.xlsx"
    )

    # Demo positions data xlsx
    # DEV_DATA_FILE_NAME_POSITIONS: Final[str] = "dev_positions_v01.xlsx"
    DEV_DATA_FILE_NAME_POSITIONS: Final[str] = "dev_positions_test_v01.xlsx"

    # ------------------------------------------------------
    # --- AS-OF DATE ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box("AS-OF DATE")

    # Set an As-Of Date
    as_of_date = date(2026, 3, 1)

    # ------------------------------------------------------
    # --- COMPILE RISK STATIC CONFIG FROM THE STATIC CONFIG XLSX SHEET ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        "COMPILE RISK STATIC CONFIG FROM THE STATIC CONFIG XLSX SHEET"
    )

    # This single call loads, validates, and compiles the Excel file into pure Python objects.
    config = compile_risk_static_config_from_xlsx(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG
    )

    # ------------------------------------------------------
    # --- LOAD THE PROXY CONFIG XLSX SHEET INTO A DATAFRAME ---
    # ------------------------------------------------------
    # Heading.
    print_wrapped_text_box("LOAD THE PROXY CONFIG XLSX SHEET INTO A DATAFRAME")

    # Load the config sheets into a dictionary of DataFrames.
    df = load_proxy_config_xlsx_into_df(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_PROXY_CONFIG
    )

    # Print.
    print_df(df)

    # ------------------------------------------------------
    # --- GET RISK FACTOR MAPPING ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        f"RESOLVE RISK FACTORS FOR AS-OF DATE ({as_of_date}) AND GET MAPPING"
    )

    # Generate the mapping
    rf_mapping = RiskFactorAbsToRelNameIdMap.from_config(
        config=config,
        as_of_date=as_of_date,
    )

    # Display sample of the mapping
    print(f"\nMapping for as_of_date ({as_of_date}): abs -> (rel, id)\n ")
    print_dict(rf_mapping.abs_to_rel_name_and_id, limit=100)

    print(f"\nMapping for as_of_date ({as_of_date}): abs -> rel\n ")
    print_dict(rf_mapping.abs_to_rel_name, limit=100)

    print(f"\nMapping for as_of_date ({as_of_date}): abs -> id\n ")
    print_dict(rf_mapping.abs_to_rel_id, limit=100)

    # Get the mapping as a DataFrame for easy viewing
    mapping_df = rf_mapping.to_df()

    print(f"\nSuccessfully resolved {len(mapping_df)} absolute risk factors.")
    print("\nSample of resolved mappings (Absolute -> Relative + ID):")
    # Print the first 10 rows to show how the spot rolling logic resolved the names
    print(mapping_df.head(10).to_string())

    # Example of accessing the dictionary mappings directly
    sample_abs_name = mapping_df.index[0]
    resolved_rel_name = rf_mapping.abs_to_rel_name[sample_abs_name]
    resolved_id = rf_mapping.abs_to_rel_id[sample_abs_name]

    print("\nDictionary Access Example:")
    print(f"  Absolute Name: {sample_abs_name}")
    print(f"  Maps to Rel:   {resolved_rel_name}")
    print(f"  Maps to ID:    {resolved_id}")

    # Inverted mapping example
    print("\nInverted Mapping: rel -> (abs, id):")
    print_dict(rf_mapping.rel_to_abs_name_and_id, limit=100)

    # ------------------------------------------------------
    # --- LOAD IN HISTORICAL RISK FACTOR LEVELS ---
    # -------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        f"LOAD IN HISTORICAL RISK FACTOR LEVELS FOR AS-OF DATE ({as_of_date})"
    )

    # Print title.
    print("\n--- Loading in historical risk factor levels ---")

    # Load in the historical risk factor levels
    rfm = build_risk_factor_matrix_from_xlsx(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_RF_LEVELS_HIST,
        name_to_id_map=rf_mapping.abs_to_rel_id,
    )

    # Display basic info about the RFM.
    print("\nRFM shape:", rfm.shape)

    print("\nRFM Head (first 5 rows, transposed):")
    print_df(rfm.head(5).T)
    print("\nRFM Tail (last 5 rows, transposed):")
    print_df(rfm.tail(5).T)

    # ------------------------------------------------------
    # --- LOAD IN SNAPSHOT RISK FACTOR LEVELS ---
    # -------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        f"LOAD IN SNAPSHOT RISK FACTOR LEVELS FOR AS-OF DATE ({as_of_date})"
    )

    # Load in the snapshot risk factor levels
    rfs = build_risk_factor_level_snapshot_from_xlsx(
        as_of_date=as_of_date,
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_RF_LEVELS_SNAP,
        name_to_id_map=rf_mapping.abs_to_rel_id,
    )

    # As df.
    print("\nSnapshot as dataframe (transposed):")
    print_df(rfs.df.T)

    # As series.
    print("\nSnapshot as series:")
    print_df(rfs.sr)

    # ------------------------------------------------------
    # --- LOAD POSITIONS DATAFRAME ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        "LOAD IN THE POSITIONS DATAFRAME FROM THE XLSX FILE (AND RESOLVE RISK FACTORS IN THE PROCESS USING THE MAPPING WE GENERATED) "
    )

    # In a real environment, you might specify a sheet name.
    raw_df = pd.read_excel(DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_POSITIONS)

    # Display the raw DataFrame to verify it loaded correctly
    print("\nRaw Positions DataFrame:")
    print_df(raw_df, index=False, float_format="_.2f", zero_tol=1e-8)

    # ------------------------------------------------------
    # --- BUILD THE INGEST BATCH ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        "BUILD THE INGEST BATCH FROM THE RAW DATAFRAME USING THE RISK FACTOR MAPPING AND THE STATIC CONFIG"
    )

    # Build the IngestBatch
    print("\n--- Validating, Translating, and Building IngestBatch ---")
    batch = compile_raw_df_into_ingest_batch(
        df=raw_df,
        rf_name_to_rf_id=rf_mapping.abs_to_rel_id,
        container_name_to_container_id=config.container_name_to_container_id,
        rf_id_to_ccy=config.rf_id_to_ccy,
        container_id_to_ccy=config.container_id_to_ccy,
        container_name_to_y_axis_type_name=config.container_name_to_y_axis_type_name,
    )

    # Audit the output
    print("\n--- Auditing internal arrays via .to_df() ---")

    # Use your print_df utility to display the reconstructed DataFrame nicely
    print_df(batch.df, float_format="_.2f", zero_tol=1e-8)

    # ------------------------------------------------------
    # --- COMPILE PORTFOLIO ---
    # ------------------------------------------------------

    # Heading.
    print_wrapped_text_box(
        "COMPILE THE PORTFOLIO FROM THE INGEST BATCH AND AUDIT THE RUNTIME BLOCKS"
    )

    print("\n--- Compiling the Batch into CompiledPortfolio ---")
    portfolio = compile_portfolio_from_ingest_batch(batch)

    print("\n--- Compilation Audit ---")
    print(f"    - Raw Ingested Trades: {batch.size}")
    print(f"    - Compiled Trades:     {portfolio.total_trades}")
    print(f"    - Unique Dates Count:  {len(portfolio.unique_dates_ord32)}")

    # Fast-fail if we dropped any trades during the dispatch routing
    assert batch.size == portfolio.total_trades, (
        "CRITICAL: Trade count mismatch during compilation!"
    )

    print("\n--- Inspecting Compiled Blocks ---")

    p = print

    # Iterate dynamically over whatever blocks got created
    for bk in portfolio.iter_blocks():
        block_name = bk.__class__.__name__
        num_trades = len(bk.trade_uids)
        p(f"    - {block_name}: {num_trades} trades")

    # For blocks with many internal arrays, we can print those out too to verify the struct-of-arrays layout is correct.

    # Spot Block
    if portfolio.spot is not None:
        p("\n--- SpotBlock Internal Arrays (max 5) ---")
        bl = portfolio.spot
        bl.print_block()

    # Forward Node Block
    if portfolio.vanilla_forward_node is not None:
        p("\n--- ForwardNodeBlock Internal Arrays ---")
        bl = portfolio.vanilla_forward_node
        bl.print_block()

    # Forward Curve Block
    if portfolio.vanilla_forward_curve is not None:
        p("\n--- ForwardCurveBlock Internal Arrays ---")
        bl = portfolio.vanilla_forward_curve
        bl.print_block()

    # Vanilla European Option Blocks have many internal arrays worth inspecting

    # Vanilla European Option Node Block
    if portfolio.vanilla_european_option_node is not None:
        p("\n--- VanillaEuropeanOptionNodeBlock Internal Arrays ---")
        bl = portfolio.vanilla_european_option_node
        bl.print_block()

    # Vanilla European Option Curve Block
    if portfolio.vanilla_european_option_curve is not None:
        p("\n--- VanillaEuropeanOptionCurveBlock Internal Arrays ---")
        bl = portfolio.vanilla_european_option_curve
        bl.print_block()

    # ------------------------------------------------------
    # --- MOCK MARKET DATA MATRIX ---
    # ------------------------------------------------------
    print_wrapped_text_box("BUILD MOCK MARKET DATA MATRIX & RUN VALUATION")

    import numpy as np

    from riskquant.market_data.builder import build_market_data_matrix
    from riskquant.valuation.engine import run_valuation

    print("\n--- Sizing Mock Arrays ---")
    # Size arrays safely so any UIX lookup won't go out of bounds
    max_rf = config.max_risk_factor_id + 1
    max_container = config.num_containers + 1
    num_scenarios = 1  # Base scenario only for the demo

    # 1. Scalars & Nodes (Prices: 1_000.0, FX: 1.0, Vol: MOCK_VOLATILITY)
    raw_fx_spots = np.ones((max_rf, num_scenarios), dtype=np.float64)
    raw_prices = np.full((max_rf, num_scenarios), 1_000.0, dtype=np.float64)
    raw_forward_nodes = np.full((max_rf, num_scenarios), 1_000.0, dtype=np.float64)
    raw_vol_nodes = np.full(
        (max_rf, num_scenarios),
        MOCK_VOLATILITY,
        dtype=np.float64,
    )

    # 2. Curves (Discount Factors = 1.0 -> ln(DF) = 0.0)
    raw_discount_tenors = np.zeros((max_container, 1), dtype=np.int32)
    raw_discount_ln_dfs = np.zeros((max_container, 1, num_scenarios), dtype=np.float64)

    raw_price_curve_tenors = np.zeros((max_container, 1), dtype=np.int32)
    raw_price_curve_values = np.full(
        (max_container, 1, num_scenarios), 1_000.0, dtype=np.float64
    )

    curve_interp_methods = np.ones(max_container, dtype=np.int8)  # 1 = LINEAR
    curve_extrap_methods = np.ones(max_container, dtype=np.int8)  # 1 = FLAT

    # 3. Surfaces (Empty for now; pricers will fall back to MOCK_VOLATILITY)
    surface_interp_x_methods = np.ones(max_container, dtype=np.int8)
    surface_extrap_x_methods = np.ones(max_container, dtype=np.int8)
    surface_y_axis_type_ids = np.ones(max_container, dtype=np.int8)
    raw_vol_surface_tenors = np.empty((0, 0), dtype=np.int32)
    raw_vol_surface_strikes = np.empty((0, 0), dtype=np.float64)
    raw_vol_surface_values = np.empty((0, 0, 0, 0), dtype=np.float64)

    print("\n--- Building MarketDataMatrix ---")
    mdm = build_market_data_matrix(
        as_of_date=as_of_date,
        portfolio=portfolio,
        num_scenarios=num_scenarios,
        raw_fx_spots=raw_fx_spots,
        raw_prices=raw_prices,
        raw_forward_nodes=raw_forward_nodes,
        raw_vol_nodes=raw_vol_nodes,
        curve_interp_methods=curve_interp_methods,
        curve_extrap_methods=curve_extrap_methods,
        raw_discount_tenors=raw_discount_tenors,
        raw_discount_ln_dfs=raw_discount_ln_dfs,
        raw_price_curve_tenors=raw_price_curve_tenors,
        raw_price_curve_values=raw_price_curve_values,
        surface_interp_x_methods=surface_interp_x_methods,
        surface_extrap_x_methods=surface_extrap_x_methods,
        surface_y_axis_type_ids=surface_y_axis_type_ids,
        raw_vol_surface_tenors=raw_vol_surface_tenors,
        raw_vol_surface_strikes=raw_vol_surface_strikes,
        raw_vol_surface_values=raw_vol_surface_values,
    )

    # ------------------------------------------------------
    # --- RUN VALUATION KERNELS ---
    # ------------------------------------------------------
    print("\n--- Dispatched to Vectorized Valuation Kernels ---")
    # Time the valuation run
    import time

    t0 = time.perf_counter()

    pvs = run_valuation(
        portfolio=portfolio,
        valuation_state=mdm,
    )

    t1 = time.perf_counter()

    print(f"    - Valuation completed in {(t1 - t0) * 1000:.3f} ms.")
    print(f"    - Calculated PVs array shape: {pvs.shape}")

    print("\nSample PVs (First 100 trades):")
    print("\n".join([f"- {x:+18_.2f}" for x in pvs[:100].ravel()]))

    # ------------------------------------------------------
    # --- FINISH ---
    # ------------------------------------------------------
    print_success_message()


if __name__ == "__main__":
    # Run the compiled demo
    run_demo()
