# WS_SOLUTIONS_SPOT_INPUTS: Appendix A_GAP_MAP

Path: docs/workstreams/WS_SOLUTIONS_SPOT_INPUTS/A_GAP_MAP.md
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Used by: S1 to S10

| Capability | Core symbols the 12 files import | Solutions coverage | Gap type | Slice |
|---|---|---|---|---|
| Static config | `compile_risk_static_config_from_dict_of_dfs`, `RiskStaticConfig`, `audit_risk_static_config` | none | missing facade | S1 |
| Ingest, compile, mapper | `compile_raw_df_into_ingest_batch`, `build_uix_mapper`, `compile_portfolio_from_ingest_batch`, `IngestBatch`, `UixMapper`, `CompiledPortfolio` | `ValuationContext` accepts outputs | missing facade, leaked core type | S2 |
| Discounting mode | `DiscountingMode` | field type only | leaked core type | S2 |
| Dates | `date_to_ord32`, `ord_to_date` | none | missing facade | S2 |
| Level snapshot | `build_risk_factor_level_snapshot_from_dict`, `RiskFactorLevelSnapshot`, `extract_levels_from_snapshot` | none | missing facade | S3 |
| History matrix | `build_risk_factor_matrix_from_df`, `RiskFactorMatrix`, `ShiftType`, `SHIFT_LOG_RETURN_ID`, `ParametricRiskData`, `FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID` | run accepts type | missing facade, leaked core type | S4 |
| Proxy model | `NodeProxyTerm`, `compile_proxy_model`, `lower_axis_operator`, `world_request` | run accepts type | missing facade, leaked core type | S5, S9 |
| Run axis | none | `derive_run_factor_axis` | none | none |
| Run request | none | `parse_scenario_model_config`, `MethodologyDeclaration`, `ReportingSpec` | none | none |
| Taylor certification | `run_valuation`, `run_sensitivities`, `build_market_data_matrix_from_snapshot`, `build_universe_mapping`, `assemble_valuation_state`, `UniverseMapping`, `BlockSensitivities`, `ValuationResult` | adapters consume `BlockSensitivities` | missing facade | S8 |
| Scenario anatomy | `apply_linear_terms_to_base_levels`, `bind_term_columns`, `build_market_data_matrix_from_flat_levels`, `MarketDataMatrix`, `compute_var_es`, `NO_FX_RF_REQUIRED` | partial | owner gate O-10 | S9 |
| Reporting FX | `build_fx_conversion_plan`, `convert_scenario_pnl_to_reporting_currency`, `ReportingCurrencyPnl` | inside the run only | owner gate O-10 | S9 |
| Display | `riskquant.utils.printer` | none | owner gate O-13 | S10 |

First consumer path (SPOT, FULL_REVAL, no Taylor block):

| Input | Solutions entry today | Slice |
|---|---|---|
| Static config | none | S1 |
| Positions frame | none | S2 |
| Level snapshot | none | S3 |
| History matrix | none | S4 |
| Proxy terms | none | S5 |
| Run request | `parse_scenario_model_config`, `MethodologyDeclaration`, `ReportingSpec`, `derive_run_factor_axis` | none |
