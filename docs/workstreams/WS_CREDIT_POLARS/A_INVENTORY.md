# WS_CREDIT_POLARS: Appendix A_INVENTORY: pandas uses at 19d097a

Path: docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md
Revision: 1
Snapshot: SNAPSHOT commit=19d097a dirty=false content_hash=c00f240898f32d74
Method: AST walk over every `.py` file of the src, scripts and lab profiles rebuilt in the investigator sandbox; `grep -rlE "^\s*(import pandas|from pandas)"` cross-check. `pd.` counts the attribute uses in the file. Lines are the import statements. Labels: Observed for every row.

## 1. src modules that import pandas (57)

| Module | Import lines | pd. | Classification | Slice |
|---|---|---|---|---|
| `riskquant/config/static_config/audit.py` | 19 | 12 | print-only (sort_values for display) | S2 |
| `riskquant/credit/exposure/add_on/ingest.py` | 18 | 1 | ingest entry signature; to_numpy casts; numpy argsort | S3 |
| `riskquant/credit/exposure/obligor_ead/ingest.py` | 21 | 1 | ingest entry signature; to_numpy casts; numpy argsort | S4 |
| `riskquant/credit/exposure/saccr/ingest.py` | 9 | 2 | ingest entry signature; to_numpy casts; numpy argsort | S3 |
| `riskquant/credit/jtd/ingest/_shared.py` | 13 | 2 | typed column extraction (to_numpy, read-only copy) | S5 |
| `riskquant/credit/jtd/ingest/builders/beta_binomial.py` | 7 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/forced_default.py` | 7 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/independent_bernoulli.py` | 7 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/marshall_olkin.py` | 5 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/mf_gaussian.py` | 10 | 2 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/mf_students_t.py` | 12 | 2 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/sf_gaussian.py` | 7 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/sf_students_t.py` | 8 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/builders/stress_regime_bernoulli.py` | 9 | 1 | frame reads only; sums are numpy (unchanged) | S5 |
| `riskquant/credit/jtd/ingest/bundle.py` | 22 | 15 | frame slots + from_instrument_frame (boolean filter, drop, reset_index) | S5 |
| `riskquant/credit/jtd/ingest/validate.py` | 25 | 16 | ADMISSION ARITHMETIC (groupby sum, line 613) + frame reads | S1, S5 |
| `riskquant/credit/masters/factor_universe/ingest.py` | 35 | 8 | ingest entry signature; stable sort on unique key; nullable Int astype/fillna | S4 |
| `riskquant/credit/masters/netting_set/ingest.py` | 22 | 4 | ingest entry signature; stable sort on unique key; nullable Int astype/fillna | S4 |
| `riskquant/credit/masters/obligors/ingest.py` | 25 | 10 | ingest entry signature; stable sort on unique key; nullable Int astype/fillna | S4 |
| `riskquant/ingest_fixtures/credit/credit_exposure/minimal_spot_with_netting_sets.py` | 20 | 1 | fixture frame assembly | S3 |
| `riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_data.py` | 15 | 7 | fixture frame assembly | S4 |
| `riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py` | 45 | 1 | fixture frame assembly | S4 |
| `riskquant/ingest_fixtures/credit/jtd/beta_binomial/minimal_beta_binomial_jtd.py` | 16 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/forced_default/minimal_forced_default_jtd.py` | 20 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/independent_bernoulli/minimal_independent_bernoulli_jtd.py` | 14 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/mf_copula/minimal_mf_copula_gaussian.py` | 20 | 2 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/mixed/minimal_mixed.py` | 25 | 5 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_gaussian.py` | 13 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_students_t_global_chi.py` | 17 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/jtd/stress_regime_bernoulli/minimal_stress_regime_bernoulli_jtd.py` | 15 | 1 | fixture frame assembly | S5 |
| `riskquant/ingest_fixtures/credit/netting_set/minimal_netting_set_master.py` | 20 | 1 | fixture frame assembly | S4 |
| `riskquant/ingest_fixtures/credit/obligors/minimal_obligors_master.py` | 20 | 1 | fixture frame assembly | S4 |
| `riskquant/ingest_fixtures/market_data/canonical_market_data_history.py` | 37 | 2 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/exhaustive.py` | 40 | 9 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/spot.py` | 45 | 1 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/vanilla_european_option_curve.py` | 41 | 1 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/vanilla_european_option_node.py` | 38 | 1 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/vanilla_forward_curve.py` | 46 | 1 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/vanilla_forward_node.py` | 26 | 1 | fixture frame assembly | S7 |
| `riskquant/ingest_fixtures/portfolio/vanilla_irs.py` | 8 | 1 | fixture frame assembly | S7 |
| `riskquant/market_data/risk_factors/builder_matrix.py` | 7, 8 | 8 | MARKET-RISK EDGE: entry signature, pd.to_datetime index, read_excel, pandas views | S7 |
| `riskquant/market_data/risk_factors/builder_snapshot.py` | 19 | 2 | MARKET-RISK EDGE: entry signature, pd.to_datetime index, read_excel, pandas views | S7 |
| `riskquant/market_data/risk_factors/matrix.py` | 8 | 10 | MARKET-RISK EDGE: entry signature, pd.to_datetime index, read_excel, pandas views | S7 |
| `riskquant/market_data/risk_factors/snapshot.py` | 9 | 9 | MARKET-RISK EDGE: entry signature, pd.to_datetime index, read_excel, pandas views | S7 |
| `riskquant/portfolio/ingest/batch.py` | 48, 255 | 0 | D096 pandas output adapter (TYPE_CHECKING + in-function import) | S2 |
| `riskquant/solutions/credit_exposure/golden_master_credit_exposure.py` | 22 | 8 | test-only in src (groupby sum as oracle) | S3 |
| `riskquant/solutions/credit_exposure/handoff.py` | 21 | 2 | public signature (attach_exposure_ead); frame copy | S3 |
| `riskquant/solutions/credit_exposure/recipes.py` | 34 | 27 | internal frame assembly; to_numpy casts; numpy argsort | S3 |
| `riskquant/solutions/credit_exposure/run.py` | 20 | 10 | internal frame assembly; to_numpy casts; numpy argsort | S3 |
| `riskquant/solutions/credit_exposure/schemas.py` | 36 | 9 | public signature (from_dataframe); pd.isna cells | S3 |
| `riskquant/solutions/credit_var/recipes.py` | 40 | 47 | internal frame assembly; pd.to_numeric(coerce) on validated rho; astype(str) | S4, S5 |
| `riskquant/solutions/credit_var/run.py` | 79 | 4 | internal frame assembly from request rows | S4, S5 |
| `riskquant/solutions/credit_var/schemas.py` | 92 | 4 | public signature (from_dataframe, exempt D106.1); pd.notna cells | S1, S6 |
| `riskquant/utils/dataframes.py` | 2, 3 | 0 | dead (no caller) | S2 |
| `riskquant/utils/dates.py` | 13, 32, 209 | 10 | in-function imports: Timestamp coercion (caller builder_matrix); CustomBusinessDay (no caller) | S7 |
| `riskquant/utils/printer.py` | 9 | 4 | print-only; pandas-typed signature (38 callers) | S2 |
| `riskquant_lab/fixtures/credit_exposure.py` | 31 | 2 | fixture frame assembly | S3 |

Count: 57.

## 2. src files that name pandas in comments or docstrings only (no import; no change needed)

`riskquant/credit/masters/cross_master/validate.py`, `riskquant/credit/masters/factor_universe/validate.py`, `riskquant/credit/masters/netting_set/validate.py`, `riskquant/credit/masters/obligors/validate.py`, `riskquant/dependence/_kernels/_kendall_tau_b.py`, `riskquant/dependence/estimators/kendall.py`, `riskquant/ingest_fixtures/config/risk_proxy_model/canonical_risk_proxy_model.py`, `riskquant/ingest_fixtures/config/risk_static_config/canonical_risk_static_config.py`, `riskquant/market_state/indices.py`, `riskquant/portfolio/ingest/canonicalize.py`, `riskquant/portfolio/ingest/compile.py`, `riskquant/portfolio/ingest/fx_resolver.py`, `riskquant/portfolio/ingest/materialize.py`, `riskquant/portfolio/ingest/pipeline.py`, `riskquant/portfolio/ingest/rf_resolver.py`, `riskquant/portfolio/ingest/utils.py`, `riskquant/portfolio/ingest/validate.py`, `riskquant/solutions/sensitivity_pnl/validate.py`, `riskquant/utils/business_days.py`.

Count: 19. `riskquant_lab/fixtures/credit_var.py` is in pandas_inventory.txt because of a variable named `pd`; it imports no pandas.

## 3. scripts that import pandas (35)

| Script | Import line | pd. | Calls a changed entry | Slice |
|---|---|---|---|---|
| `demos/credit/exposure/demo_e2e_cem_trade_to_exposure.py` | 17 | 1 | credit_exposure entries; print_df | S3 |
| `demos/credit/exposure/demo_e2e_saccr_trade_to_exposure.py` | 20 | 3 | credit_exposure entries; print_df | S3 |
| `demos/credit/jtd/e2e/demo_e2e_full_jtd_portfolio_core.py` | 18 | 6 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/e2e/demo_e2e_full_jtd_portfolio_mf_chi.py` | 18 | 6 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/e2e/demo_e2e_full_jtd_portfolio_mf_chi_streaming.py` | 19 | 6 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_asset_correlation.py` | 14 | 3 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_beta_binomial_jtd_engine.py` | 18 | 3 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_forced_default_engine.py` | 164 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_independent_bernoulli_engine.py` | 149 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_marshall_olkin_engine.py` | 118 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_sf_copula_jtd_engine_sf_gaussian.py` | 89 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_sf_copula_jtd_engine_sf_students_t.py` | 89 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/jtd/validation/demo_validate_stress_regime_bernoulli_engine.py` | 175 | 5 | JTD ingest, masters; print_df | S5 |
| `demos/credit/rc/demo_spot_rc_end_to_end.py` | 28 | 1 | IngestBatch.df; exhaustive fixture; print_dict_of_dfs | S2, S7 |
| `demos/e2e/03_spot_fx_and_reporting_currency.py` | 54 | 3 | risk_factors builders; print_df | S7 |
| `demos/e2e/_fixtures.py` | 46 | 4 | risk_factors builders; print_df | S7 |
| `demos/e2e/_spot_world.py` | 48 | 14 | risk_factors builders; print_df | S7 |
| `demos/e2e/demo_e2e_sensi_based_var.py` | 53 | 2 | risk_factors builders; print_df | S7 |
| `reporting/credit_risk/stress_tests/calc_deteministic_jtd.py` | 20 | 21 | JTD ingest, masters | S5 |
| `solutions/credit_var/demo_e2e_all_models.py` | 4 | 9 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_beta_binomial.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_beta_binomial_grouped.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_beta_binomial_params.py` | 3 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_forced_default.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_independent_bernoulli.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_mf_gaussian.py` | 4 | 7 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_mf_students_t.py` | 4 | 7 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_sf_gaussian.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_sf_gaussian_basel.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_sf_students_t.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_sf_students_t_basel.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_stress_regime_bernoulli.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_stress_regime_bernoulli_grouped.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `solutions/credit_var/demo_e2e_stress_regime_bernoulli_params.py` | 4 | 1 | from_dataframe; print_df | S6 |
| `tests/credit/jtd/test_ingest_bundle_validation.py` | 15 | 10 | JTD ingest | S5 |

Scripts that name pandas without an import: `demos/config/static_config/demo_full.py`, `demos/dependence/audit_dependence_core.py`, `demos/e2e/00_spot_getting_started.py`, `demos/e2e/07_spot_production_example.py`, `solutions/credit_var/bench_credit_var_api.py` (comments only).

## 4. Uses that can change a number or an order (Q2)

| Use | Today | Replacement | Bit-identical | Proof |
|---|---|---|---|---|
| `credit/jtd/ingest/validate.py:_semantic_mf_gaussian` line 613 | `pd.Series(sq).groupby(uid).sum()`: Kahan in row order (F-3) | C.5 function (Kahan loop, same state update) | yes, pandas 2.2.3 to 3.0.x: 0 of 14,467 groups differ (sandbox) | S1.T1 golden hex; S1.T2 edge cases |
| `builders/mf_gaussian.py` line 64, `mf_students_t.py` line 68 | `np.add.reduceat` over band order | none (numpy stays) | not applicable | S1.T3, S5.T2 |
| `masters/{obligors,netting_set,factor_universe}/ingest.py` `sort_values(key, kind="stable")` | stable sort on a unique key | `np.argsort(key, kind="stable")` or Polars `sort(maintain_order=True)` | yes: same permutation (sandbox: `[3, 1, 0, 2]` both) | S4.T3 |
| `masters/obligors/ingest.py` `astype("Int*").fillna(sentinel)` | null to sentinel; 3.5 raises | `fill_null(sentinel)` after an integral check, then cast | yes for integral input; 3.5 must raise by C.8 (Polars cast truncates, F-11) | S4.T2 |
| `factor_universe/ingest.py` nu, alpha, beta NaN to sentinel | `np.where(np.isnan(...))` after `to_numpy` | same; Polars null becomes NaN in `to_numpy` (F-12) | yes | S4.T2, S4.T3 |
| `credit_var/recipes.py:_per_obligor_rho` `pd.to_numeric(errors="coerce")` | no-op on a validated float-or-None column | `to_numpy()` of the `pl.Float64` column; `pl.Null` column means no explicit rho | yes | S4.T4, C.4 |
| `jtd/ingest/bundle.py:from_instrument_frame` boolean filter | row order kept | `filter(pl.col(instrument_col) == name)` keeps order | yes | S5.T2 |
| `credit_var/recipes.py:_mf_universe_and_bundle` `dict.fromkeys(astype(str))` | first-seen order | `to_list()` first-seen order | yes | S5.T2, C.4 |
| `ingest_fixtures/portfolio/exhaustive.py` `pd.concat` after `dropna(axis=1, how="all")` | union of columns, NaN fill | `pl.concat(how="diagonal")`, drop all-null columns; None fill (F-14) | values equal; missing cells None instead of NaN | S7.T4 |
| `market_data/risk_factors/builder_matrix.py` `pd.to_datetime(df.index)` | DatetimeIndex to the matrix dates | `pl.Date` column to `datetime64[D]` ordinals | to be proven in S7 | S7.T2 |
| `golden_master_credit_exposure.py` `groupby("netting_set_uid")["_g"].sum()` | test oracle with tolerance | `np.add.at` or `math.fsum` per netting set | not required (oracle) | the file's own `test_*` functions |
