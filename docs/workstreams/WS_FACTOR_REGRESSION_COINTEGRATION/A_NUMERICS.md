# WS_FACTOR_REGRESSION_COINTEGRATION: Appendix A, numerics, constants and oracles

Path: docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/A_NUMERICS.md
Revision: 4
Status: FROZEN
Used by: S1 to S4
Label rule: every definition is Proposed. Every sandbox number is Observed (sandbox evidence). Every constant is
Observed from the source text named in A.2. In S2 the implementer compared all 24 cells of the six rows with the page
images (0 differences); the reviewer compared the PDF text layer and statsmodels 0.15.0 (0 differences).

## A.1 ADF regression (S3)

Input: aligned dates `d_0 < ... < d_{m-1}` (int32), combination values `z_0 .. z_{m-1}` (float64, C.2).
Declared: lag `p >= 0`, deterministic case CONSTANT or CONSTANT_TREND.

- Differences: `dz_t = z_t - z_{t-1}` for `t = 1 .. m-1`.
- Rows: `t = p+1 .. m-1`. Row count `n = m - 1 - p`. Row start date `d[t-1]`, row end date `d[t]`.
- Size check: after its type and dtype checks, `adf.py` checks `n < 20` first, before it builds the design and before any call of `fit_core`, and
  raises `CoreSampleSizeError(reason=codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM, n_available=n, n_required=20)` (C.6). `n` is zero or negative when `p >= m - 1`.
- Response: `dz_t`.
- Design columns, in this order: ones; trend `t - (p+1)` (CONSTANT_TREND only); `z_{t-1}`; `dz_{t-1} .. dz_{t-p}`.
  The trend origin does not change tau, because the ones column is present.
- Solve: `fit.fit_core(design, y, intercept=True, rank_rtol=<declared>, covariance_kind=codes.IID, covariance_lag=0)`.
- Statistic: `tau = coef[j] / se[j]`, `j` the column of `z_{t-1}` (1 for CONSTANT, 2 for CONSTANT_TREND).
  `se` uses `rss / (n - k)`, `k = 1 + trend + 1 + p`. This is the DF convention of MacKinnon (2010) section 3 for N = 1.
- `gamma = coef[j]`. The result reports `coef` and `se` of every column. It reports no p-value: the Student t
  p-values of `fit_core` are wrong for `gamma` and are not copied.
- IID covariance only: the lag augmentation absorbs serial correlation. HC0 or HAC changes the statistic and the
  tables do not apply. Heteroskedasticity does not change the asymptotic distribution (MacKinnon 2010, section 2, citing
  Phillips 1987).
- Calendar: the aligned index is the time index. A gap in the common calendar is one step.
- Stride and overlap: the test uses consecutive aligned dates. A weekly series is a separate alignment input, not a
  stride. The test never uses h-period overlapping differences, which add an MA(h-1) error.

## A.2 Critical values (S2)

Source: MacKinnon, J. G. (2010), "Critical Values for Cointegration Tests", Queen's Economics Department Working
Paper 1227, appendix Tables 2 and 3. URL: https://ageconsearch.umn.edu/record/273723/files/qed_wp_1227.pdf
Result text (`critical_value_source`), in full: CONSTANT gives "MacKinnon (2010) QED WP 1227, Table 2 tau_c, N=1, declared weights"; CONSTANT_TREND gives
"MacKinnon (2010) QED WP 1227, Table 3 tau_ct, N=1, declared weights". N=1 means one series: the combination is a single series whose weights the caller
declares and the test does not estimate.

`c(n) = b_inf + b1/n + b2/n^2 + b3/n^3`. An empty cell is 0. `n` is the number of observations in the test regression.

| Case | Level | b_inf | b1 | b2 | b3 |
|---|---|---|---|---|---|
| tau_c, N=1 (Table 2) | 0.01 | -3.43035 | -6.5393 | -16.786 | -79.433 |
| tau_c, N=1 | 0.05 | -2.86154 | -2.8903 | -4.234 | -40.040 |
| tau_c, N=1 | 0.10 | -2.56677 | -1.5384 | -2.809 | |
| tau_ct, N=1 (Table 3) | 0.01 | -3.95877 | -9.0531 | -28.428 | -134.155 |
| tau_ct, N=1 | 0.05 | -3.41049 | -4.3904 | -9.036 | -45.374 |
| tau_ct, N=1 | 0.10 | -3.12705 | -2.5856 | -3.925 | -22.380 |

Source conditions (Observed, same paper):
- The finite-n values are strictly valid for IID normal errors. With lags the use of `n` is a convention.
- tau_c assumes no drift in `z` under the null (West 1988). tau_ct is invariant to the drift.
- tau_nc needs highly unrealistic assumptions. v1 omits it (O-6).
- Simulated sample sizes are 20 to 1400. C.6 sets `n >= 20`.

Rows recorded for O-4 only (Engle-Granger, not v1). Disputed cells are marked.

| Case | Level | b_inf | b1 | b2 | b3 |
|---|---|---|---|---|---|
| tau_c, N=2 | 0.01 | -3.89644 | -10.9519 | -22.527 (statsmodels 0.15.0: -33.527) | |
| tau_c, N=2 | 0.05 | -3.33613 | -6.1101 | -6.823 | |
| tau_c, N=2 | 0.10 | -3.04445 | -4.2412 | -2.720 | |
| tau_c, N=3 | 0.01 | -4.29374 | -14.4354 | -33.195 | 47.433 |
| tau_c, N=3 | 0.05 | -3.74066 | -8.5631 (statsmodels 0.15.0: -8.5632) | -10.852 | 27.982 |
| tau_c, N=3 | 0.10 | -3.45218 | -6.2143 | -3.718 | |

EG note (Observed, section 3 of the paper): for N >= 2 the residual regression has no constant, and its dof
differs from the DF convention by the factor `((T-1)/(T-2))^(1/2)` (no trend). An EG slice must follow this exactly.

Pre-audit comparison (Observed, sandbox, 2026-10-19): the 23 N = 1 constants above equal the text layer of the PDF and
`statsmodels.tsa.adfvalues.tau_2010s` of statsmodels 0.15.0 (24 cells with the empty b3 cell, 0 differences). Nobody compared them with the
page images. The two disputed EG cells above reproduce (PDF text -22.527 and -8.5631; statsmodels -33.527 and -8.5632).

Verification plan: (1) the implementer types the 23 N = 1 constants from the PDF and checks six of them, one per row, against the page
images; (2) the reviewer types them again in the test file (S2.T1); (3) S2.T2 checks the rows against a Monte Carlo oracle; (4) the reviewer
repeats the comparison with `statsmodels.tsa.adfvalues.tau_2010s` in a sandbox and resolves every difference from the page image.

## A.3 Digests (S4)

`unit_root_spec_digest = digest_named_arrays(` in this order `)`:
`schema_version` int64 [1]; `engine_version` uint8 (unchanged `ENGINE_VERSION`, O-8); `test_kind` int8 [ADF code];
`terms` uint8 lines `T\x1f{name}\x1f{unit}` joined by `\x1e`; `weights` float64 `(n_terms,)`; `scale` int8;
`deterministic` int8; `lag` int64; `rank_rtol` float64; `window` int32 `[start_ord, end_ord]` taken from `AlignedLevels.start_ord` and
`AlignedLevels.end_ord`; `critical_value_source` uint8 (UTF-8 of the A.2 string of the declared case).
`unit_root_sample_digest`: `row_start_date_ord`, `row_end_date_ord` int32 `(n,)`; `column_names` uint8;
`y` float64 `(n,)`; `X` float64 `(n, k)` C-order. `row_digest`: the two date arrays, as in WS_FACTOR_REGRESSION A.6.

ADF column labels (fixed text). The intercept column is `const` (`api._INTERCEPT_NAME`). The trend column is `trend`. The column of `z_{t-1}` is
`z_lag1`. The columns of `dz_{t-1}` to `dz_{t-p}` are `dz_lag1` to `dz_lag<p>`, with the decimal lag number and no padding. The response is `dz`.
These labels name the columns in `UnitRootResult` and in every error that names a column (C.10).

`column_names` grammar (same bytes as `digest.sample_digest`). The lines are `Y\x1fdz`, then `C\x1f`, then one line `X\x1f{label}` for each design column
after the intercept, in design order (`trend` if CONSTANT_TREND, then `z_lag1`, then `dz_lag1` to `dz_lag<p>`), joined by `\x1e`, encoded as UTF-8 into a
`uint8` array with no trailing separator. The `terms` lines of the spec digest use the grammar above.

`UnitRootResult.window` is `(levels.start_ord, levels.end_ord)`, the pair of the spec digest. `UnitRootResult.exclusions` is `levels.exclusions`, the
alignment exclusions only (C.8). The first aligned dates `d_0 .. d_p` are not added to it.

## A.4 Null hypothesis text (S4)

- CONSTANT: "H0: the combination has a unit root (gamma = 0) and no drift. H1: it is stationary around a constant."
- CONSTANT_TREND: "H0: the combination has a unit root (gamma = 0). H1: it is stationary around a linear trend."
The result states the hypothesis. It states no decision.

## A.5 Sandbox evidence (Python 3.12.3, numpy 2.4.4, scipy 1.17.1, statsmodels 0.15.0)

| Check | Setup | Result |
|---|---|---|
| Constants against simulation | random walks, lag 0, tau_c, 100000 replications at n = 100, 250, 1000 | 5 % and 10 %: within 0.006 of `c(n)`; 1 %: within 0.018 |
| S2.T2 tolerance | 200000 replications, n = 100, seeds 1 to 3, tau_c and tau_ct | max deviation 0.0093; 1.1 s to 1.3 s per seed |
| ADF statistic | own normal-equation ADF against `adfuller(autolag=None)` on 600 points: (c, 0), (c, 4), (ct, 4) | equal to about 1e-14 relative |
| Power, tau_c 5 %, lag 0 | AR(1) spread with half-life H days, 4000 replications | n=250: H5 1.00, H20 0.26, H60 0.09. n=500: H20 0.73, H60 0.15. n=1000: H20 1.00, H60 0.42, H120 0.15. n=2500: H60 0.99, H120 0.58, H250 0.18 |
| Span against frequency | same 1000-day span, H60, daily (n=1000) against every 5th date (n=200) | power 0.41 against 0.40 |
| Size against lag | unit root, MA(1) theta, n=500, nominal 5 % | theta 0: 0.040 to 0.050 for p 0 to 24. theta -0.5: p0 0.629, p1 0.224, p4 0.056, p12 0.045. theta -0.8: p0 0.999, p4 0.411, p12 0.068, p24 0.040 |
| Levels HAC t-test | `y = x + u`, x random walk, u AR(1) 0.9, true slope 1, Bartlett lag `floor(4(T/100)^(2/9))` and 50 | corr(u, dx) 0: 0.335 / 0.184 (T=500), 0.267 / 0.109 (T=2500). corr 0.5: 0.466 / 0.280, 0.370 / 0.171 |
| Granger-Newbold rule | independent random walks, T=500 | no drift: DW median 0.032, R2 > DW in 0.808. drift 0.1: DW median 0.032, R2 > DW in 0.974 |
| Pre-audit: critical values | 100000 random walks, lag 0, n = 100, other code and seed, tau_c and tau_ct | within 0.020 (1 %) and 0.007 (5 %, 10 %) of `c(100)` |
| Pre-audit: power, tau_c 5 %, lag 0 | AR(1) spread, 4000 replications | n=250: H20 0.255, H60 0.080. n=500: H20 0.736. n=1000: H20 0.999, H60 0.416 |
| Pre-audit: size against lag | unit root, MA(1) theta -0.5, n=500, 3000 replications | p0 0.617, p1 0.245, p4 0.061, p12 0.047 |
| Pre-audit: levels HAC t-test | same design as above, 1500 replications | T=500: lag 5 0.317, lag 50 0.191. T=2500: lag 8 0.249, lag 50 0.099 (nominal 0.05) |
| Pre-audit: Granger-Newbold rule | independent random walks, T=500, 6000 replications | no drift: DW median 0.031, R2 > DW 0.815. drift 0.1: DW median 0.032, R2 > DW 0.975 |
| Pre-audit: ADF against `adfuller(autolag=None)` | 600 points: (c, 0), (c, 4), (ct, 4), (ct, 12) | statistic equal to 2e-14 relative; same `n`; same critical values at 1 %, 5 %, 10 % to 6 decimals |
| Audit R1 (C4): power defect in `c(n)` | b2 and b3 swapped, six rows | error 0.2511 at n = 20 and 0.0105 at n = 100; S2.T2 allows 0.03 and 0.05 at n = 100; the S2.T3 residual is 9.05e-9 for every defect (limit 1e-8) |

## A.6 What the test cannot say

- No rejection is not evidence of no relation. At 250 rows a 20-day half-life is detected 26 % of the time.
- A rejection says nothing about stability across regimes. A break lowers the power of the ADF test.
- A rejection does not measure the size of the basis risk. The spread volatility does.
- The known-vector test checks one vector. A different vector (partial pass-through) needs Engle-Granger (O-4).
- The test is on log levels. It checks multiplicative tracking `P = c * Z * M`, not additive tracking.
- Several tests on one data set inflate the false rejection rate. The library reports; it does not correct.

## A.7 Module placement

| Module | Tier | Reason |
|---|---|---|
| `riskquant/factor_regression/combination.py` | core | weighted sum of float64 levels with integer codes |
| `riskquant/factor_regression/mackinnon.py` | core | float64 constants and a polynomial in 1/n |
| `riskquant/factor_regression/adf.py` | core | design assembly and a call of `fit_core` |
| `riskquant/solutions/factor_regression/contracts.py` | facade | names, units, hypothesis and source text |
| `riskquant/solutions/factor_regression/digest.py` | facade | digests hash names (only importer of `hashing`) |
| `riskquant/solutions/factor_regression/labels.py` | facade | names of the new code groups (S1), unit-root texts (S4) |
| `riskquant/solutions/factor_regression/errors.py` | facade | `ScaleDomainError` and messages with series names (S4) |
| `riskquant/solutions/factor_regression/api.py` | facade | `unit_root_test`, the only public entry point |

## A.8 New code constants (S1)

Rule: a new constant never reuses the name of an existing constant (F-16: `LEVEL` is a transform code). Each new group is a `*_CODES` tuple of
`np.int8` with unique values, and has a labels group in the same slice (C.14, F-14). The values below enter the digests (A.3). S1.T7 fixes
them with a test, as `test_covariance_codes_match_the_canonical_spec_bytes` does for the covariance codes.

| Group tuple | Constant | Value | Facade member |
|---|---|---|---|
| `SCALE_CODES` | `SCALE_LEVEL` | 0 | `Scale.LEVEL` |
| `SCALE_CODES` | `SCALE_LOG` | 1 | `Scale.LOG` |
| `DETERMINISTIC_CODES` | `DET_CONSTANT` | 0 | `Deterministic.CONSTANT` |
| `DETERMINISTIC_CODES` | `DET_CONSTANT_TREND` | 1 | `Deterministic.CONSTANT_TREND` |
| `TEST_KIND_CODES` | `TEST_ADF` | 0 | none (the digest field only) |
| `SAMPLE_SIZE_ERROR_CODES` (existing group, append at the end) | `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` | 2 | none (reason of `InsufficientObservationsError`) |
| `SCALE_DOMAIN_ERROR_CODES` | `SCALE_LOG_LEVEL_NOT_POSITIVE` | 0 | none (reason of `ScaleDomainError`) |

The label of each code equals its constant name (S6.T10): `SCALE_LEVEL`, `SCALE_LOG`, `DET_CONSTANT`, `DET_CONSTANT_TREND`, `TEST_ADF`,
`SAMPLE_ROWS_BELOW_TABLE_MINIMUM`, `SCALE_LOG_LEVEL_NOT_POSITIVE`. The existing group `SAMPLE_SIZE_ERROR_CODES` keeps the values 0 and 1 and the order of its
members; the new member is appended. Its label entries (`SAMPLE_SIZE_ERROR_NAMES` and the reverse map) are added in S1 (F-18, S6.T10).
For `SAMPLE_ROWS_BELOW_TABLE_MINIMUM`, `n_available` is `n` and `n_required` is 20 (C.6). The docstring of `CoreSampleSizeError` gains one line for it.
The facade Enum values are the core codes. Their member names are API names and need not equal the labels (S4.T12).

New core exception class (S1, B-1): `CoreScaleDomainError(RiskQuantError)` in `riskquant/factor_regression/errors.py`, built as `CoreDomainError`.
Keyword-only constructor `CoreScaleDomainError(*, reason, term, aligned_index, level)`. Fields: `reason: int` (a code of `SCALE_DOMAIN_ERROR_CODES`),
`term: int` (index of the offending term in the declared combination order), `aligned_index: int` (index of the offending level on the aligned
date axis), `level: float`. No `str` field, parameter or annotation. `test_errors.py` gets the name in `_EXPECTED_CLASSES` and the factory
`errors.CoreScaleDomainError(reason=codes.SCALE_LOG_LEVEL_NOT_POSITIVE, term=1, aligned_index=4, level=0.0)` in `_FACTORIES`.
