# WS_FACTOR_REGRESSION: Appendix A, numerics, digests and oracles

Path: docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md
Revision: 3
Status: FROZEN
Used by: S1 to S7
Label rule: every definition below is Proposed. Every sandbox number is Observed (sandbox evidence).

## A.1 Row construction (S2)

Input: `AlignedLevels` with `m` strictly increasing dates `d_i` (int32 ordinals, `i = 0` to `m - 1`) and levels `v_i` per series.
The spec declares step `h >= 0` and stride `s >= 1`.

- Row `r = 0..R-1` has end index `e_r = h + r*s` and start index `b_r = e_r - h`.
- The row count is `R = floor((m - 1 - h) / s) + 1` when `m - 1 >= h`. Otherwise the library raises.
- `row_start_date_ord[r] = d[b_r]`. `row_end_date_ord[r] = d[e_r]`. For `h = 0` both are equal.
- LEVEL value: `v[e_r]`.
- LOG_RETURN value: `ln(v[e_r] / v[b_r])`. The ratio is formed before the logarithm, as in `ar1_filter._ret_log`.
- SIMPLE_RETURN value: `v[e_r] / v[b_r] - 1`.
- DIFFERENCE value: `v[e_r] - v[b_r]`.
- The two-point endpoint read matches D088 section 4 (level-backed column) and the D025 endpoint rule.
- Aligned dates that are neither a start index nor an end index of any row are recorded with reason `NOT_USED_BY_ROWS`.
  For `h >= 1`, `s = 1` and `m >= 2h` this set is empty. For `m < 2h` the aligned indices `m - h` to `h - 1` are unused.
- Overlap order `q = ceil(h / s) - 1` for `h >= 1`, and `q = 0` for `h = 0`.
  `h = 5, s = 1` gives `q = 4`: adjacent rows share increments, so the errors have an MA(4) structure.
  `h = 5, s = 5` gives `q = 0`.
- Order against alignment: align first, then build rows. If each series built returns on its own calendar,
  the pair `(r_P[t], r_Z[t])` would cover different intervals. S2.T1 pins the difference.
- Lags: v1 has no lagged regressor. Every term of one row covers the same interval.

## A.2 Scaling, rank gate, condition number (S3)

Design `X` has shape `(R, k)`. Column order: intercept (ones) if declared, then regressors in declared order.

1. `d_j = ||X[:, j]||_2`. A zero norm raises `RankDeficiencyError` naming column `j`. `Xs = X / d`.
2. `u, s, vh = numpy.linalg.svd(Xs, full_matrices=False)`. With `R >= k + 1`, `vh` has shape `(k, k)`.
3. `tol = rank_rtol * s[0]`. `rank = count(s > tol)`.
4. If `rank < k`, build the error payload:
   - Participation score `p_j = sum over i with s[i] <= tol of vh[i, j]^2`. It is unique for any null-space basis.
   - Involved columns: `p_j > 1e-12`. The core reports column indices; the facade adds the names.
   - Relation `i` (for each `s[i] <= tol`): `w_j = vh[i, j] / d_j`, divided by the `w_j` of largest magnitude
     (lowest column index on ties), so that `sum_j w_j * X[:, j]` is approximately 0 in original units.
     For nullity above 1 the relations are one basis. The participation scores are the unique naming.
5. If `rank == k`: `kappa = s[0] / s[k-1]`.
6. API names: `numpy.linalg.svd` is the gate. `numpy.linalg.matrix_rank(Xs, tol=rank_rtol * s[0])` is the test cross-check.
   The keyword `rtol` exists in numpy 2.4.4 (sandbox). Its presence at the declared floor numpy 1.26 is Unresolved.
   The library and the tests use `tol`, which both versions accept.

## A.3 Solve and covariance (S4)

- `gamma = vh.T @ ((u.T @ y) / s)`. `coef = gamma / d`. Residual `e = y - X @ coef`. `dof = R - k`. `rss = e @ e`.
- Bread `B = vh.T @ diag(1 / s^2) @ vh = (Xs.T @ Xs)^-1`.
- IID: `C = (rss / dof) * B`.
- HC0: `C = B @ M0 @ B`, with `M0 = Xs.T @ diag(e^2) @ Xs`.
- HAC_BARTLETT(L): `C = B @ M @ B`, with `M = M0 + sum_{j=1..L} w_j (G_j + G_j.T)`,
  `G_j = sum_{t=j..R-1} e_t e_{t-j} x_t x_{t-j}.T` (0-based, `x_t` is row `t` of `Xs`), `w_j = 1 - j / (L + 1)`.
  No small-sample factor. `L = 0` gives HC0. `0 <= L <= R - 1`.
- `cov = C / outer(d, d)`. `se = sqrt(diag(cov))`. `t = coef / se`. Two-sided p-value `2 * t_dist.sf(|t|, dof)`.
  For HC0 and HAC this reference distribution is an approximation. The text of the result says so.
- `r_squared = 1 - rss / sum((y - mean(y))^2)` with an intercept. Without an intercept: `1 - rss / sum(y^2)`.
- `DegenerateFitError` if `rss <= (R * eps)^2 * sum(y^2)` with `eps = numpy.finfo(numpy.float64).eps`. The rule includes
  `rss == 0.0` and `sum(y^2) == 0.0`, so the outcome of an exact fit does not depend on rounding.
- `DegenerateFitError` also if `sum((y - mean(y))^2) == 0.0` with an intercept, or if any `diag(cov) <= 0`.

## A.4 Diagnostics (S3)

- `durbin_watson(e) = sum_{t=1..R-1} (e_t - e_{t-1})^2 / sum_t e_t^2`, rows in date order.
- `LEVEL_TERM_PRESENT`: any term (response or regressor) uses LEVEL. Value: `1.0`.
- `R2_ABOVE_DURBIN_WATSON`: a LEVEL term is present and `r_squared > durbin_watson`. This is the
  Granger and Newbold rule of thumb for a spurious regression. It is a heuristic, not a test.
- `OVERLAP_NOT_CORRECTED`: `q > 0` and (covariance is IID or HC0, or HAC lag `< q`). Value: `float(q)`.
- `CONDITION_ABOVE_THRESHOLD`: threshold is not None and `kappa > threshold`. Value: `kappa`.
- The core emits each code as an `int8` from `codes.py`. The facade maps it to its name in `labels.py`.

## A.5 Wald (S5)

`R` has shape `(J, K)`, `r` has shape `(J,)`, `K = k`. Both are float64 and finite. `J >= 1`.
`W = (R @ coef - r).T @ inv(R @ cov @ R.T) @ (R @ coef - r)`, solved with a Cholesky factor of `R @ cov @ R.T`.
`F = W / J`. `chi2_pvalue = chi2.sf(W, J)`. `f_pvalue = f.sf(F, J, dof)`.
`R` must have full row rank (`numpy.linalg.svd` of `R` with `tol = 1e-12 * s[0]`).
A failed Cholesky factor raises `RestrictionError`. With IID covariance and normal errors the F test is exact.
With HC0 or HAC covariance, `W` is asymptotically chi-squared and `F` is an approximation.

## A.6 Canonical bytes and digests (S6)

All digests are computed in the facade module `digest.py` with `riskquant.solutions.run_identity.hashing.digest_named_arrays` (SHA-256, dtype, shape and C-order bytes folded in).
Text is encoded as UTF-8 into a `uint8` array. No Unicode normalisation is applied. Names and units hold no control character.
`ENGINE_VERSION = "riskquant.factor_regression/1"`, defined in the facade (the core holds no `str`).

`spec_digest = digest_named_arrays(` in this order `)`:

| Name | dtype | Content |
|---|---|---|
| `schema_version` | int64 `[1]` | constant |
| `engine_version` | uint8 | UTF-8 of `ENGINE_VERSION` |
| `terms` | uint8 | one line `Y\x1f{name}\x1f{unit}\x1f{TRANSFORM}` for the response, then one line `X\x1f{name}\x1f{unit}\x1f{TRANSFORM}` per regressor in declared order, joined by `\x1e` |
| `intercept` | int8 `[0 or 1]` | flag |
| `step`, `stride` | int64 `[h]`, `[s]` | declared values |
| `window` | int32 `[start_ord, end_ord]` | from the alignment |
| `covariance_kind` | int8 `[0, 1 or 2]` | IID 0, HC0 1, HAC_BARTLETT 2 |
| `covariance_lag` | int64 `[L]` | `-1` when not applicable |
| `rank_rtol` | float64 `[x]` | declared value |
| `condition_threshold` | float64 | `[x]`, or an empty array when None |

`sample_digest = digest_named_arrays(` in this order `)`:

| Name | dtype | Content |
|---|---|---|
| `row_start_date_ord`, `row_end_date_ord` | int32 `(R,)` | row interval dates |
| `column_names` | uint8 | lines `Y\x1f{name}`, `C\x1f` (if intercept), then `X\x1f{name}` per regressor, joined by `\x1e` |
| `y` | float64 `(R,)` | response values |
| `X` | float64 `(R, k)` C-order | design before scaling, intercept column first |

`row_digest` is `digest_named_arrays` of the two arrays `row_start_date_ord` and `row_end_date_ord`, with the dtypes and order of `sample_digest`.
Units enter only `spec_digest`. Two results are comparable when `row_digest` and the response term are equal.

## A.7 Oracle recipes (tests)

Every oracle below is independent of the library code.

- Exact OLS: treat each float as `fractions.Fraction`. `b = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)`, `a = (Sy - b*Sx) / n`.
  For `k = 3` solve the normal equations by Fraction Gaussian elimination.
- HC0 and HAC: evaluate the formulas of A.3 with explicit Fraction loops over rows and lags.
- With `dof = 2` (`R = k + 2`) the tails have closed forms:
  - Student t, two-sided: `p = 1 - |t| / sqrt(2 + t^2)`.
  - Chi-squared with 2 degrees: `sf(W) = exp(-W / 2)`.
  - F with `(2, 2)` degrees: `sf(F) = 1 / (1 + F)`.
- Two unit-norm columns with correlation `rho`: `kappa = sqrt((1 + rho) / (1 - rho))`.
- Durbin-Watson of `(1, -1, 1, -1)` is `12 / 4 = 3`.
- Simple-return identity with Fractions: `sr_Q = sr_Z + sr_S + sr_Z * sr_S`, and `sr_Q != sr_Z + sr_S`.
  Log-return additivity follows from `ln(ab) = ln a + ln b`. The float residual is the test measure.
- Digest oracle: rebuild the bytes of A.6 with `hashlib.sha256` following `digest_array` and `digest_named_arrays`.

### Sandbox evidence (Python 3.12.3, numpy 2.4.4, scipy 1.17.1)

| Check | Setup | Result |
|---|---|---|
| Log additivity | `Q = Z*S`, 4 returns | `max abs(r_Q - r_Z - r_S) = 5.55e-17`, `matrix_rank = 2` |
| Simple identity | Fraction arithmetic | `sr_Q = sr_Z + sr_S + sr_Z*sr_S` True. `sr_Q = sr_Z + sr_S` False. |
| Unscaled versus scaled solve | `x = mu + N(0,1)`, `y = 3 + 0.5x + 0.1 N(0,1)`, `numpy.random.default_rng(seed)`, 250 rows, intercept, seeds 1 to 10, exact Fraction oracle | mu 1e6: unscaled `lstsq(rcond=None)` rank 2, slope error 9.4e-12; scaled SVD slope 3.9e-11, raw intercept 3.9e-5, fitted relative 9.3e-16. mu 1e7: unscaled rank 1, slope error 1.2e-2; scaled SVD slope 2.8e-10, raw intercept 2.8e-3, fitted relative 5.6e-16. mu 1e8: unscaled rank 1, slope error 1.2e-2; scaled SVD slope 3.3e-9, raw intercept 0.33, fitted relative 1.6e-15. The raw intercept error equals mean(x) times the slope error. |
| Rank gate, log identity | 200 seeds, `R=250`, `tol = R*eps`, returns `ln(v[t]/v[t-1])` | vol 1%, 0.5% and 0.1%: detected 200/200. vol 0.05%: kappa 9.7e12 to 1.7e13, detected 0/200. vol 0.02% and 0.01%: kappa 1.9e12 to 6.4e12, detected 0/200. |
| Simple-return design | 200 seeds, `R=250` | `kappa * vol` 2.40 to 3.59 at vol 1%, 0.5%, 0.1% and 0.05%. `kappa(0.5%) / kappa(1%)` 1.996 to 2.004 per seed. Rank 3 in all runs. |
| Degenerate-fit rule | `R` in 30, 250, 2500 | exact `y`: `rss` 1.2e-26 to 1.1e-22, threshold 1.3e-23 to 3.1e-17, rule fires. Noise 1e-8: `rss` 3.0e-15 to 2.6e-13, rule does not fire. Zero `y`, intercept only: `rss = 0.0`, rule fires. |

## A.8 Module placement (S1 to S7)

Rule (C.22): a module that needs a name, unit, label, `str`-typed data or a digest is facade. Every other module is core.

| Module | Tier | Reason |
|---|---|---|
| `riskquant/factor_regression/errors.py` | core | exceptions carry integer indices, integer codes and numeric values; no str data |
| `riskquant/factor_regression/codes.py` | core | `int8` constants for transforms, covariance kinds, reasons, diagnostics; no names |
| `riskquant/factor_regression/align.py` | core | alignment is set logic on `int32` date ordinals and position arrays |
| `riskquant/factor_regression/rows.py` | core | rows and transforms are arithmetic on `float64` levels with integer codes |
| `riskquant/factor_regression/design.py` | core | scaling, SVD, rank and kappa are matrix algebra on `float64` |
| `riskquant/factor_regression/diagnostics.py` | core | Durbin-Watson and flags reduce to numbers and integer codes |
| `riskquant/factor_regression/covariance.py` | core | IID, HC0 and HAC are matrix algebra |
| `riskquant/factor_regression/fit.py` | core | the solve consumes and returns arrays and codes |
| `riskquant/factor_regression/wald.py` | core | the joint statistic is matrix algebra |
| `riskquant/solutions/factor_regression/contracts.py` | facade | `ObservedSeries`, `ModelSpec` and results hold names and units |
| `riskquant/solutions/factor_regression/errors.py` | facade | messages carry series names and dates |
| `riskquant/solutions/factor_regression/labels.py` | facade | maps core indices and codes to names and dates |
| `riskquant/solutions/factor_regression/digest.py` | facade | digests hash names; only importer of `hashing` (C.21) |
| `riskquant/solutions/factor_regression/api.py` | facade | the only public import surface (D001, C.23) |

Paths are under `src/`. Both `__init__.py` files are empty (D002).
