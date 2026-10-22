# Independent Technical Audit Report: `riskquant.factor_regression` & Cointegration Subsystem

- **Audited Packages**:
  - Core: `src/riskquant/factor_regression/` (`codes.py`, `errors.py`, `align.py`, `rows.py`, `design.py`, `covariance.py`, `fit.py`, `diagnostics.py`, `wald.py`, `combination.py`, `mackinnon.py`, `adf.py`)
  - Facade: `src/riskquant/solutions/factor_regression/` (`contracts.py`, `labels.py`, `digest.py`, `api.py`)
- **Authority Baseline**:
  - `WS_FACTOR_REGRESSION_COINTEGRATION` (Revision 4)
  - `A_NUMERICS.md` (Revision 4)
  - Architectural Decisions `D104` and `D105`
  - Reference Literature: MacKinnon (2010), Queen's Economics Department Working Paper No. 1227 (`qed_wp_1227.pdf`)

---

## Executive Summary

1. **Specification Conformance (Part I)**: The landed implementation across `riskquant.factor_regression` and `riskquant.solutions.factor_regression` conforms to `WS_FACTOR_REGRESSION_COINTEGRATION` (Revision 4), `A_NUMERICS.md` (Revision 4), and Decision `D105` with **zero contract violations or numerical defects**. All 15 acceptance criteria (`C.1`–`C.15`), all numerical appendices (`A.1`–`A.8`), and all 23 unique MacKinnon (2010) $N=1$ critical value response-surface constants match the specification and primary literature verbatim.
2. **Extended Audit Beyond the Specification (Part II)**: Stepping outside the v1 workstream specification to evaluate the full `factor_regression` package as a production econometric and cointegration library surfaces **11 concrete findings** across four categories:
   - **Core Numerical & Exception-Path Edge Cases (3 findings)**: Overflow ordering in `wald.py` prior to Step 10 (`RESTRICTION_NON_FINITE`), `t_stats` overflow misreporting in `fit.py` (`DEGENERATE_NON_POSITIVE_VARIANCE`), and missing `np.errstate` guards in `combination.py` and `rows.py`.
   - **Facade Contract & Boundary Hardening (4 findings)**: Missing `order="C"` in `_frozen_array_copy`, missing structural validation on manually constructed `AlignedLevels`, missing control-character validation on `Term.series`, and unvalidated duplicate regressors in `ModelSpec`.
   - **Econometric & Cointegration Workflow Limitations (4 findings)**: Discarded ADF residuals/diagnostics in `adf_core` needed for lag selection, unexposed historical combination series $z_t$, absence of a log-level transform (`Transform.LOG`) in `fit_ols`, and Granger-Newbold $R^2 > DW$ firing on uncentered $R^2$ when `intercept=False`.
   - **Digest & Type Hygiene (2 findings)**: Direct calls to `digest.py` functions without control-character assertions, and the 13 strict-typing errors in `run_identity/hashing.py` requiring the `mypy baseline` gate workaround.

---

## Part I: Specification & Contract Conformance Audit

### 1.1 Contract & Decision Conformance Matrix

| Clause | Requirement Summary | Landed Implementation | Verdict |
| :--- | :--- | :--- | :--- |
| **C.1** | `CombinationTerm(series, weight)` & `LinearCombination(terms, scale)`: `kw_only`, no defaults; strict `str` / `Cc` category checks; numeric non-bool `weight` stored as `float`, rejecting `0.0`, non-finite, and `int` overflow ($10^{400}$) with `ModelSpecError`; non-empty `tuple` of unique terms; `Scale` enum check. | `contracts.py`: `CombinationTerm.__post_init__` and `LinearCombination.__post_init__` validate types before domain rules, catch `OverflowError` on `_check_float_param` to raise `ModelSpecError`, and enforce `_has_control` (`unicodedata.category == "Cc"`). | **PASS** |
| **C.2** | $z_t = \sum_i w_i g(v_i[t])$ on aligned dates ($g = \text{id}$ or $\ln$) in left-to-right declared term order; unknown series raises `ModelSpecError`. | `api.py:unit_root_test` runs `_missing_term_series` before core calls; `combination.py:combine_core` accumulates left-to-right using a single `(m,)` scratch buffer. | **PASS** |
| **C.3** | `LOG` domain requires every aligned level of every referenced term $> 0$; term-major scan in ascending date order before any `np.log`; maps `CoreScaleDomainError` to `ScaleDomainError` naming series, date, and level. | `combination.py:_validate_log_domain` scans term-major prior to `_weighted_term`; `api.py:_map_core_error` maps to `ScaleDomainError` with `__cause__` chained. | **PASS** |
| **C.4** | `UnitRootSpec(combination, deterministic, lag, rank_rtol)`: no defaults; `Deterministic` member check; non-bool integer index for `lag` with $0 \le \text{lag} \le 2^{63}-1$; non-bool real `rank_rtol` in $(0, 1)$. | `contracts.py:UnitRootSpec.__post_init__` reuses `_check_int_param`, `_INT64_MAX`, and `_check_float_param` with negated interval check `not (0.0 < float(self.rank_rtol) < 1.0)`. | **PASS** |
| **C.5 / A.1** | ADF regression of $\Delta z_t$ on `[1, (trend), z_{t-1}, dz_{t-1} .. dz_{t-p}]` for $t = p+1 \dots m-1$; single call to `fit.fit_core` with `IID` covariance and `lag=0`; $\tau = \text{coef}[j] / \text{se}[j]$. | `adf.py:_build_regression` and `adf_core` assemble C-order `(n, k)` design and `(n,)` response, call `fit.fit_core(..., intercept=True, covariance_kind=int(codes.IID), covariance_lag=0)`, and extract `gamma` and `tau` at column `1 + int(has_trend)`. | **PASS** |
| **C.6 / A.1** | $n = m - 1 - p \ge 20$ checked in `adf.py` after input/parameter checks and **before** building design or calling `fit_core`; raises `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` even when $n < k + 1$ also holds; `rank_gate` raises `SAMPLE_ROWS_NOT_ABOVE_COLUMNS` when $n \ge 20$ and $n < k + 1$. | `adf.py:adf_core` computes `n_rows = m - 1 - p` with Python `int` arithmetic and checks `n_rows < _TABLE_MINIMUM_ROWS` ($20$) prior to `_build_regression` and `fit.fit_core`. | **PASS** |
| **C.7 / A.2** | MacKinnon (2010) $N=1$ response surface $c(n) = \beta_\infty + \beta_1/n + \beta_2/n^2 + \beta_3/n^3$ at $(0.01, 0.05, 0.10)$; 23 constants stored once in `mackinnon.py`. | `mackinnon.py:_COEFFICIENTS` and `critical_values(table, n)` evaluate `rows[:, 0] + rows[:, 1] / n_1 + rows[:, 2] / n_2 + rows[:, 3] / n_3` and return a read-only `(3,)` `float64` array. | **PASS** |
| **C.8 / A.4** | `UnitRootResult` fields, read-only array copies, `window = (levels.start_ord, levels.end_ord)`, alignment-only `exclusions`, fixed `null_hypothesis` and `critical_value_source` strings, no p-value or verdict fields (`D104.4`). | `contracts.py:UnitRootResult`, `labels.py:NULL_HYPOTHESIS_TEXTS`, `labels.py:CRITICAL_VALUE_SOURCES`, and `api.py:unit_root_test` match all field, text, and immutability rules. | **PASS** |
| **C.9 / A.3** | `unit_root_spec_digest`, `unit_root_sample_digest`, and `row_digest` byte layouts, array names, dtypes, and `column_names` / `terms` grammars (`\x1f` / `\x1e`). | `digest.py:unit_root_spec_digest` and `unit_root_sample_digest` (delegating to `sample_digest` with `response_name="dz"`, `intercept=True`) match A.3 field-for-field. | **PASS** |
| **C.10** | Core-to-facade exception mapping with `__cause__` chaining; overflow of combination or difference (`INPUT_NON_FINITE` in stage `"adf"`) mapped to `DegenerateFitError`; other `CoreInputError`, `TypeError`, `ValueError` propagate unmapped. | `api.py:_map_core_error` and `_facade_errors` handle all 6 mapped exception paths and let unmapped internal errors propagate unchanged. | **PASS** |
| **C.11** | Core modules hold no `str` fields/annotations/returns and no `Enum`; core imports restricted to stdlib, `numpy`, `riskquant.exceptions`, and `riskquant.factor_regression`; facade imports within D104 allow-list (`hashing` imported only in `digest.py`). | Verified across `combination.py`, `mackinnon.py`, `adf.py`, `codes.py`, `errors.py`, and all 5 facade modules. | **PASS** |
| **C.12** | `align_levels`, `fit_ols`, `wald_test`, and `ENGINE_VERSION` (`"riskquant.factor_regression/1"`) unchanged. | Signatures, implementations, digests, and `ENGINE_VERSION` are untouched. | **PASS** |
| **C.13** | No caller array mutation; all retained/returned arrays are read-only copies. | `_read_only` in core modules and `_frozen_array_copy` in `contracts.py` enforce `copy=True` and `write=False`. | **PASS** |
| **C.14 / A.8** | New `np.int8` code groups (`SCALE_CODES`, `DETERMINISTIC_CODES`, `TEST_KIND_CODES`, `SCALE_DOMAIN_ERROR_CODES`), appended `SAMPLE_ROWS_BELOW_TABLE_MINIMUM = 2`, and bijective entries in `labels.ALL_CODE_GROUPS`. | `codes.py` and `labels.py` define all constants, reverse maps, and `ALL_CODE_GROUPS` entries with exact A.8 values. | **PASS** |
| **C.15** | `api.__all__` exports the 8 new public names; `Scale` and `Deterministic` enum values equal their core `codes.py` integers. | `api.__all__` contains all 28 expected public symbols; `Scale` and `Deterministic` bind `int(codes.*)`. | **PASS** |

---

### 1.2 Verification of MacKinnon (2010) $N=1$ Response-Surface Constants

Cross-checking `src/riskquant/factor_regression/mackinnon.py:_COEFFICIENTS` against `A_NUMERICS.md` (§A.2) and MacKinnon (2010) `qed_wp_1227.pdf` (Table 2 on p. 13 and Table 3 on p. 14) for the polynomial

$$c(n) = \beta_\infty + \frac{\beta_1}{n} + \frac{\beta_2}{n^2} + \frac{\beta_3}{n^3}$$

| Table Code | Deterministic Case | Level | $\beta_\infty$ | $\beta_1$ | $\beta_2$ | $\beta_3$ | Source Verification (`qed_wp_1227.pdf`) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `TABLE_TAU_C = 0` | `CONSTANT` ($\tau_c$, $N=1$) | $0.01$ (1%) | $-3.43035$ | $-6.5393$ | $-16.786$ | $-79.433$ | Matches Table 2, $N=1$, 1% (p. 13) |
| `TABLE_TAU_C = 0` | `CONSTANT` ($\tau_c$, $N=1$) | $0.05$ (5%) | $-2.86154$ | $-2.8903$ | $-4.234$ | $-40.040$ | Matches Table 2, $N=1$, 5% (p. 13) |
| `TABLE_TAU_C = 0` | `CONSTANT` ($\tau_c$, $N=1$) | $0.10$ (10%) | $-2.56677$ | $-1.5384$ | $-2.809$ | $0.0$ | Matches Table 2, $N=1$, 10% (p. 13; $\beta_3$ omitted) |
| `TABLE_TAU_CT = 1` | `CONSTANT_TREND` ($\tau_{ct}$, $N=1$) | $0.01$ (1%) | $-3.95877$ | $-9.0531$ | $-28.428$ | $-134.155$ | Matches Table 3, $N=1$, 1% (p. 14) |
| `TABLE_TAU_CT = 1` | `CONSTANT_TREND` ($\tau_{ct}$, $N=1$) | $0.05$ (5%) | $-3.41049$ | $-4.3904$ | $-9.036$ | $-45.374$ | Matches Table 3, $N=1$, 5% (p. 14) |
| `TABLE_TAU_CT = 1` | `CONSTANT_TREND` ($\tau_{ct}$, $N=1$) | $0.10$ (10%) | $-3.12705$ | $-2.5856$ | $-3.925$ | $-22.380$ | Matches Table 3, $N=1$, 10% (p. 14) |

---

### 1.3 Specification-Level Implementation Notes

1. **Label Allocation Guard in `api.py:unit_root_test`**:
   - `api.py` bounds the ADF column label loop via `for q in range(1, min(lag, int(levels.date_ord.shape[0])) + 1):`. Because `UnitRootSpec` accepts lags up to $2^{63} - 1$ (`C.4`), this cap prevents an out-of-memory allocation in the facade prior to `adf_core` raising `SAMPLE_ROWS_BELOW_TABLE_MINIMUM`, while guaranteeing exact column labels whenever $n = m - 1 - \text{lag} \ge 20$ (since $\text{lag} \le m - 21 < m$).
2. **Section 7 Backlog Items Confirmed Contained**:
   - The items recorded in Section 7 of `WS_FACTOR_REGRESSION_COINTEGRATION` (`codes.py` and `digest.py` module docstring summaries, `CombinationTerm` overflow message wording for negative integers below $-1.8 \times 10^{308}$, `int32` `np.arange` in `adf_core` bounded upstream by `align.py:INPUT_TOO_LONG`, and `OverflowError` on $10^{400}$ for `rank_rtol` matching `ModelSpec`) behave as documented and do not violate any v1 contract guarantee.

---

## Part II: Extended Technical Audit (Beyond the Specification)

Stepping beyond the `WS_FACTOR_REGRESSION_COINTEGRATION` contract to evaluate the complete `riskquant.factor_regression` core and `riskquant.solutions.factor_regression` facade as a cohesive econometric subsystem reveals the following findings.

---

### 2.1 Core Numerical & Exception-Path Edge Cases

#### Finding N-1 (Medium): `wald.py` — Overflow of `rhs_unit` or `m` Fails Before Step 10 (`RESTRICTION_NON_FINITE`)
- **Location**: `src/riskquant/factor_regression/wald.py:303-338`
- **Mechanism**:
  ```python
  with np.errstate(over="ignore"):
      rhs_unit = np.ldexp(rhs / unit_norm, -exponent)
  ...
  gap = rows @ coef - rhs_unit
  m = rows @ cov @ rows.T
  m = 0.5 * (m + m.T)
  try:
      chol = np.linalg.cholesky(m)
  except np.linalg.LinAlgError as exc:
      raise CoreRestrictionError(
          reason=codes.RESTRICTION_NOT_POSITIVE_DEFINITE, shape=rest_shape
      ) from exc

  z = np.linalg.solve(chol, gap)
  ```
- **Impact**:
  1. **Misclassified Exception Reason on `m` Overflow**: Although Step 10 (`lines 340-346`) is documented as catching overflow of the rescaled restriction or sandwich product via `RESTRICTION_NON_FINITE`, if `cov` has large finite entries ($> 10^{300}$) and `m = rows @ cov @ rows.T` overflows to $\pm\infty$ or $\text{NaN}$, `np.linalg.cholesky(m)` raises `np.linalg.LinAlgError` at Step 9. Step 9 catches this and raises `RESTRICTION_NOT_POSITIVE_DEFINITE` instead of `RESTRICTION_NON_FINITE`.
  2. **Unguarded NumPy `RuntimeWarning`s**: `gap = rows @ coef - rhs_unit`, `m = rows @ cov @ rows.T`, and `z = np.linalg.solve(chol, gap)` execute outside `with np.errstate(over="ignore", invalid="ignore"):`. When `rhs_unit` overflows to $\infty$ for $J \ge 2$ and `chol` has non-zero off-diagonal entries, forward substitution inside `np.linalg.solve` computes $\infty - L_{1,0}\cdot\infty = \text{NaN}$, emitting a `RuntimeWarning: invalid value encountered in subtract` before Step 10 is reached.
- **Recommendation**:
  Wrap the `gap`, `m`, and `solve` computations in `with np.errstate(over="ignore", invalid="ignore"):` and check `if not ( bool(np.isfinite(gap).all()) and bool(np.isfinite(m).all()) ): raise CoreRestrictionError(reason=codes.RESTRICTION_NON_FINITE, shape=rest_shape)` **before** calling `np.linalg.cholesky(m)`.

#### Finding N-2 (Medium): `fit.py` — `t_stats` Overflow Misreported as `DEGENERATE_NON_POSITIVE_VARIANCE`
- **Location**: `src/riskquant/factor_regression/fit.py:381-405`
- **Mechanism**:
  `fit_core` first verifies that `diagonal = np.diagonal(cov)` is finite and strictly positive (`diagonal > 0.0`). It then computes `se = np.sqrt(diagonal)`, `t_stats = coef / se`, and `p_values`, followed by a combined finiteness check:
  ```python
  cov_finite = np.isfinite(cov)
  affected = ~(
      np.isfinite(coef)
      & np.isfinite(se)
      & np.isfinite(t_stats)
      & np.isfinite(p_values)
      & cov_finite.all(axis=0)
      & cov_finite.all(axis=1)
  )
  if bool(affected.any()):
      raise _non_positive_variance(diagonal, np.flatnonzero(affected))
  ```
- **Impact**:
  When `diagonal[j]` is finite and strictly positive (e.g., $10^{-300}$, so $\text{se}_j = 10^{-150}$) and $\text{coef}_j$ is large (e.g., $10^{160}$), `t_stats[j] = coef[j] / se[j]` overflows `float64` to $\pm\infty$. `_non_positive_variance` then raises `CoreDegenerateFitError(reason=codes.DEGENERATE_NON_POSITIVE_VARIANCE, column=j, value=1e-300, bound=0.0)`—producing a contradictory diagnostic payload where a non-positive variance is reported with `value = 1e-300 > bound = 0.0`.
- **Recommendation**:
  Either clamp `np.abs(t_stats)` at `np.finfo(np.float64).max` when `coef` and `se > 0` are both finite (so $p\text{-value} = 0.0$ cleanly without failing a valid regression), or introduce a distinct `DEGENERATE_NON_FINITE_INFERENCE` code so the error payload does not claim `diagonal[j] <= 0.0`.

#### Finding N-3 (Low): `combination.py` and `rows.py` — Missing `np.errstate` Guards During Overflow
- **Location**: `src/riskquant/factor_regression/combination.py:217-244` and `src/riskquant/factor_regression/rows.py:110-169`
- **Mechanism**:
  Both `combine_core` ($\sum_i w_i g(v_i)$) and `build_rows` (`DIFFERENCE`, `SIMPLE_RETURN`, `LOG_RETURN`) intentionally defer overflow rejection to downstream stages (`adf_core`, `assemble_design`, `fit_core`), which raise `CoreInputError(reason=codes.INPUT_NON_FINITE)` and map to `DegenerateFitError` at the facade. However, unlike `adf.py`, `design.py`, `covariance.py`, and `fit.py`, neither `combine_core` nor `build_rows` wraps its arithmetic in `with np.errstate(over="ignore", invalid="ignore"):`.
- **Impact**:
  When two combination terms overflow in opposite directions ($+\infty$ and $-\infty$) or a difference overflows `float64`, NumPy emits `RuntimeWarning`s to stderr even though the facade catches and translates the condition into a clean domain exception.
- **Recommendation**:
  Wrap the ufunc arithmetic in `combination.py:combine_core` and `rows.py:build_rows` in `with np.errstate(over="ignore", invalid="ignore"):`.

---

### 2.2 Facade Contract & Boundary-Hardening Gaps

#### Finding B-1 (Medium): `contracts.py` — `_frozen_array_copy` Does Not Enforce C-Contiguity (`order="C"`)
- **Location**: `src/riskquant/solutions/factor_regression/contracts.py:49-56`
- **Mechanism**:
  ```python
  def _frozen_array_copy(value: NDArray[Any]) -> NDArray[Any]:
      out = np.array(value, copy=True)
      out.flags.writeable = False
      return out
  ```
- **Impact**:
  In NumPy, `np.array(value, copy=True)` defaults to `order="K"`, which preserves Fortran (`order="F"`) or non-contiguous strides when copying a 2-D array. By contrast, `src/riskquant/config/proxy_model/compiled.py:76` uses `np.array(value, dtype=dtype, copy=True, order="C")`. Furthermore, `digest.py:75` (`_as_c_array`) asserts `arr.flags["C_CONTIGUOUS"]`. If a caller constructs `AlignedLevels` or `OlsResult` with a Fortran-ordered 2-D array, `_frozen_array_copy` retains non-C-contiguous memory despite the class docstrings promising C-order arrays.
- **Recommendation**:
  Change `_frozen_array_copy` to `out = np.array(value, copy=True, order="C")`.

#### Finding B-2 (Medium): `contracts.py` & `api.py` — `AlignedLevels.__post_init__` Does Not Validate Shape, Monotonicity, or Tuple Lengths
- **Location**: `src/riskquant/solutions/factor_regression/contracts.py:288-308` and `src/riskquant/solutions/factor_regression/api.py:521-546, 926-951`
- **Mechanism**:
  While `ObservedSeries.__post_init__` enforces 1-D shapes, equal lengths, strict monotonicity of `date_ord`, finiteness of `values`, and string validity, `AlignedLevels.__post_init__` only checks `isinstance` and `dtype` of `date_ord` and `levels`.
- **Impact**:
  1. In `unit_root_test` (`api.py:926-951`), `_core_aligned_levels(levels)` is not called. `levels.levels` is passed directly to `combination.combine_core`, and `levels.date_ord` is indexed at `lines 950-951` (`levels.date_ord[adf_result.row_start_index]`). If a caller passes a manually constructed `AlignedLevels` where `date_ord` has fewer elements than `levels.shape[1]` or is not strictly increasing, `unit_root_test` either raises an unhandled `IndexError` or silently produces non-monotonic `window` and `row_digest` dates.
  2. In `fit_ols` (`api.py:534-546`), `_core_aligned_levels` indexes `levels.units[j]`, `index_of[record.series]`, and `labels.EXCLUSION_NAME_TO_CODE[record.reason]`, which can raise raw `IndexError` or `KeyError` on manually constructed `AlignedLevels` instances.
- **Recommendation**:
  Validate in `AlignedLevels.__post_init__` that:
  - `date_ord.ndim == 1`, `levels.ndim == 2`, `date_ord.shape[0] >= 1`, `levels.shape == (len(names), date_ord.shape[0])`
  - `len(names) >= 1`, `len(units) == len(names)`, `len(set(names)) == len(names)`
  - `bool(np.all(date_ord[1:] > date_ord[:-1]))` and `bool(np.isfinite(levels).all())`
  - `window == ( int(date_ord[0]), int(date_ord[-1]) )`

#### Finding B-3 (Low): `contracts.py` — `Term.__post_init__` Does Not Reject Empty Strings or Control Characters
- **Location**: `src/riskquant/solutions/factor_regression/contracts.py:150-159`
- **Mechanism**:
  `ObservedSeries` and `CombinationTerm` both reject `series == ""` and `_has_control(series)` (Unicode category `Cc`, which includes the `\x1e` and `\x1f` record/unit separators used in `digest.py`). `Term.__post_init__` only checks `isinstance(self.series, str)`.
- **Impact**:
  A `Term(series="", transform=...)` or `Term(series="a\x1fb", transform=...)` can be constructed without raising `ModelSpecError` at `Term` construction time (failing only later if checked against `AlignedLevels.names`, or polluting the separator grammar if passed directly to `digest.spec_digest`).
- **Recommendation**:
  Add `if self.series == "" or _has_control(self.series): raise ModelSpecError(...)` to `Term.__post_init__`.

#### Finding B-4 (Low): `contracts.py` — `ModelSpec.__post_init__` Does Not Reject Duplicate Regressors
- **Location**: `src/riskquant/solutions/factor_regression/contracts.py:323-355`
- **Mechanism**:
  `LinearCombination.__post_init__` rejects duplicate series names immediately at specification construction time with `ModelSpecError`. By contrast, `ModelSpec.__post_init__` accepts duplicate `(series, transform)` pairs in `regressors` (and allows `response` to appear identically in `regressors`), deferring failure until SVD rank checking in `design.rank_gate` (`RankDeficiencyError`).
- **Recommendation**:
  Consider rejecting duplicate `(term.series, term.transform)` entries in `ModelSpec.regressors` at construction time with `ModelSpecError`, matching `LinearCombination`.

---

### 2.3 Econometric & Cointegration Workflow Limitations

#### Finding E-1 (Medium): `adf.py` & `contracts.py` — `adf_core` Discards Residuals and Fit Diagnostics Required for Lag Selection
- **Location**: `src/riskquant/factor_regression/adf.py:329-356` and `src/riskquant/solutions/factor_regression/contracts.py:678-731`
- **Mechanism**:
  Inside `adf_core`, `ols = fit.fit_core(...)` computes `ols.resid`, `ols.fitted`, `ols.rss`, `ols.dof`, `ols.cov`, and `ols.durbin_watson`. `adf_core` then discards all of these except `coef`, `se`, and `condition_number`.
- **Econometric Impact**:
  As highlighted in `WS_FACTOR_REGRESSION_COINTEGRATION` (F-9, A.5) and MacKinnon (2010, §2), the finite-sample size of the ADF test under moving-average or serially correlated innovations depends critically on selecting an augmentation lag $p$ that whitens the regression residuals $\hat{e}_t$. Because `UnitRootResult` does not expose `resid`, `rss`, `dof`, or `durbin_watson`:
  - A caller cannot run Ljung-Box, Breusch-Godfrey, or residual ACF diagnostics on the ADF regression residuals.
  - A caller cannot evaluate AIC, BIC, or Ng-Perron MAIC across a grid of lags $p \in \{0, \dots, p_{\max}\}$ without manually reconstructing the combination series $z_t$ and the ADF design matrix outside the library.
- **Recommendation**:
  In a future revision (`v2`), retain `resid: NDArray[np.float64]`, `rss: float`, `dof: int`, and `durbin_watson: float` on `CoreAdf` and `UnitRootResult`. Because `fit_core` already computes these quantities, the runtime cost is zero.

#### Finding E-2 (Low): `api.py` & `contracts.py` — Validated Combination Series $z_t$ Is Not Accessible to Callers
- **Location**: `src/riskquant/solutions/factor_regression/api.py:854-985`
- **Mechanism**:
  `combination.combine_core` is internal to the core package (`D104.2`), and `UnitRootResult` does not expose the `(m,)` aligned combination series $z_t = \sum_i w_i g(v_{i,t})$.
- **Econometric Impact**:
  `A_NUMERICS.md` (§A.6) emphasizes that ADF rejection does not measure basis risk magnitude—the spread volatility and half-life of $z_t$ do. Currently, a user must re-implement `combine_core` outside `riskquant` to inspect or plot $z_t$ or $\Delta z_t$.
- **Recommendation**:
  Either include `combination_values: NDArray[np.float64]` (shape `(m,)`) on `UnitRootResult` or expose a public helper `combine_levels(levels: AlignedLevels, combination: LinearCombination) -> NDArray[np.float64]` on the facade.

#### Finding E-3 (Low): `codes.py` & `rows.py` — `fit_ols` Lacks a `Transform.LOG` Level Transform
- **Location**: `src/riskquant/factor_regression/codes.py:29-34` and `src/riskquant/factor_regression/rows.py:110-169`
- **Mechanism**:
  `unit_root_test` supports `Scale.LEVEL` and `Scale.LOG` ($\sum_i w_i \ln v_{i,t}$) for multiplicative proxy combinations. However, `fit_ols` supports only `Transform.LEVEL`, `LOG_RETURN`, `SIMPLE_RETURN`, and `DIFFERENCE`.
- **Econometric Impact**:
  A researcher who tests a log-linear proxy combination ($\ln P_t - w_1 \ln Z_t - w_2 \ln M_t$) in `unit_root_test` cannot estimate the corresponding static cointegrating regression in log-levels ($\ln P_t = \alpha + \beta_1 \ln Z_t + \beta_2 \ln M_t + u_t$) using `fit_ols` at `step=0` without pre-transforming the raw `ObservedSeries` outside the package.
- **Recommendation**:
  Consider adding `Transform.LOG` (at `step = 0`, reusing the positive-domain check from `combination.py`) when Engle-Granger (O-4) is scheduled.

#### Finding E-4 (Low): `diagnostics.py` — `R2_ABOVE_DURBIN_WATSON` Fires on Uncentered $R^2$ When `intercept=False`
- **Location**: `src/riskquant/factor_regression/fit.py:406-410` and `src/riskquant/factor_regression/diagnostics.py:278-283`
- **Mechanism**:
  When `has_intercept=False`, `fit.py` computes the uncentered coefficient of determination:

  $$R^2_{\text{uncentered}} = 1 - \frac{\sum_{t=0}^{n-1} e_t^2}{\sum_{t=0}^{n-1} y_t^2}$$

  In `diagnostics.py:278-283`, `diagnose` emits `R2_ABOVE_DURBIN_WATSON` whenever `level_present and r2 > dw`, regardless of whether the regression includes an intercept.
- **Econometric Impact**:
  For any level series with a non-zero mean ($\bar{y} \gg s_y$) fit without an intercept, $\sum y_t^2$ is dominated by $n\bar{y}^2$, forcing $R^2_{\text{uncentered}}$ close to $1.0$ even when the regressor has zero explanatory power for fluctuations around the mean. Granger & Newbold (1974) formulated the $R^2 > DW$ spurious-regression rule of thumb for regressions with an intercept (centered $R^2$).
- **Recommendation**:
  Either gate `R2_ABOVE_DURBIN_WATSON` on `has_intercept=True` or evaluate the diagnostic against centered $R^2$.

---

### 2.4 Digest & Type-Hygiene Observations

#### Finding T-1 (Low): `digest.py` — Direct Calls to Digest Helpers Do Not Assert Absence of Control Characters
- **Location**: `src/riskquant/solutions/factor_regression/digest.py:130-318`
- **Mechanism**:
  `sample_digest`, `spec_digest`, and `unit_root_spec_digest` join caller-provided strings (`response_name`, `regressor_names`, `term_names`, `term_units`) using `\x1f` (ASCII unit separator) and `\x1e` (ASCII record separator). While `ObservedSeries` and `CombinationTerm` enforce `not _has_control(...)`, `digest.py` does not assert that its string arguments are free of `\x1e` and `\x1f` when called directly.
- **Recommendation**:
  Add a lightweight helper `_assert_no_separators(value: str)` in `digest.py` to make the bijective grammar self-defending even when invoked outside `api.py`.

#### Finding T-2 (Low): `run_identity/hashing.py` — 13 Strict Mypy Errors Require Gate Exemption
- **Location**: `src/riskquant/solutions/run_identity/hashing.py`
- **Mechanism**:
  Because `digest.py` imports `digest_named_arrays` from `riskquant.solutions.run_identity.hashing`, running `mypy --strict` on `src/riskquant/solutions/factor_regression` follows the import into `hashing.py` and exits with status `1` due to 13 pre-existing missing type parameters (`npt.NDArray` without `[Any]` and untyped parameters on 6 legacy helper functions).
- **Recommendation**:
  Execute a small `LIGHT` cleanup workstream on `src/riskquant/solutions/run_identity/hashing.py` to annotate those 6 functions (`hash_risk_factor_matrix`, `hash_parametric_risk_data`, `hash_compiled_book`, `digest_reporting_plan`, `hash_compiled_portfolio`, `hash_universe_mapping`) so `mypy --strict` exits with `0` across all factor-regression targets without a baseline exemption.

---

## Part III: Prioritized Remediation Roadmap

| ID | Severity | Category | Target File(s) | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **N-1** | **Medium** | Numerical / Exception | `factor_regression/wald.py` | Wrap `gap`, `m`, and `solve` in `np.errstate(over="ignore", invalid="ignore")` and check finiteness of `gap` and `m` before `np.linalg.cholesky(m)` so `m` overflow raises `RESTRICTION_NON_FINITE`. |
| **N-2** | **Medium** | Numerical / Exception | `factor_regression/fit.py` | Separate `t_stats` overflow (`coef / se` overflowing `float64` when `diagonal > 0`) from `_non_positive_variance` so valid positive variances are never reported with `value > bound = 0.0`. |
| **B-1** | **Medium** | Contract Immutability | `solutions/factor_regression/contracts.py` | Pass `order="C"` in `_frozen_array_copy` (`np.array(value, copy=True, order="C")`). |
| **B-2** | **Medium** | Boundary Hardening | `solutions/factor_regression/contracts.py` | Validate 1-D/2-D shapes, strict monotonicity of `date_ord`, finiteness of `levels`, and `window` consistency inside `AlignedLevels.__post_init__`. |
| **E-1** | **Medium** | Econometric Workflow | `factor_regression/adf.py`, `contracts.py` | Expose `resid`, `rss`, `dof`, and `durbin_watson` from `fit_core` on `CoreAdf` and `UnitRootResult` to support residual whitening and lag-selection criteria. |
| **N-3** | **Low** | Numerical Hygiene | `combination.py`, `rows.py` | Wrap ufunc arithmetic in `with np.errstate(over="ignore", invalid="ignore"):` so intentional overflow paths do not emit `RuntimeWarning`s. |
| **B-3** | **Low** | Boundary Hardening | `solutions/factor_regression/contracts.py` | Reject empty strings and Unicode `Cc` control characters in `Term.__post_init__`. |
| **B-4** | **Low** | Boundary Hardening | `solutions/factor_regression/contracts.py` | Reject duplicate `(series, transform)` entries in `ModelSpec.regressors` at construction time. |
| **E-2** | **Low** | Econometric Workflow | `solutions/factor_regression/api.py`, `contracts.py` | Expose the aligned combination series $z_t$ (`combination_values` or `combine_levels`). |
| **E-3** | **Low** | Econometric Workflow | `codes.py`, `rows.py`, `contracts.py` | Add `Transform.LOG` (log-level at `step=0`) to `fit_ols` when Engle-Granger cointegration (`O-4`) is introduced. |
| **E-4** | **Low** | Econometric Diagnostics | `factor_regression/diagnostics.py` | Gate `R2_ABOVE_DURBIN_WATSON` on `has_intercept=True` or use centered $R^2$. |
| **T-1** | **Low** | Digest Hygiene | `solutions/factor_regression/digest.py` | Assert absence of `\x1e` and `\x1f` in `digest.py` string packing helpers. |
| **T-2** | **Low** | Type Hygiene | `solutions/run_identity/hashing.py` | Add strict type annotations to the 6 legacy helpers in `hashing.py` to retire the `mypy` baseline workaround. |