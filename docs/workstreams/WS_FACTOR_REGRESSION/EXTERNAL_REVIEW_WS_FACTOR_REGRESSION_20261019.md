# Independent Review: Landed Factor Regression Package (`RiskQuant`)

External review of landed packages:
- `riskquant.factor_regression`; and
- `riskquant.solutions.factor_regression`.

---

## A. Executive Assessment

1. **Is the package statistically and numerically credible for what it claims to implement?**
**Yes, within its strictly delimited scope.** The package implements single-equation Ordinary Least Squares (OLS) with IID, HC0 (White), and HAC (Newey-West with Bartlett kernel) standard errors, supplemented by linear hypothesis testing via a joint Wald test. Numerically, the implementation is high quality: it uses power-of-two exponent scaling to avoid intermediate float64 overflow/underflow in column norms and residual quadratic forms, solves OLS directly from a thin SVD of the column-normalized design matrix, computes HC0/HAC covariance matrices through factorized projections ($Q$) rather than sandwiching rounding errors, and computes the Wald statistic via Cholesky decomposition of the restriction covariance without explicit matrix inversion.


2. **Can it meaningfully investigate the THG/USD-reference hypothesis?**
**Only partially.** It can estimate the short-run return sensitivities ($r_{P,t} = c + \gamma_Z r_{Z,t} + \gamma_M r_{M,t} + u_t$) and formally test the joint hypothesis of exact proportionality ($c=0, \gamma_Z=1, \gamma_M=1$) using heteroskedasticity- and autocorrelation-consistent inference. It can also fit static level regressions ($P_t = \alpha + \beta Q_t + \epsilon_t$) and inspect basic fit metrics ($R^2$, Durbin-Watson). However, it cannot conduct cointegration analysis, error-correction modeling (ECM), unit-root testing, rolling/recursive stability testing, or out-of-sample residual tail/stress evaluation.


3. **Which specific business questions can it answer today?**
* Empirical linear sensitivity of THG returns to USD-reference returns and USDEUR FX returns.


* Joint Wald test of parameter restrictions (e.g., $c=0, \gamma_Z=1, \gamma_M=1$).


* Static in-sample OLS fit on price levels ($P_t$ on $Q_t$) and evaluation of the Granger-Newbold rule of thumb ($R^2 > DW$).


* Multi-period overlapping return sensitivity with HAC Bartlett correction for induced MA($q$) errors.




4. **Which conclusions would be unjustified?**
* Concluding that THG and the USD reference are cointegrated based on levels $R^2$ or failure to trigger the `R2_ABOVE_DURBIN_WATSON` heuristic.


* Inferring that the relationship is stable over time or across stressed market regimes (no structural break tests, recursive residuals, or rolling windows are implemented).


* Concluding that residual basis risk is negligible for risk management (no residual distribution analysis, extreme value tail metrics, or historical simulation P&L backtests are provided).


* Concluding that market participants price THG in USD (an unidentifiable economic proposition from observational time-series regressions).




5. **What must be corrected before relying on the analysis?**
No mathematical blocker exists in the implemented OLS/Wald code. However, to answer the business question safely, researchers **must not rely on the package in isolation**. Complementary tooling or scripts must provide stationarity testing (ADF/KPSS), Engle-Granger/Johansen cointegration testing, and residual stress analysis.



---

## B. Observed Implementation

### 1. Inventory of Modules and APIs

| Module Path | Layer | Role / Exports |
| --- | --- | --- |
| `src/riskquant/factor_regression/codes.py` | Core (Private) | Defines `np.int8` integer codes for transforms, covariance types, exclusions, error reasons, and diagnostics.

 |
| `src/riskquant/factor_regression/errors.py` | Core (Private) | Numeric, label-free exception classes subclassing `RiskQuantError`.

 |
| `src/riskquant/factor_regression/align.py` | Core (Private) | `align_core`: Inner-join date intersection across series within a half-open ordinal window `[start_ord, end_ord)`.

 |
| `src/riskquant/factor_regression/rows.py` | Core (Private) | `build_rows`: Builds term rows (`LEVEL`, `LOG_RETURN`, `SIMPLE_RETURN`, `DIFFERENCE`) over step $h$ and stride $s$.

 |
| `src/riskquant/factor_regression/design.py` | Core (Private) | `assemble_design`, `rank_gate`: Design assembly with optional intercept; column 2-norm scaling; thin SVD rank gate and condition number $\kappa$.

 |
| `src/riskquant/factor_regression/covariance.py` | Core (Private) | `coefficient_covariance`: Calculates $\text{Cov}(\hat{\beta})$ for `IID`, `HC0`, and `HAC_BARTLETT`.

 |
| `src/riskquant/factor_regression/diagnostics.py` | Core (Private) | `durbin_watson`, `diagnose`: Heuristic diagnostics (`CONDITION_ABOVE_THRESHOLD`, `LEVEL_TERM_PRESENT`, `R2_ABOVE_DURBIN_WATSON`, `OVERLAP_NOT_CORRECTED`).

 |
| `src/riskquant/factor_regression/fit.py` | Core (Private) | `fit_core`: Solves OLS via SVD; computes $R^2$, RSS, $t$-statistics, and two-sided $p$-values.

 |
| `src/riskquant/factor_regression/wald.py` | Core (Private) | `wald_core`: Joint Wald statistic $W$, $F$-statistic, asymptotic $\chi^2$ and $F$ $p$-values via Cholesky factor of $R \hat{V} R^T$.

 |
| `src/riskquant/solutions/factor_regression/contracts.py` | Facade (Public) | Immutable dataclasses: `ObservedSeries`, `Term`, `Transform`, `CovarianceSpec`, `CovarianceKind`, `ModelSpec`, `AlignedLevels`, `OlsResult`, `WaldResult`, `Diagnostic`, `Exclusion`.

 |
| `src/riskquant/solutions/factor_regression/labels.py` | Facade (Private) | Bijective mappings between core `int8` codes and human-readable string names.

 |
| `src/riskquant/solutions/factor_regression/digest.py` | Facade (Private) | Cryptographic SHA-256 digest helpers for provenance tracking (`row_digest`, `sample_digest`, `spec_digest`, `restriction_digest`).

 |
| `src/riskquant/solutions/factor_regression/errors.py` | Facade (Public) | Named facade exceptions chained from core errors.

 |
| `src/riskquant/solutions/factor_regression/api.py` | Facade (Public) | Public API boundary: `align_levels`, `fit_ols`, `wald_test`.

 |

### 2. External Dependencies

* **`numpy`**: Dense linear algebra, SVD, indexing, bit-level manipulations (`frexp`, `ldexp`).


* **`scipy.stats`**: Reference distributions for inference: `stats.t.sf`, `stats.chi2.sf`, and `stats.f.sf`.


* Standard library: `math`, `operator`, `hashlib`, `dataclasses`, `typing`.


* Repository internal: `riskquant.exceptions.RiskQuantError`, `riskquant.solutions.run_identity.hashing.digest_named_arrays`.



---

## C. Mathematical and Statistical Correctness

### 1. Regression Estimation and SVD Solve

* **OLS Estimation:** In `src/riskquant/factor_regression/fit.py`, the coefficients are estimated from the thin SVD of the column-normalized design matrix $X_s = X D^{-1} = U \Sigma V^T$, where $D = \text{diag}(\Vert{}X_{\cdot, j}\Vert{}_2)$:



$$\hat{\gamma} = V \Sigma^{-1} U^T y, \quad \hat{\beta} = D^{-1} \hat{\gamma}$$



This avoids forming $X^T X$ and is invariant to column scale disparity.


* **Goodness-of-Fit:** $R^2$ is correctly centered when an intercept is present ($1 - \frac{\text{RSS}}{\sum (y_i - \bar{y})^2}$) and uncentered ($1 - \frac{\text{RSS}}{\sum y_i^2}$) when fitted without an intercept (`fit.py`, lines 268–272).


* **Degrees of Freedom:** Degrees of freedom are strictly enforced as $\text{dof} = R - k \ge 1$ (`rank_gate` enforces $R \ge k+1$).



### 2. Covariance and Standard Error Formulation

In `src/riskquant/factor_regression/covariance.py`, covariance estimation is structured to avoid squaring the condition number $\kappa$.

* Setting $P = \Sigma^{-1} V^T$ and $B = (X_s^T X_s)^{-1} = P^T P$, the bread-projection matrix is $X_s B = U P$.


* Defining row $t$ of $Q$ as $e_t \cdot (u_t P)$, the meat matrix satisfies:

$$B M_0 B = Q^T Q, \quad B G_j B = Q_{j:}^T Q_{:R-j}$$


* The HAC Bartlett formulation is:

$$\hat{V}(\hat{\gamma}) = Q^T Q + \sum_{j=1}^L \left(1 - \frac{j}{L+1}\right) \left( Q_{j:}^T Q_{:R-j} + Q_{:R-j}^T Q_{j:} \right)$$


* The unscaled covariance is $\hat{V}(\hat{\beta}) = D^{-1} \hat{V}(\hat{\gamma}) D^{-1}$.


* **Limitation:** The HAC estimator uses a fixed, user-specified truncation lag $L$ without small-sample degrees-of-freedom corrections ($R / (R-k)$) and lacks data-dependent automatic bandwidth selection (such as Andrews 1991 or Newey-West 1994).



### 3. Hypothesis Testing (Wald Test)

In `src/riskquant/factor_regression/wald.py`:

* Tests the joint restriction $R \beta = r$ for $R \in \mathbb{R}^{J \times K}$.


* Before solving, rows of $R$ are normalized by their 2-norms to prevent artificial ill-conditioning.


* Rather than inverting $M = R \hat{V} R^T$, it computes the lower-triangular Cholesky factor $L$ such that $L L^T = M$, solves $L z = (R \hat{\beta} - r)$, and computes $W = z^T z$.


* $F = W / J$ is computed alongside both asymptotic $\chi^2(J)$ and finite-sample $F(J, R-k)$ $p$-values.


* **Statistical Nuance:** The $F$-distribution assumption is exact only under Gaussian IID errors. Under HC0 or HAC, $W$ converges asymptotically to $\chi^2(J)$, while $F$ is an ad-hoc finite-sample heuristic. This is properly documented in the docstrings (`wald.py`, lines 181–183).



### 4. Stationarity, Cointegration, and Unit Roots

* **Demonstrated Gap:** The package contains **no unit-root tests** (ADF, Phillips-Perron, KPSS) and **no cointegration estimators or tests** (Engle-Granger, Johansen, Phillips-Ouliaris).


* **Spurious Regression Heuristic:** In `diagnostics.py`, the package checks `R2_ABOVE_DURBIN_WATSON` (Granger & Newbold 1974: $R^2 > DW$ in level regressions). As explicitly noted in docstrings (`diagnostics.py`, line 17), this is a crude heuristic, not a statistical test. It cannot confirm cointegration or reject spurious drift.



### 5. Multi-Horizon and Overlapping Returns

* In `rows.py`, multi-period returns over step $h$ with stride $s$ calculate the overlap order $q = \lceil h/s \rceil - 1$.


* In `diagnostics.py`, if $q > 0$ and the covariance estimator is not HAC with $L \ge q$, the diagnostic `OVERLAP_NOT_CORRECTED` is raised. This is mathematically sound, as an overlapping return aggregation over step $h$ introduces at least an $\text{MA}(q)$ error structure.



---

## D. Business-Question Coverage

| Business Question | Implementation Status | Evidence & API Entry Points |
| --- | --- | --- |
| **1. Does THG exhibit an empirical relationship with the FX-converted USD reference?** | **SUPPORTED** | `fit_ols(levels, spec)` with `Transform.LEVEL` or `Transform.LOG_RETURN`. Provides coefficients, standard errors, $t$-statistics, and $R^2$.

 |
| **2. Is the apparent levels relationship potentially spurious?** | **PARTIALLY SUPPORTED** | Diagnostic `R2_ABOVE_DURBIN_WATSON` flags severe non-stationarity symptoms. However, formal unit-root tests (ADF/KPSS) on $P_t, Q_t$ are absent.

 |
| **3. Is there evidence of a stable long-run equilibrium relationship?** | **NOT SUPPORTED** | Cointegration tests (Engle-Granger, Johansen) and Error Correction Models (ECM) are completely absent from the codebase.

 |
| **4. How sensitive are THG returns to the USD reference and FX separately?** | **SUPPORTED** | `fit_ols` supports multivariate return regressions ($r_{P,t} = c + \gamma_Z r_{Z,t} + \gamma_M r_{M,t} + u_t$) with `HC0` or `HAC_BARTLETT` covariance.

 |
| **5. Does a strict proportional model fit the data?** | **SUPPORTED** | `wald_test(result, restriction, rhs)` evaluates the linear restrictions $c=0, \gamma_Z=1, \gamma_M=1$ jointly.

 |
| **6. Is the relationship stable across regimes?** | **NOT SUPPORTED** | No rolling window OLS, recursive residuals, CUSUM tests, or Chow break tests are implemented.

 |
| **7. Does the proposed representation reproduce relevant risk?** | **PARTIALLY SUPPORTED** | Residual array $e_t$ is accessible via `OlsResult.resid`. However, tail risk diagnostics (VaR/ES of residuals, max drawdown of basis, tail dependence) are absent.

 |
| **8. Can we conclude that traders actually price THG in USD?** | **NOT IDENTIFIABLE** | Observational factor regressions establish statistical correlation and linear projection weights, not behavioral price formation or trader intent.

 |

---

## E. Adversarial and Analytical Validation Results

The mathematical and algorithmic properties were evaluated against the 10 adversarial scenarios:

1. **Exact Proportional Relationship ($P_t = \beta Z_t M_t$):**
*Result:* $\ln P_t = \ln \beta + \ln Z_t + \ln M_t$. Log-return regression $r_P = \gamma_Z r_Z + \gamma_M r_M$ yields $\hat{\gamma}_Z = 1.0, \hat{\gamma}_M = 1.0, \hat{c} = 0.0$ and $\text{RSS} \approx 0$.
*Implementation Trap:* Triggers `CoreDegenerateFitError(DEGENERATE_EXACT_FIT)` because $\text{RSS} \le (R \cdot \epsilon_{\text{mach}})^2 \sum y^2$ (`fit.py`, line 241). In synthetic unit tests with zero noise, the code deliberately rejects exact fit to prevent division by zero in downstream covariance matrices. Realistic data with noise passes normally.


2. **Additive Intercept ($P_t = \alpha + \beta Z_t M_t$ with $\alpha \ne 0$):**
*Result:* Log returns $r_{P,t} \approx \frac{\beta Q_{t-1}}{\alpha + \beta Q_{t-1}} r_{Q,t}$. Slopes $\gamma_Z, \gamma_M$ deviate systematically from $1.0$, and the intercept in returns becomes non-zero. The package's `wald_test` correctly rejects the hypothesis $c=0, \gamma_Z=1, \gamma_M=1$.


3. **Independent Driftless Random Walks ($I(1)$ series):**
*Result:* Spurious regression occurs. Levels OLS exhibits moderate to high $R^2$, $t$-statistics that diverge as $\sqrt{R}$, and $DW \to 0$. The diagnostic `R2_ABOVE_DURBIN_WATSON` fires as expected (`diagnostics.py`, line 214). However, if both series have deterministic drift, $DW$ may remain moderate while $R^2$ is inflated, demonstrating the insufficiency of heuristic screening alone.


4. **Cointegrated Processes ($P_t = \alpha + \beta Q_t + \epsilon_t, \epsilon_t \sim \text{AR}(1)$):**
*Result:* Superconsistent estimates of $\beta$ in levels. However, standard OLS standard errors underestimate parameter uncertainty due to residual autocorrelation. If `HAC_BARTLETT` is selected with an inadequate lag, inference remains distorted. The lack of an Augmented Dickey-Fuller (ADF) test prevents formal verification that $\epsilon_t \sim I(0)$.


5. **Near-Collinear Factors ($\text{Corr}(r_Z, r_M) \approx 0.9999$):**
*Result:* Tested by `rank_gate` (`design.py`, line 232). When $\kappa > \text{threshold}$, the diagnostic `CONDITION_ABOVE_THRESHOLD` correctly fires (`diagnostics.py`, line 210). If singular values fall below $\text{rank\_rtol} \cdot s_0$, `CoreRankError(RANK_DEFICIENT)` is cleanly raised before solving.


6. **Structural Break in Sensitivity:**
*Result:* Fitting full-sample OLS produces an average beta that fits neither regime well. Residuals exhibit persistent regime-dependent clustering. Because recursive/rolling window estimators and CUSUM tests are absent, the model silently delivers an in-sample compromise.


7. **Large Residual Basis Risk:**
*Result:* Regressing $r_P$ on $r_Z, r_M$ can produce $\hat{\gamma}_Z \approx 1, \hat{\gamma}_M \approx 1$ with $R^2 = 0.30$. The Wald test accepts the hypothesis $\gamma = 1$, but $70\%$ of variance remains unexplained basis risk. Relying on coefficient point estimates without examining $\hat{\sigma}_e$ or residual quantiles would understate proxy risk in downstream VaR models.


8. **Calendar Misalignment and Illiquidity:**
*Result:* `align_levels` strictly intersects dates (`align.py`, line 186). If series $Z$ has public holidays that $P$ does not share, all discordant dates are dropped. If one market is stale and updates prices with a one-day lag, the contemporaneous return regression attenuates beta toward zero (Epps effect). The package lacks lead-lag cross-correlation diagnostics.


9. **Zero or Negative Prices under Log Transforms:**
*Result:* Handled robustly by `_validate_domains` (`rows.py`, lines 180–193). Any level $\le 0$ under `LOG_RETURN` immediately raises `CoreDomainError`, mapped to `TransformDomainError` with the offending date and level.


10. **Reversed FX Quotation Direction:**
*Result:* If $M$ is provided as USD per EUR ($1/M$) instead of EUR per USD ($M$), the theoretical return relationship becomes $r_P = r_Z - r_M$. A regression of $r_P$ on $r_Z$ and $r_M$ estimates $\hat{\gamma}_M \approx -1.0$. `wald_test` on $(1, 1)$ would reject decisively, alerting the researcher.



---

## F. Findings

### Finding F-01: Absence of Unit-Root and Cointegration Testing Framework

* **Severity:** HIGH
* **Status:** OBSERVED
* **Affected Business Question:** Questions 2 and 3 (Spurious levels regression and long-run equilibrium).


* **Exact Code Location:** `src/riskquant/factor_regression/diagnostics.py`

* **Evidence:** The only diagnostic for spurious correlation is `R2_ABOVE_DURBIN_WATSON` (`diagnostics.py`, lines 213–214), which is explicitly documented as a heuristic. There are no implementations of Augmented Dickey-Fuller (ADF), Phillips-Perron, KPSS, or Engle-Granger two-step cointegration tests.


* **Consequence:** Researchers running levels regressions ($P$ on $Q$) cannot determine whether $t$-statistics are spurious due to $I(1)$ stochastic trends or whether the residual represents a genuine stationary error-correction equilibrium.


* **Minimum Surgical Correction:** Implement an ADF unit-root test function on 1-D residuals and series, or provide an explicit bridge to `statsmodels.tsa.stattools` in exploratory workflows.



### Finding F-02: Lack of Dynamic Stability and Structural Break Analysis

* **Severity:** HIGH
* **Status:** OBSERVED
* **Affected Business Question:** Question 6 (Stability across regimes and stressed periods).


* **Exact Code Location:** `src/riskquant/solutions/factor_regression/api.py` (`fit_ols`)


* **Evidence:** `fit_ols` only accepts a static sample window `[start_ord, end_ord)`. There are no rolling-window fits, recursive residual tracking, or Chow/Sup-Wald parameter break tests.


* **Consequence:** Changes in market microstructure, shipping disruptions, or regulatory adjustments that shift the THG pricing basis across regimes will be masked by full-sample averaging.


* **Minimum Surgical Correction:** Add a rolling estimation wrapper or document the requirement to script sequential date windows externally using `fit_ols`.



### Finding F-03: Inability to Measure Residual Risk for Tail and Stress Performance

* **Severity:** MEDIUM
* **Status:** OBSERVED
* **Affected Business Question:** Question 7 (Adequacy for risk modeling without understating residual risk).


* **Exact Code Location:** `src/riskquant/solutions/factor_regression/contracts.py` (`OlsResult`)


* **Evidence:** `OlsResult` outputs raw residuals (`resid`), RSS, and $R^2$, but computes no residual distribution statistics (skewness, kurtosis, Jarque-Bera, residual quantiles, or max tracking errors).


* **Consequence:** A model with $\hat{\gamma}_Z \approx 1, \hat{\gamma}_M \approx 1$ can easily pass statistical Wald restrictions while exhibiting fat-tailed residual shocks that would cause significant P&L underestimation in VaR models.


* **Minimum Surgical Correction:** Expose residual descriptive statistics and tracking error quantiles in the results summary.



### Finding F-04: Fixed Lag Truncation in HAC Estimator Without Automatic Bandwidth Selection

* **Severity:** MEDIUM
* **Status:** OBSERVED
* **Affected Business Question:** Questions 4 and 5 (Multi-factor return sensitivity and restriction testing).


* **Exact Code Location:** `src/riskquant/factor_regression/covariance.py` (`coefficient_covariance`, lines 183–185)


* **Evidence:** `HAC_BARTLETT` requires a manually specified integer lag `covariance_lag` (`contracts.py`, `CovarianceSpec`). No automatic bandwidth selection (e.g., Newey-West plug-in $4(T/100)^{2/9}$) is supported.


* **Consequence:** An unguided researcher may select an arbitrary lag $L$, leading to either inflated Type I error (if $L$ is too small) or excessive variance/loss of test power (if $L$ is unnecessarily large).


* **Minimum Surgical Correction:** Provide a standard rule-of-thumb lag calculation in the facade or helper layer.



---

## G. Recommended Changes

### 1. Must Fix (Prerequisites for Defensible Decision-Making)

* **Do not modify the landed core package code directly** (the core OLS engine is mathematically clean and stable).


* **Mandate an External Econometric Prequalification Script:** Before adopting a proxy model based on `fit_ols`, researchers must run cointegration and unit-root diagnostics (e.g., via `statsmodels`). The landed package must not be cited as "proof of cointegration".



### 2. Should Fix (Material Improvements to Reliability)

* **Residual Distribution Analytics:** Add a post-processing helper function in `solutions.factor_regression` that computes residual quantiles (95th, 99th percentile), tracking error volatility, and maximum drawdown of the cumulative residual.


* **Rolling Window Utility:** Provide an orchestration helper that iterates over rolling date slices, invoking `fit_ols` to output beta stability paths.



### 3. Optional (Extensions)

* Formal Error Correction Model (ECM) specification if long-run mean reversion between THG and USD reference needs to be integrated into intraday trading strategies.


* Automatic Newey-West bandwidth calculation in `CovarianceSpec`.



### 4. No Change Required

* Core OLS solver (`fit_core`, SVD numerical decomposition).


* Covariance projection implementation ($Q$-matrix formulation).


* Cholesky-based Wald restriction test (`wald_core`).


* Date ordinal alignment and strict domain validation logic (`align_core`, `build_rows`).


* Content-addressed SHA-256 provenance hashing (`digest.py`).



---

## H. Proposed Investigation Using the Existing Package

The business question can be investigated systematically using the landed API through the following disciplined protocol:

### Step 1: Data Preparation and Alignment

* Retrieve daily price series:
* $P_t$: THG closing price in EUR (`EUR_THG`).
* $Z_t$: Reference asset closing price in USD (`USD_REF`).
* $M_t$: USDEUR spot rate (EUR per 1 USD) (`EUR_PER_USD`).


* Construct $Q_t = Z_t \cdot M_t$ (EUR equivalent reference price) externally as an additional series `EUR_REF_CONV`.
* Convert date axes to `int32` date ordinals and instantiate `ObservedSeries`:
```python
from riskquant.solutions.factor_regression.api import (
    CovarianceKind,
    CovarianceSpec,
    ModelSpec,
    ObservedSeries,
    Term,
    Transform,
    align_levels,
    fit_ols,
    wald_test,
)

series_p = ObservedSeries(name="EUR_THG", unit="EUR", date_ord=dates_p, values=prices_p)
series_z = ObservedSeries(name="USD_REF", unit="USD", date_ord=dates_z, values=prices_z)
series_m = ObservedSeries(
    name="EUR_PER_USD", unit="EUR/USD", date_ord=dates_m, values=fx_m
)
series_q = ObservedSeries(
    name="EUR_REF_CONV", unit="EUR", date_ord=dates_q, values=prices_q
)

aligned = align_levels(
    [series_p, series_z, series_m, series_q],
    start_ord=start_date,
    end_ord=end_date,
)
```



```

### Step 2: Multifactor Return Sensitivity and Joint Identity Testing
- Specify the 1-day log-return regression ($r_{P,t} = c + \gamma_Z r_{Z,t} + \gamma_M r_{M,t} + u_t$):
  ```python
  spec_returns = ModelSpec(
      response=Term(series="EUR_THG", transform=Transform.LOG_RETURN),
      regressors=(
          Term(series="USD_REF", transform=Transform.LOG_RETURN),
          Term(series="EUR_PER_USD", transform=Transform.LOG_RETURN),
      ),
      intercept=True,
      step=1,
      stride=1,
      covariance=CovarianceSpec(kind=CovarianceKind.HC0, lag=None),
      rank_rtol=1e-7,
      condition_threshold=30.0,
  )
  res_returns = fit_ols(aligned, spec_returns)

```

* Test the theoretical identity $c = 0, \gamma_Z = 1, \gamma_M = 1$:
```python
# Coefficients order: ['const', 'USD_REF', 'EUR_PER_USD']
restriction = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
rhs = np.array([0.0, 1.0, 1.0], dtype=np.float64)

wald_res = wald_test(res_returns, restriction, rhs)
```



```
- **Decision Rule:**
  - If $p_{\chi^2} < 0.01$, reject the exact proportional hypothesis[cite: 1, 2].
  - Inspect individual point estimates $\hat{\gamma}_Z$ and $\hat{\gamma}_M$ to see if one factor dominates or if THG under-reacts to FX[cite: 1, 2].

### Step 3: Multi-Day Overlapping Return Analysis (Checking Flow Horizon)
- If price adjustment takes more than one trading day due to cross-market liquidity frictions, fit 5-day overlapping returns ($h=5, s=1$):
  ```python
  spec_5d = ModelSpec(
      response=Term(series="EUR_THG", transform=Transform.LOG_RETURN),
      regressors=(
          Term(series="USD_REF", transform=Transform.LOG_RETURN),
          Term(series="EUR_PER_USD", transform=Transform.LOG_RETURN),
      ),
      intercept=True,
      step=5,
      stride=1,
      covariance=CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=5),
      rank_rtol=1e-7,
      condition_threshold=30.0,
  )
  res_5d = fit_ols(aligned, spec_5d)

```

* Verify whether $\hat{\gamma}_Z$ and $\hat{\gamma}_M$ converge closer to $1.0$ at a weekly horizon.



### Step 4: Levels Tracking and Spurious Correlation Screening

* Fit levels regression $P_t = \alpha + \beta Q_t + \epsilon_t$:
```python
spec_levels = ModelSpec(
    response=Term(series="EUR_THG", transform=Transform.LEVEL),
    regressors=(Term(series="EUR_REF_CONV", transform=Transform.LEVEL),),
    intercept=True,
    step=0,
    stride=1,
    covariance=CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=10),
    rank_rtol=1e-7,
    condition_threshold=30.0,
)
res_levels = fit_ols(aligned, spec_levels)
```



```
- **Evaluation:**
  - Check `res_levels.diagnostics`: Does `R2_ABOVE_DURBIN_WATSON` fire? If yes, the relationship is exposed to Granger-Newbold spurious regression suspicion[cite: 1, 2].
  - Check whether $\hat{\alpha}$ is economically near zero and $\hat{\beta} \approx 1$[cite: 1, 2].

### Step 5: External Residual Stress Evaluation (Risk Proxy Suitability)
- Extract residuals: $e_t = \text{res\_returns.resid}$[cite: 1, 2].
- Compute:
  - $\text{Tracking Error Volatility} = \text{std}(e_t) \times \sqrt{252}$.
  - Tail risk: $\text{VaR}_{99\%}(e_t)$ and $\text{ES}_{99\%}(e_t)$.
  - Maximum 1-day basis shock: $\max \vert{}e_t\vert{}$.
- **Decision Rule for Risk Modeling:** Even if $\hat{\gamma}_Z = 1, \hat{\gamma}_M = 1$, if $\text{std}(e_t) / \text{std}(r_P) > 0.30$, substituting $(Z, M)$ for $THG$ without an idiosyncratic residual risk term will severely understate tail risk[cite: 1, 2].

---

## I. Final Recommendation

### **SUITABLE WITH EXPLICIT LIMITATIONS**

The package is **suitable** for estimating empirical return sensitivities, testing exact coefficient restrictions, and verifying numerical stability under robust covariance estimators[cite: 1, 2].

However, it is subject to the following **explicit limitations**:
1. **No Cointegration Proof:** It cannot prove or disprove cointegration or long-run equilibrium[cite: 1, 2].
2. **No Regime Invariance:** It cannot establish whether the proxy relationship holds during market stress or structural breaks without external rolling scripting[cite: 1, 2].
3. **No Basis Risk Clearance:** A high $R^2$ or failure to reject $(1, 1)$ loadings does not authorize dropping idiosyncratic residual risk in VaR or capital calculations[cite: 1, 2].
4. **No Behavioral Identification:** It cannot prove that market participants formulate prices by reference to USD; it only measures observable statistical co-movement[cite: 1, 2].

```
