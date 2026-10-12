# WS_FACTOR_REGRESSION: Appendix B, slice tests and gates

Path: docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md
Revision: 2
Status: DRAFT
Used by: S1 to S7
Label rule: every test below is Proposed. Tolerances reflect float64 and the sandbox evidence of `A_NUMERICS.md` A.7.

Gate rule: WORKFLOW.md section 6 order. Run from the repository root in PowerShell. `$f` lists the changed files of the slice.
No repository import-all command exists (F-17); the import line imports every new module of the slice. No line is a placeholder.

## S1

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T5 | three `int32` date arrays with different calendars | kept dates equal a Python set intersection; kept positions index the inputs exactly | fill or value change |
| S1.T6 | window cuts and missing dates | per input, kept plus excluded equals length; reason codes correct; no duplicate record | unrecorded exclusion |
| S1.T7 | empty intersection; `start_ord >= end_ord` | `CoreAlignmentError` with integer payload | empty sample |
| S1.T9 | every class in `factor_regression/errors.py` | subclass of `RiskQuantError`; no `str` field besides the message | foreign base, labels in core |
| S1.T10 | `"factor_regression"` in the package tuple; run `test_core_packages_obey_the_mechanical_rules` | passes with the core present | core outside the FORBIDDEN rules |
| S1.T11 | `test_factor_regression_import_direction`: AST scan (any import, `TYPE_CHECKING` blocks included) | core imports none of the C.21 forbidden modules and not the facade; no file under `src/` or `scripts/` outside the two packages imports the core | wrong import direction |
| S1.T12 | `test_every_init_file_is_empty`, `test_every_module_declares_its_repository_relative_path`, `tests/test_no_stale_path_comments.py` | pass with the core files present | new files outside existing gates |
| S1.T13 | `test_factor_regression_core_holds_no_str_type`: AST scan of the core | no `str` in any annotation; no class derived from `Enum` | labels in the core (C.22) |

Core exception names (Proposed): `CoreInputError`, `CoreAlignmentError`, `CoreParameterError`, `CoreDomainError`, `CoreRankError`,
`CoreSampleSizeError`, `CoreDegenerateFitError`, `CoreRestrictionError`. Facade mapping (S7): `SeriesContractError`, `AlignmentError`,
`ModelSpecError`, `TransformDomainError`, `RankDeficiencyError`, `InsufficientObservationsError`, `DegenerateFitError`, `RestrictionError`.

Gate block:

```powershell
$f = "src/riskquant/factor_regression/__init__.py", "src/riskquant/factor_regression/errors.py", "src/riskquant/factor_regression/codes.py", "src/riskquant/factor_regression/align.py", "tests/factor_regression/test_errors.py", "tests/factor_regression/test_align.py", "tests/test_architecture.py"
$t = "tests/factor_regression/test_errors.py", "tests/factor_regression/test_align.py", "tests/test_architecture.py", "tests/test_no_stale_path_comments.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.factor_regression.errors, riskquant.factor_regression.codes, riskquant.factor_regression.align"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S2

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | two calendars; aligned-first rows versus per-series-first returns | core equals an explicit-loop oracle; per-series-first differs | return intervals that differ across series |
| S2.T2 | `h=1, s=1`; LOG, SIMPLE, DIFFERENCE codes | equals `math.log`, ratio and difference per element | wrong endpoint formula |
| S2.T3 | `h=3, s=1` and `h=3, s=3` and `h=1, s=2` | `R` equals the formula; row indices match; `q` is 2, 0, 0; unused indices recorded | wrong stride, overlap or dropped date |
| S2.T4 | LEVEL rows; `h=0` with a return code; `h>0` with LEVEL only; `s=0` | LEVEL equals end values; the other three raise `CoreParameterError` | hidden step, wrong row interval |
| S2.T5 | LOG column with a level 0.0 or negative; LEVEL column with a negative level | LOG raises `CoreDomainError` with column index, aligned row index and level; LEVEL accepted | log of non-positive level |
| S2.T6 | SIMPLE column with level 0.0; with a negative level | zero raises; negative accepted | division by zero |
| S2.T7 | DIFFERENCE column with zero and negative levels | accepted | wrong domain |
| S2.T8 | `h` at or above the aligned count | `CoreSampleSizeError` | empty row set |
| S2.T9 | `Q = Z*S`; LOG and SIMPLE rows | LOG: `abs(r_Q - r_Z - r_S) <= 1e-15`; SIMPLE: `sr_Q - sr_Z - sr_S` equals `sr_Z*sr_S` to 1e-15 | wrong claim of additivity |

Gate block:

```powershell
$f = "src/riskquant/factor_regression/rows.py", "tests/factor_regression/test_rows.py"
$t = "tests/factor_regression/test_rows.py", "tests/test_architecture.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.factor_regression.rows"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S3

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | a zero column; one column multiplied by 1e9 | zero column raises `CoreRankError` with its index; `kappa` unchanged by the multiplier | missing scaling |
| S3.T2 | columns `c1, c2, c1+c2, c4` | rank 3 of 4; involved indices exactly 0, 1, 2; relation is proportional to `(1, 1, -1, 0)` | wrong or missing dependent columns |
| S3.T3 | log-return design `r_Z, r_S, r_Q` from `Q = Z*S`, vol 1% and 0.5%, 250 rows, `rank_rtol=1e-10` | `CoreRankError` with indices 0, 1, 2; `matrix_rank` also gives 2 | exact identity not detected |
| S3.T4 | same at vol 0.1% and 0.05% with `rank_rtol = 250*eps` | no raise; `kappa >= 1e12` reported | unflagged near-singular design (F-12) |
| S3.T5 | unit columns with `rho = 0.99`, one column scaled by 1e6 | `kappa` equals `sqrt((1+rho)/(1-rho))` to 1e-12 | wrong conditioning formula |
| S3.T6 | threshold None, below `kappa`, above `kappa`, 1.0, 0.5 | no flag, flag, no flag, `CoreParameterError`, `CoreParameterError` | silent default threshold |
| S3.T7 | simple-return `sr_Z, sr_S, sr_Q`, vol 1% and 0.5%, 250 rows, `rank_rtol=1e-10` | rank 3; `kappa` between 3e2 and 7e2; threshold 100 flags | rank-only gate misses the design |
| S3.T8 | residuals `(1, -1, 1, -1)` | Durbin-Watson equals 3 | wrong statistic |
| S3.T9 | LEVEL present with `r_squared` above and below DW; no LEVEL term | LEVEL code; R2 code only when above; no code | silent level regression |
| S3.T10 | `h=5, s=1` with IID, HC0, HAC lag 3, HAC lag 4; `h=5, s=5` | code, code, code, none, none | understated standard errors |

Gate block:

```powershell
$f = "src/riskquant/factor_regression/design.py", "src/riskquant/factor_regression/diagnostics.py", "tests/factor_regression/test_design.py", "tests/factor_regression/test_diagnostics.py"
$t = "tests/factor_regression/test_design.py", "tests/factor_regression/test_diagnostics.py", "tests/test_architecture.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.factor_regression.design, riskquant.factor_regression.diagnostics"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S4

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | 5 rows, intercept and 2 regressors | coefficients, `rss`, `resid`, `fitted`, `dof` equal Fraction values to 1e-12 relative | wrong solve |
| S4.T2 | mean to sd 1e7, 250 rows, intercept | slope within 1e-8, intercept within 1e-6 of the Fraction value | unscaled solve loses a rank (F-11) |
| S4.T3 | IID on the 5-row case, `dof = 2` | `se` equals Fraction value; p equals `1 - abs(t)/sqrt(2 + t^2)` | wrong scale or tail |
| S4.T4 | HC0 on the 5-row case | `cov` equals Fraction sandwich to 1e-12 | wrong meat |
| S4.T5 | HAC lag 0 versus HC0; HAC lag 1 versus an explicit Fraction double sum | equal to 1e-12 relative | wrong Bartlett weight |
| S4.T6 | lag 5 on 5 rows; lag with IID; negative lag | `CoreParameterError` | invalid lag |
| S4.T7 | `y = 1 + 2x` exactly on small integer `x`; the same with `R` of 30, 250, 2500 on non-integer `x`; `y = x + 1e-8 noise` | the exact fits raise `CoreDegenerateFitError` whether `rss` is 0.0 or not; the noisy fit returns finite values | rounding-dependent outcome; absurd t value |
| S4.T8 | all-zero `y` with intercept only; core result array check; caller mutation after fit | `CoreDegenerateFitError`; arrays read-only; no accepted, approved or valid field; result unchanged | NaN in result, aliasing |

Gate block:

```powershell
$f = "src/riskquant/factor_regression/covariance.py", "src/riskquant/factor_regression/fit.py", "tests/factor_regression/test_covariance.py", "tests/factor_regression/test_fit.py"
$t = "tests/factor_regression/test_covariance.py", "tests/factor_regression/test_fit.py", "tests/test_architecture.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.factor_regression.covariance, riskquant.factor_regression.fit"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S5

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | `alpha = 0` and `beta = 1` jointly, `dof = 2` | `W` equals the Fraction value; `exp(-W/2)` and `1/(1+F)` equal the p-values | wrong joint statistic |
| S5.T2 | correlated regressors, `J = 2` | `W` differs from the sum of the squared t values and matches the oracle | separate t-tests used |
| S5.T3 | `J = 1`, `beta_j = 0` | `F` equals `t_j^2` | wrong single restriction |
| S5.T4 | duplicate row in `R`; wrong shape; NaN in `r` | `CoreRestrictionError` | degenerate restriction accepted |
| S5.T5 | HC0 core fit | `W` uses the HC0 `cov` and equals the oracle | classical covariance used by mistake |

Gate block:

```powershell
$f = "src/riskquant/factor_regression/wald.py", "tests/factor_regression/test_wald.py"
$t = "tests/factor_regression/test_wald.py", "tests/test_architecture.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.factor_regression.wald"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S6

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | empty name, empty unit, control character in a name or unit | `SeriesContractError` | unusable labels in the digest |
| S6.T2 | int64 dates, unsorted dates, duplicate dates | `SeriesContractError`, no repair | silent sort or cast |
| S6.T3 | NaN or inf value | error names series and `date_ord` | non-finite value in a fit |
| S6.T4 | float32 values; caller mutates its arrays after construction | error for float32; stored arrays unchanged and read-only | aliasing, silent cast |
| S6.T5 | each omitted `ModelSpec` field | `TypeError` | inferred parameter |
| S6.T6 | each invalid value of C.5 except the unknown series | `ModelSpecError` | invalid spec accepted |
| S6.T7 | same inputs in-process and in a fresh subprocess | three digests equal | non-deterministic digest |
| S6.T8 | five single mutations: one-ulp change in `y`; term rename; unit change; window change; lag change | `y`: sample only. Rename: sample and spec. Unit: spec only. Window: spec, and row and sample when rows change. Lag: spec only. | digest misses an input |
| S6.T9 | rebuild the A.6 bytes with `hashlib` | equals the facade digests | canonical bytes differ from the spec |
| S6.T10 | every code in `codes.py` | exactly one name in `labels.py`, and the reverse map returns the code | unlabelled or ambiguous code |
| S6.T11 | every class in facade `errors.py` | subclass of `RiskQuantError` | foreign exception base |
| S6.T12 | build `OlsResult` and `WaldResult` directly | arrays read-only; frozen; no accepted, approved or valid field | mutable result, approval semantics |
| S6.T13 | the three tests of S1.T12 | pass with the facade files present | facade outside existing gates |
| S6.T14 | AST scan of the facade | only `digest.py` imports `riskquant.solutions.run_identity.hashing`; no direct `riskquant.config` import | digest logic spread, C.21 breach |

Gate block:

```powershell
$f = "src/riskquant/solutions/factor_regression/__init__.py", "src/riskquant/solutions/factor_regression/contracts.py", "src/riskquant/solutions/factor_regression/errors.py", "src/riskquant/solutions/factor_regression/labels.py", "src/riskquant/solutions/factor_regression/digest.py", "tests/solutions/factor_regression/test_contracts.py", "tests/solutions/factor_regression/test_digest.py"
$t = "tests/solutions/factor_regression/test_contracts.py", "tests/solutions/factor_regression/test_digest.py", "tests/test_architecture.py", "tests/test_no_stale_path_comments.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.solutions.factor_regression.contracts, riskquant.solutions.factor_regression.errors, riskquant.solutions.factor_regression.labels, riskquant.solutions.factor_regression.digest"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

## S7

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S7.T1 | `align_levels` on three named series | exclusions carry the series name, date and reason name; matches the core record count | lost names |
| S7.T2 | LOG term with a level 0.0 | `TransformDomainError` names series, transform, `date_ord` and level; `__cause__` is the core error | cell address lost at the facade |
| S7.T3 | `r_Q` with `r_Z`, `r_S` | `RankDeficiencyError` names the three columns | names lost |
| S7.T4 | Z-only, S-only, Z+S on one `AlignedLevels`; a different step | equal `row_digest` for the first three; different for the last | invalid model comparison |
| S7.T5 | `wald_test`; change `r`; compare `R` and `r` before and after | `restriction_digest` changes; result digests copied; `R` and `r` unchanged | unbound test result, input mutation |
| S7.T6 | motivating case and renamed series `A, B, C` | bitwise equal numbers; LEVEL fit carries a `Diagnostic` with code `LEVEL_TERM_PRESENT` | hard-coded names; lost diagnostics |
| S7.T7 | `api.__all__` | lists `align_levels`, `fit_ols`, `wald_test`, the contract types and facade exceptions; every name resolves; no other facade module is named public | second public surface |
| S7.T8 | a term names a series absent from `AlignedLevels` | `ModelSpecError` naming the series | unknown term accepted |

Motivating case (fixture only): P in EUR, Z in USD, S in EUR per USD, `Q = Z*S`, synthetic data. Fits: P on Q in levels;
`r_P` on `r_Z, r_S` in log returns (a design with `r_Q` raises `RankDeficiencyError`); intercept-only `ln(P/Q)` built by the caller
from aligned levels.

Gate block:

```powershell
$f = "src/riskquant/solutions/factor_regression/api.py", "tests/solutions/factor_regression/test_api.py"
$t = "tests/solutions/factor_regression/test_api.py", "tests/test_architecture.py"
& {
  "== py_compile";          python -m py_compile $f;  "exit=$LASTEXITCODE"
  "== import";              python -c "import riskquant.solutions.factor_regression.api"; "exit=$LASTEXITCODE"
  "== ruff check";          ruff check $f;            "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f;   "exit=$LASTEXITCODE"
  "== mypy --strict";       mypy --strict $f;         "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;             "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;                "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```
