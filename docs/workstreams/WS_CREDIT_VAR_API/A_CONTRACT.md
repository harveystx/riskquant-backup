# WS_CREDIT_VAR_API / A_CONTRACT: target shapes of request and result

Path: docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md
Revision: 5
Used by: S1, S2, S5
Snapshot: SNAPSHOT commit=925533b dirty=false content_hash=046ed5dddb6305d3

Every row is [Proposed] unless the Change column says "none". "none" means the
field keeps its c060f06 type, constraint and JSON form. An optional input field
(`None` in Python, `null` in JSON) is an absent value. It is not an error state.

## 1. Model configuration

| Group | ConfigDict |
|---|---|
| Request models (ObligorRow, LoadingRow, GroupRow, CreditVarRequest) | `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`, `protected_namespaces=()`, `strict=True` (O-8) |
| Result models (section 3) | `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`, `protected_namespaces=()` |

`frozen=True` blocks attribute assignment only (F-3). Sequence fields are tuples
for that reason. Dict fields stay dicts; C.15 covers their mutation.

Coercion of request input (O-8, C.18, F-34):

| Field type | Accepts | Rejects (raises R) |
|---|---|---|
| `int` | int | bool, float, str, numpy integer |
| `float` | float, int (stored as float) | bool, str |
| `bool` | bool | int, str, numpy bool |
| `str` | str | every other type |
| sequence (`obligors`, `loadings`, `groups`, `alphas`) | list, tuple (stored as tuple) | every other type, including set, generator and str |

The rows apply to top-level fields, to dict keys, to dict values and to sequence
elements.
`None` stays valid for each optional field.

## 2. Request

### 2.1 ObligorRow

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| obligor_uid | `int` | -2**63 <= v <= 2**63 - 1 | integer | int64 range added (C.17) |
| ead | `float` | finite, >= 0 | number | finite added |
| pd | `float` | 0 <= v <= 1 | number | none |
| lgd | `float` | 0 <= v <= 1 | number | none |
| in_stress_test | `bool` | default False | boolean | none |
| force_default | `bool` | default False | boolean | none |
| group | `str \| None` | min_length 1; default None; contains no NUL character ("\x00") | string or null | none |
| rho | `float \| None` | 0 <= v < 1; default None | number or null | none |

### 2.2 LoadingRow

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| obligor_uid | `int` | int64 range | integer | int64 range added |
| factor | `str` | min_length 1; contains no NUL character ("\x00") | string | collision rule (C.17) |
| loading | `float` | -1 <= v <= 1 | number | none |

### 2.3 GroupRow

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| name | `str` | min_length 1; contains no NUL character ("\x00") | string | none |
| params | `dict[str, float]` | keys and ranges per catalog | object | none |

### 2.4 CreditVarRequest

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| model | `str` | a MODEL_CATALOG key | string | none |
| obligors | `tuple[ObligorRow, ...]` | length >= 1; unique obligor_uid | array | list to tuple |
| params | `dict[str, float]` | keys and ranges per catalog | object | none |
| loadings | `tuple[LoadingRow, ...] \| None` | no duplicate (obligor_uid, factor); per obligor sum of loading^2 <= 1 + 1e-9 | array or null | list to tuple; 2 rules added |
| groups | `tuple[GroupRow, ...] \| None` | as today | array or null | list to tuple |
| n_scenarios | `int` | > 0; no upper limit (O-12) | integer | none |
| alphas | `tuple[float, ...]` | non-empty; each in (0, 1); pairwise distinct under `==` | array | list to tuple; distinct added |
| seed | `int` | >= 0 | integer | lower bound added |

The cross-field validators of c060f06 stay: model and params, unique uids,
loadings coverage, groups, Marshall-Olkin PD >= systemic_pd, per-obligor rho.
Section 2.5 gives the loc of each rule. The request keeps the caller's alpha
order. The result sorts (C.3).

Request JSON example:

```json
{"model": "sf_gaussian",
 "obligors": [{"obligor_uid": 10, "ead": 1000000.0, "pd": 0.02, "lgd": 0.45,
               "in_stress_test": false, "force_default": false,
               "group": null, "rho": null}],
 "params": {"rho": 0.15}, "loadings": null, "groups": null,
 "n_scenarios": 250000, "alphas": [0.99, 0.999], "seed": 12345}
```

### 2.5 Request error locations (C.20, S2.T10)

R is `pydantic.ValidationError`. Each rule below gives one `errors()` entry with
the stated loc. i, j and g are 0-based row indices of `obligors`, `loadings` and
`groups` in request order. k is a param key. "First" means the lowest index that
has the fault. "Second occurrence" means the first index whose value occurs at a
lower index.

| # | Rule | loc |
|---|---|---|
| 1 | A field constraint of sections 2.1 to 2.4: type, range, finite, min_length, int64 range, seed >= 0, n_scenarios > 0 | the field path, for example ("obligors", i, "ead"), ("seed",) |
| 2 | An alphas element outside (0, 1) | ("alphas", n), n the element index |
| 3 | alphas empty | ("alphas",) |
| 4 | Duplicate alphas under `==` | ("alphas", n), second occurrence |
| 5 | model is not a MODEL_CATALOG key | ("model",) |
| 6 | params key unknown to the model | ("params", k) |
| 7 | params value out of the catalog range | ("params", k) |
| 8 | Duplicate obligor_uid | ("obligors", i, "obligor_uid"), second occurrence |
| 9 | loadings present for a model without loadings | ("loadings",) |
| 10 | loadings absent or empty for a model that requires them | ("loadings",) |
| 11 | A loadings obligor_uid not in obligors | ("loadings", j, "obligor_uid"), first |
| 12 | An obligor without a loadings row | ("obligors", i, "obligor_uid"), first |
| 13 | Duplicate (obligor_uid, factor) | ("loadings", j, "factor"), second occurrence |
| 14 | Per-obligor sum of loading^2 > 1 + 1e-9 | ("loadings", j, "loading"), j the first row at which the running sum over the rows of that obligor, in request order, exceeds 1 + 1e-9 |
| 15 | factor in the reserved set (below) | ("loadings", j, "factor"), first |
| 16 | groups present for a model without groups | ("groups",) |
| 17 | An obligor group set for a model without groups | ("obligors", i, "group"), first |
| 18 | Request params present for a grouped model | ("params",) |
| 19 | groups absent or empty for a grouped model | ("groups",) |
| 20 | An obligor without a group, grouped model | ("obligors", i, "group"), first |
| 21 | Duplicate group name | ("groups", g, "name"), second occurrence |
| 22 | An obligor group not defined in groups | ("obligors", i, "group"), first |
| 23 | A group param key unknown, missing or out of range | ("groups", g, "params", k), first g, then first k in catalog order |
| 24 | marshall_olkin: obligor pd < systemic_pd | ("obligors", i, "pd"), first |
| 25 | obligor rho set for a model without per-obligor rho | ("obligors", i, "rho"), first |

Reserved set (rule 15, C.17, F-36): for mf_gaussian and mf_students_t, the name
`IDIO_{uid:03d}` for each request obligor_uid, in the format of
`recipes._mf_universe_and_bundle`; for mf_students_t, also `CHI_GLOBAL`.

## 3. Result

### 3.1 CreditVarResult

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| result_schema_version | `Literal["1.0"]` | exactly "1.0" | string | new (C.2) |
| model | `str` | request model | string | none |
| headline_alpha | `float` | max of request alphas | number | none |
| summary | `RiskSummary` | section 3.2 | object | changed |
| per_model_tail | `dict[str, ModelTail]` | keys are JtdModel names (C.7) | object | key vocabulary |
| loss_percentiles | `tuple[PercentileLoss, ...]` | 16 records, strictly ascending percentile | array | dict to records |
| obligor_contributions | `tuple[ObligorContribution, ...]` | one per request obligor, strictly ascending obligor_uid | array | tuple; order defined |
| loss_histogram | `LossHistogram` | section 3.8 | object | tuples |
| stress_test | `StressTestResult \| None` | None when no obligor has in_stress_test | object or null | none |
| provenance | `RunProvenance` | section 3.10 | object | changed |

### 3.2 RiskSummary

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| loss_mean | `float` | finite | number | none |
| loss_std_dev | `float` | finite, >= 0; 0.0 when loss_min == loss_max (ZV) | number | ZV rule added; validator added in S1 |
| loss_skewness | `float` | finite; 0.0 when loss_min == loss_max (ZV) | number | ZV rule added; validator added in S1 |
| loss_excess_kurtosis | `float` | finite; 0.0 when loss_min == loss_max (ZV) | number | ZV rule added; validator added in S1 |
| loss_probability | `float` | 0 <= v <= 1 | number | validator added in S1 |
| loss_min | `float` | finite | number | none |
| loss_max | `float` | finite | number | none |
| coefficient_of_variation | `float` | finite | number | none |
| by_alpha | `tuple[AlphaRisk, ...]` | one per request alpha, strictly ascending | array | new; replaces var, es, economic_capital_var, economic_capital_es; validator added in S1 |
| expected_loss_by_model | `dict[str, float]` | keys are JtdModel names (C.7) | object | key vocabulary; validator added in S1 |

Methods: `at(alpha: float) -> AlphaRisk` (new, C.4). It raises KeyError for a
level that the run did not compute (O-9). Removed: `max_alpha`, `get_var`,
`get_es`, `get_es_to_var_ratio`, `es_to_var_ratio`. No ratio member returns
(O-5).

### 3.3 AlphaRisk (new)

| Field | Python type | Constraint | JSON |
|---|---|---|---|
| alpha | `float` | bit-equal to a request alpha | number |
| var | `float` | finite | number |
| es | `float` | finite | number |
| economic_capital_var | `float` | var - expected loss, formula of c060f06 | number |
| economic_capital_es | `float` | es - expected loss, formula of c060f06 | number |

### 3.4 PercentileLoss (new)

| Field | Python type | Constraint | JSON |
|---|---|---|---|
| percentile | `float` | percent units, the 16 values of `run._PERCENTILE_GRID` | number |
| loss | `float` | finite; `np.percentile` value of c060f06 | number |

`percentile` keeps percent units so that the values stay bitwise equal to
c060f06. `alpha` is a probability. The two names mark the two units.

### 3.5 ModelTail

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| var | `float` | finite | number | none |
| es | `float` | finite | number | none |

The `es_to_var_ratio` property is removed (O-5).

### 3.6 ObligorContribution

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| obligor_uid | `int` | a request uid | integer | none |
| ead | `float` | equals the request ead; never NaN | number | NaN default removed (C.6) |
| expected_loss | `float` | finite | number | none |
| es_contribution | `float` | finite | number | none |
| es_pct | `float` | finite | number | none |
| incremental_var | `float` | finite | number | none |
| incremental_es | `float` | finite | number | none |
| ul_contribution | `float` | finite | number | none |
| ul_pct | `float` | finite | number | none |

### 3.7 StressTestObligor and StressTestResult

| Model | Field | Python type | Change |
|---|---|---|---|
| StressTestObligor | obligor_uid, ead, lgd, loss | `int`, `float`, `float`, `float` | none |
| StressTestResult | n_obligors | `int` | none |
| StressTestResult | total_loss | `float` | none |
| StressTestResult | obligors | `tuple[StressTestObligor, ...]`, strictly ascending obligor_uid | tuple; order defined |

### 3.8 LossHistogram

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| bin_edges | `tuple[float, ...]` | length nbins + 1, strictly ascending | array | list to tuple; validator added in S1 |
| counts | `tuple[int, ...]` | length nbins; sum = n_scenarios | array | list to tuple; validator added in S1 |

### 3.9 Removed from the result surface

`RiskSummary.var`, `es`, `economic_capital_var`, `economic_capital_es`
(replaced by `by_alpha`); the dict form of `loss_percentiles`; the
"forced_default" overlay key (merged into "FORCED_DEFAULT"); the ES/VaR ratio
members of RiskSummary and ModelTail (O-5).

### 3.10 RunProvenance

| Field | Python type | Constraint | JSON | Change |
|---|---|---|---|---|
| engine_version | `str` | `^[^+]+\+src\.[0-9a-f]{64}$` (P-001 clause 5) | string | new format |
| scenario_construction_id | `Literal["MixedCreditFactorGenerator/D-C15"]` | constant | string | new |
| numpy_version | `str` | `numpy.__version__` | string | new |
| scipy_version | `str` | `scipy.__version__` | string | new |
| numba_version | `str` | `numba.__version__` | string | new |
| model | `str` | request model | string | none |
| params | `dict[str, float]` | request params as given | object | none |
| n_obligors | `int` | number of request obligors | integer | none |
| n_scenarios | `int` | request n_scenarios | integer | none |
| seed | `int` | request seed | integer | none |
| alphas | `tuple[float, ...]` | request alphas, request order | array | list to tuple |
| input_hash | `str` | `^[0-9a-f]{64}$` (section 4.2) | string | 16 to 64 hex; canonical form |
| generated_at | `str` | ISO 8601 UTC | string | none; outside reproducibility equality |

Result JSON example (abbreviated):

```json
{"result_schema_version": "1.0", "model": "sf_gaussian", "headline_alpha": 0.999,
 "summary": {"loss_mean": 9012.5, "...": "...",
   "by_alpha": [{"alpha": 0.99, "var": 45000.0, "es": 61234.1,
                 "economic_capital_var": 35987.5, "economic_capital_es": 52221.6},
                {"alpha": 0.999, "var": 90000.0, "es": 104321.9,
                 "economic_capital_var": 80987.5, "economic_capital_es": 95309.4}],
   "expected_loss_by_model": {"COPULA_SF_GAUSSIAN": 9012.5}},
 "per_model_tail": {"COPULA_SF_GAUSSIAN": {"var": 90000.0, "es": 104321.9}},
 "loss_percentiles": [{"percentile": 0.1, "loss": 0.0}, "..."],
 "obligor_contributions": [{"obligor_uid": 10, "ead": 1000000.0, "...": "..."}],
 "loss_histogram": {"bin_edges": [0.0, "..."], "counts": [240000, "..."]},
 "stress_test": null,
 "provenance": {"engine_version": "0.2.0+src.<64 hex>",
   "scenario_construction_id": "MixedCreditFactorGenerator/D-C15",
   "numpy_version": "2.4.4", "scipy_version": "<x.y.z>", "numba_version": "<x.y.z>",
   "model": "sf_gaussian", "params": {"rho": 0.15}, "n_obligors": 1,
   "n_scenarios": 250000, "seed": 12345, "alphas": [0.99, 0.999],
   "input_hash": "<64 hex>", "generated_at": "2026-10-29T12:00:00+00:00"}}
```

The numbers in the example are illustrations. They are not test values.

## 4. Algorithms

### 4.1 engine_version

P-CREDIT_VAR_API-001 clause 5. Compute once per process. The cached value is
immutable after the first computation. Exclude `__pycache__`, `.nbi`, `.nbc`
and every non-`.py` file. Exclude `riskquant_lab`. S6 measures the first
computation.

### 4.2 input_hash

1. Take `model_dump(mode="python")` of the validated request.
2. Replace each float with `{"__f64__": value.hex()}`; replace each tuple with a
   list; keep each bool as a bool.
3. Serialize with `json.dumps(sort_keys=True, separators=(",", ":"),
   ensure_ascii=True)` and encode as UTF-8.
4. Return the lowercase SHA-256 hex digest.

This is the D040 canonical form of `run_identity.schemas._canonicalize` (F-27).
`float.hex()` keeps the sign of zero, so -0.0 and 0.0 give different hashes.
C.10 therefore states equality as field-wise bit equality.

### 4.3 Reproducibility equality (C.13)

Remove `provenance.generated_at` from both results. Compare the
`model_dump_json()` bytes. The comparison is valid only when engine_version,
scenario_construction_id, numpy_version, scipy_version and numba_version are
equal (D-C15, environment-relative).

### 4.4 Construction guard (C.12)

`SCENARIO_CONSTRUCTION_ID` and the recorded SHA-256 of the LF-normalized text
of `src/riskquant/credit/factors/simulate/mixed.py` live side by side in the
solution tier. A change of `mixed.py` fails S1.T13. The owner then keeps the
identifier (and records the new digest) or sets a new identifier. The guard
reads the core file. It does not change it.
