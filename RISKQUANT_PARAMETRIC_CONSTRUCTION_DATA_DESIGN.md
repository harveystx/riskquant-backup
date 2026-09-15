# Parametric construction-data architecture — design decision (proposal)

Status: **PROPOSAL. No production edits made.** D031–D033 seam unchanged;
Parametric unwired. Awaiting explicit approval before implementation.

---

## 1. Parametric construction / data-flow map

```
market data (covariance estimation, NOT yet built in-repo)
   │   available_factor_ids   cholesky_matrix        volatilities
   │        (K,) int32          (K,K) float64          (K,) float64
   ▼
ParametricGenerator.__init__(available_factor_ids, cholesky_matrix,
                             volatilities, total_scenarios,
                             distribution_kind, seed, <dist params>)
   │  __post_init__:
   │    - coerce factor_ids→int32(1D), cholesky→float64(2D square finite),
   │      vols→float64(1D positive finite); dims all == K
   │    - persist cholesky C-contiguous; derive _cholesky_transpose (C-contig)
   │    - require_positive_int(total_scenarios); seed int|None
   │    - per-kind param validation (see §3)
   ▼
iter_batches: rng → normals (K_total) → matmul(cholesky_T sliced) →
              distribution aux (D029) → vols scale → shocks (B,K_req) f32 F
   ▼
ScenarioSetBatch → runner (unchanged ScenarioGenerator protocol)
```

Current callers: **none in production or demo** — only two test fixtures
(`test_batch_size_reproducibility.py`, `test_boundary_contracts.py`). There is
**no in-repo producer** of `cholesky_matrix`/`volatilities` yet; the tests build
them synthetically (`np.eye`, literal arrays). Parametric is fully implemented
but entirely unwired.

## 2. Exact parameter matrix

| Input | Kind | dtype / shape / layout | Validation (in generator `__post_init__`) | Origin |
|---|---|---|---|---|
| `available_factor_ids` | **data** | int32, `(K,)`, 1D | 1D; length == K | market data (factor universe) |
| `cholesky_matrix` | **data** | float64, `(K,K)`, persisted C-contig | 2D, square, finite; transpose derived | market data (covariance → Cholesky) |
| `volatilities` | **data** | float64, `(K,)`, 1D | 1D; length == K; finite; strictly > 0 | market data (per-factor vol) |
| `total_scenarios` | **config** | int | `require_positive_int` (strict > 0) | user config |
| `seed` | **config** | int \| None | int or None | user config |
| `distribution_kind` | **config** | str Literal | selector (see §3) | user config |
| `student_t_df` | **config (dist)** | float \| None | required iff student_t; finite, > 2 | user config |
| `vg_nu` | **config (dist)** | float \| None | required iff variance_gamma; finite, > 0 | user config |
| `prob_crisis` | **config (dist)** | float \| None | required iff regime; finite, ∈ [0,1] | user config |
| `crisis_vol_multipliers` | **config (dist), K-coupled** | float \| `(K,)` float64 \| None | required iff regime; scalar>0 **or** per-factor length K, positive finite | user config (per-factor form couples to data K) |

The three **data** inputs are dimensionally coupled by K and are validated
together — they form one coherent object. The **config** inputs are independent
of K (except the per-factor `crisis_vol_multipliers` form, which is checked
against K at construction).

## 3. Distribution-configuration matrix

| `distribution_kind` | Required params | Optional | Aux RNG draw (D029) |
|---|---|---|---|
| `standard_normal` | — | — | none |
| `laplace` | — | — | exponential(K) |
| `student_t` | `student_t_df` (>2) | — | chisquare(K) |
| `variance_gamma` | `vg_nu` (>0) | — | gamma(K) |
| `regime_switching_gmm` | `prob_crisis` ∈[0,1], `crisis_vol_multipliers` (scalar or per-factor>0) | — | uniform(K) |

The generator today represents these with **nullable fields**
(`student_t_df=None`, …) plus runtime dispatch on `distribution_kind`. That makes
invalid combinations *representable* (e.g. `standard_normal` with `vg_nu` set, or
`student_t` with `vg_nu` and no `df`); the generator only catches them at
`__post_init__`. At the **configuration** layer these are naturally **separate
typed variants** — a nested discriminated union — so invalid combinations become
*unrepresentable*.

**Answer to archaeology Q8:** `ParametricGenerator` is one generator *class* with
internal variants, but its distribution parameters are best modelled as
**separate config variants** (nested discriminated union on distribution kind),
not one nullable blob.

## 4. Candidate architecture comparison

The configuration and the construction-data are **two separate concerns**;
each needs its own decision.

### 4a. Construction-data representation

| Option | Shape | Verdict |
|---|---|---|
| **A. Generic build with nullable data args** `build(config, *, matrix=None, cholesky=None, vols=None, factor_ids=None)` | one entry, nullable | **Rejected** — explicitly forbidden data blob; invalid combos representable (parametric config + matrix; matrix config + cholesky); no fail-fast |
| **B. Typed per-model data object + typed data union** new `ParametricRiskData` frozen dataclass; `build(config, *, data: RiskFactorMatrix \| ParametricRiskData)` | one entry, typed union | **Recommended** — mirrors existing `RiskFactorMatrix` / credit `AssembledInputs` frozen-dataclass pattern; invalid combos caught by (config, data) dispatch; no nulls |
| **C. `ScenarioData` protocol** `build(config, *, data: ScenarioData)` | protocol | **Rejected** — `RiskFactorMatrix` (T×N time series) and the parametric bundle (K factors + K×K Cholesky + K vols) share **no behavioural interface**; a protocol adds a layer with nothing common to abstract. A type union + `isinstance` dispatch is strictly simpler and sufficient. **Protocol not required — proven by absence of any shared surface.** |
| **D. Separate build functions** `build_matrix_generator(...)`, `build_parametric_generator(...)` | fragmented | **Rejected** — breaks the single seam entry point; callers must know which builder maps to which model |

### 4b. Parametric distribution configuration

| Option | Verdict |
|---|---|
| **Flat nullable config** (`student_t_df=None`, `vg_nu=None`, …) | **Rejected** — the exact anti-pattern the seam avoids; invalid combos representable |
| **Nested discriminated union on distribution kind** (`StandardNormalDistribution`, `LaplaceDistribution`, `StudentTDistribution{df}`, `VarianceGammaDistribution{nu}`, `RegimeSwitchingDistribution{prob_crisis, crisis_vol_multipliers}`) discriminated by a `kind` tag | **Recommended** — each variant carries only its params; invalid combos unrepresentable; mirrors the outer model union |

## 5. Recommended minimal design

Two small, orthogonal additions — configuration union and data representation
kept separate.

**(i) Construction data — new CORE-tier frozen dataclass** (mirrors
`RiskFactorMatrix`; Pydantic stays out of numeric core):

```
# market_data/risk_factors/parametric_data.py  (proposed)
@dataclass(frozen=True, slots=True)
class ParametricRiskData:
    factor_ids: np.ndarray        # (K,) int32
    cholesky_matrix: np.ndarray   # (K,K) float64, square, finite
    volatilities: np.ndarray      # (K,) float64, positive, finite
    # __post_init__: the SAME dimensional/finiteness/positivity checks the
    # generator does today — relocated here as the single authority for this
    # data bundle (no duplication; see §7).
```

**(ii) Configuration — new solution-tier Pydantic config + nested distribution
union**, added as the fourth member of the existing `ScenarioModelConfig`:

```
# solutions/scenario_models/schemas.py  (proposed additions)
class StandardNormalDistribution(BaseModel):   kind: Literal["standard_normal"]
class LaplaceDistribution(BaseModel):          kind: Literal["laplace"]
class StudentTDistribution(BaseModel):         kind: Literal["student_t"]; df: float (>2, finite)
class VarianceGammaDistribution(BaseModel):    kind: Literal["variance_gamma"]; nu: float (>0, finite)
class RegimeSwitchingDistribution(BaseModel):  kind: Literal["regime_switching_gmm"];
                                               prob_crisis: float [0,1];
                                               crisis_vol_multipliers: float | list[float] (>0)
Distribution = Annotated[<those five>, Field(discriminator="kind")]

class ParametricGeneratorConfig(BaseModel):
    model: Literal["parametric"]
    total_scenarios: StrictInt (gt=0)
    seed: StrictInt | None = None
    distribution: Distribution
ScenarioModelConfig = Annotated[Historical | Sbb | FilteredHistorical | Parametric,
                                Field(discriminator="model")]
```

**(iii) Build API — generalise the data argument to a typed union (not
nullable):**

```
build_scenario_generator(config, *, data: RiskFactorMatrix | ParametricRiskData)
  - Historical/SBB/FHS config → require data is RiskFactorMatrix
  - Parametric config         → require data is ParametricRiskData
  - mismatch → TypeError at the solution boundary (fail fast)
```

Meets every decision criterion: protocol preserved; config separate from data;
invalid (config, data) pairs rejected; Pydantic out of core; no strings/objects
in kernels; validation single-authored per tier; scales by one config + one
`isinstance` branch per future model; smallest new abstraction (one frozen
dataclass, reusing an established pattern).

## 6. Why the rejected designs are worse

- **Nullable data blob (A):** permits `matrix=None`/`cholesky=None`; a parametric
  config paired with a matrix (or vice versa) type-checks and only fails deep
  inside construction. Forbidden by the brief and by "make invalid combinations
  unrepresentable".
- **`ScenarioData` protocol (C):** the two data types have nothing in common to
  abstract; the protocol is ceremony with no shared method/attribute, and still
  needs per-type handling in `build`. A union does the same job with less.
- **Separate build functions (D):** fragments the one entry point that D031–D033
  deliberately established; the caller must re-derive the config→builder mapping
  the union already encodes.
- **Flat nullable Parametric config:** re-introduces exactly the nullable-blob
  anti-pattern the seam exists to prevent, at the configuration layer where it is
  most avoidable.

## 7. Migration impact on the existing Historical/SBB/FHS seam

- **Config union:** Parametric is a purely additive 4th member. The three
  existing configs are untouched.
- **Build signature:** rename the data keyword `matrix` → `data`, typed
  `RiskFactorMatrix | ParametricRiskData`. The three matrix models still receive
  a `RiskFactorMatrix` (dispatch requires it). Call-site churn is mechanical —
  `matrix=` → `data=` in the demo and the three seam-test files — with **no
  behavioural change** (the matrix models build identically, so D019 goldens and
  all equivalence tests stay green).
- **Generators / kernels / protocol / D029 / shift / layout / P3:** unchanged.
- **Duplicate validation:** the Parametric data-shape checks move *out of* the
  generator *into* `ParametricRiskData.__post_init__` when the generator is
  refactored to consume the bundle — a relocation, not duplication. If the
  generator refactor is deferred, the build passes the bundle's already-validated
  arrays to the current constructor, which re-validates (harmless
  belt-and-suspenders across tiers, as with StrictInt vs `require_positive_int`
  in D031–D033). Either way, no new validation logic is written twice.
- **New files:** `market_data/risk_factors/parametric_data.py` (data) and
  additions to `solutions/scenario_models/schemas.py` + `api.py` (config +
  dispatch).

## Recommendation

Adopt **§5**: a typed `ParametricRiskData` frozen dataclass (reusing the
`RiskFactorMatrix`/`AssembledInputs` pattern), a `ParametricGeneratorConfig` with
a nested distribution discriminated union, and a build data argument generalised
to the typed union `RiskFactorMatrix | ParametricRiskData`. No `ScenarioData`
protocol. Implement only on explicit approval, in two steps (data object +
config/dispatch), each independently validated.
