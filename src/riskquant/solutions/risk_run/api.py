# src/riskquant/solutions/risk_run/api.py
"""
The production scenario-VaR entry point.

    typed inputs -> composition contract -> run identity -> [verify environment
    / inputs against a prior manifest] -> P1-A gate -> execute
    -> RiskRun (result + identity)

D081 §11 composition. `MethodologyDeclaration` partitions the explicit
production universe U into FULL_REVAL, TAYLOR/SUPPLIED and TAYLOR/DERIVED. Every
methodology reads ONE canonical shocked-level buffer and writes a disjoint block
of ONE `PnlGrid` ([U_F | U_S | U_D], float32). The grid is scattered onto the
canonical economic axis sorted(U) and passes through ONE D081 conversion:

    dW = dV * q_s + V_0 * (q_s - q_0)

FULL_REVAL enters with dV = V_s - V_0, both from the same valuation route
(V_0 at S = 1), so dW = V_s * q_s - V_0 * q_0 exactly; there is no second
reporting equation. TAYLOR/SUPPLIED is refused while owner policy P-1 is open.

P1-B: the run owns its identity. One `ValuationContext` supplies the static
configuration, the compiled representation, the mapper (as-of, D067) and the
discounting mode (D080). The run builds the universe map, values x_0 ONCE for
every position of U (so every V_0, FULL and TAYLOR/DERIVED, is run-computed),
derives the full Taylor base binding itself and requires each certified book to
carry exactly that binding. No caller-supplied V_0 or binding exists.

Scenario levels come from the Proxy V1 operator (D088): the run compiles the
authored proxy terms over its full axis (valuation prefix + reporting-FX suffix,
D088 §13.8(a)), lowers the compiled model onto that axis, binds its (d, tau)
terms to the world's (d, tau, h) columns before generation, and applies
X_A = sum_j beta_Aj * Y_(d_j, tau_j, h), A_s = T^-1_kappa_A(A_0, X_A).

Composes ONLY existing production functions (the D036 scenario seam, the D088
compiler, lowering, world binding and linear-term application, market-data
assembly, `run_valuation`, the
certified `SensitivityPnlCalculator`, `PnlGrid`, the FX conversion and
`compute_var_es`). Its own arithmetic is limited to summing valuation rows into
their parent economic UID (D055). The manifest is built from the very objects
handed to the computation, before any numerical work; it never enters a
generator, a kernel or a Numba signature (D001).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.schema_objects import AuthoredProxyTerm
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import INVALID_UIX, NO_FX_RF_REQUIRED
from riskquant.market_data.assemble import (
    UniverseMapping,
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.instruments import (
    INSTRUMENT_TYPE_TO_FAMILY_ARRAY,
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
    InstrumentTypeFamily,
)
from riskquant.reporting.fx_conversion import (
    FxConversionPlan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics._validate import _validate_confidence_level
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    NonReproducibleRunError,
    ReportingSpec,
    ReproducibilityMode,
    RiskRun,
    RunFactorAxis,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.run_identity.api import (
    InputMismatchError,
    build_run_manifest,
    verify_environment,
    verify_run_inputs,
)
from riskquant.solutions.run_identity.hashing import (
    digest_named_arrays,
    digest_reporting_plan,
    hash_compiled_book,
    hash_compiled_portfolio,
    hash_universe_mapping,
)
from riskquant.solutions.run_identity.schemas import (
    CompositionIdentity,
    MethodologyIdentity,
    RunContextIdentity,
    RunManifest,
)
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    resolve_parametric_binding_horizon,
    resolve_parametric_crisis_multipliers,
)
from riskquant.solutions.scenario_models.schemas import (
    ParametricGeneratorConfig,
    ScenarioModelConfig,
)
from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world
from riskquant.solutions.sensitivity_pnl.api import FactorAxis
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorContractError,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
    verify_base_binding,
)
from riskquant.universe import UixMapper
from riskquant.utils.business_days import ord_to_date
from riskquant.valuation.engine import run_valuation

__all__ = ["derive_run_factor_axis", "run_scenario_var_es"]


_NO_SEED = object()

# PnlGrid block order. Fixed, so a UID's grid row depends only on the partition.
_FULL_BLOCK, _SUPPLIED_BLOCK, _DERIVED_BLOCK = 0, 1, 2

# Digest of the EMPTY named bundle: the Taylor representation of a run that
# prices no Taylor block. A defined content identity, not a placeholder.
_NO_TAYLOR_REPRESENTATION_DIGEST = digest_named_arrays()


def _is_reproducible(config: ScenarioModelConfig) -> bool:
    """Deterministic models carry no `seed` field and are always reproducible;
    stochastic ones are reproducible only with an explicit seed (D039/D041)."""
    seed = getattr(config, "seed", _NO_SEED)
    return seed is _NO_SEED or seed is not None


# =============================================================================
# Composition contract (fail closed, before identity and numerics)
# =============================================================================
def _require_block_wiring(
    *,
    methodology: MethodologyDeclaration,
    taylor_supplied: TaylorBlock | None,
    taylor_derived: TaylorBlock | None,
) -> None:
    """A Taylor block is supplied iff its partition is non-empty."""
    _, u_supplied, u_derived = methodology.blocks()
    if taylor_supplied is not None or u_supplied.size:
        raise CompositionContractError(
            f"TAYLOR/SUPPLIED is not admissible: owner policy P-1 is unresolved "
            f"({u_supplied.size} position(s) declared SUPPLIED). A supplied "
            "representation is never priced as DERIVED or FULL_REVAL instead."
        )
    if u_derived.size and taylor_derived is None:
        raise CompositionContractError(
            f"{u_derived.size} position(s) are declared TAYLOR/DERIVED but no "
            "TAYLOR/DERIVED block was supplied."
        )
    if taylor_derived is not None and not u_derived.size:
        raise CompositionContractError(
            "a TAYLOR/DERIVED block was supplied but no position is declared "
            "TAYLOR/DERIVED; an unused methodology input is a wiring defect."
        )


def _require_taylor_capable_spot(
    *, context: ValuationContext, taylor_uids: npt.NDArray[np.int64]
) -> None:
    """
    D084: an inverted-FX SPOT leg (effective FX `invert = 1`) is FULL_REVAL only.

    Its value `q * S / X` is not bilinear in the quoted factor X, and no
    admitted Taylor representation of it exists (the SPOT sensitizer refuses
    it, D010). The P1-A gate binds a Taylor book's base, order and
    completeness, not its Greek values, so a caller-built book for such a leg
    would still be certified. The route is therefore refused here, from the
    run's own compiled identity graph, before identity and numerics. FULL_REVAL
    positions are not inspected: an inverted FULL_REVAL leg coexists with
    direct TAYLOR/DERIVED legs.
    """
    block = context.portfolio.spot
    if block is None or taylor_uids.size == 0:
        return
    mapper = context.uix_mapper
    fx_eff_uix = mapper.spot_effective_pairs[block.effective_spot_price_uix, 1]
    has_fx = fx_eff_uix != INVALID_UIX
    inverted = np.zeros(fx_eff_uix.shape[0], dtype=np.bool_)
    inverted[has_fx] = mapper.fx_effective_pairs[fx_eff_uix[has_fx], 1] == 1
    refused = np.intersect1d(
        np.asarray(block.trade_uids, dtype=np.int64)[inverted], taylor_uids
    )
    if refused.size:
        raise CompositionContractError(
            f"{refused.size} position(s) declared TAYLOR are inverted-FX SPOT legs "
            f"(effective FX invert = 1; first 10: {refused[:10].tolist()}). An "
            "inverted-FX SPOT leg is admitted as FULL_REVAL only (D084)."
        )


#: D102.2: the instrument ids of the FORWARD family. The registry array is
#: indexed by instrument id, so a position in it IS the id.
_FORWARD_FAMILY_INSTRUMENT_IDS: npt.NDArray[np.int32] = np.flatnonzero(
    np.asarray(INSTRUMENT_TYPE_TO_FAMILY_ARRAY) == int(InstrumentTypeFamily.FORWARD)
).astype(np.int32)


def _require_forward_capability(
    *, context: ValuationContext, methodology: MethodologyDeclaration
) -> None:
    """
    D102 and D103: source-specific admission of the UIDs that own a
    `vanilla_forward` row.

    Block type and rf namespace do not establish the instrument source:
    `BlockType.VANILLA_FORWARD` is a dispatch key, and `GlobalIdNamespace.NODES`
    holds node ids and curve-point rf ids alike. The source is read from the
    ingest rows bound to the UID (D102.2). A UID may own several ingest rows
    (D055), so every row of the UID is inspected; no single-row assumption is
    made.

    Checks, in this order:

    D102.3     every FORWARD-family row of a forward UID is VANILLA_FORWARD_NODE
               (every methodology partition);
    D103.3     no used FX leg of a forward row of a TAYLOR/DERIVED UID is
               inverted (`fx_effective_pairs[:, 1] == 1`, asset and strike leg);
    D103.4(b)  a TAYLOR/DERIVED forward UID requires DiscountingMode.NONE;
    D103.4(a)  every non-forward row of a TAYLOR/DERIVED forward UID is SPOT
               (its D084 admissibility is proven earlier by
               `_require_taylor_capable_spot`);
    D102.4     a forward UID in TAYLOR/SUPPLIED is refused (D103.5; P-1 refuses
               the partition earlier in the run).

    D102.3 precedes D103.3, which precedes D103.4 (revision 8). (b) before (a)
    is a deterministic implementation precedence, not a contract ordering.
    FULL_REVAL forward UIDs are subject to D102.3 only.
    """
    block = context.portfolio.vanilla_forward
    if block is None:
        return
    trade_uids = np.asarray(block.trade_uids, dtype=np.int64)
    forward_uids = np.unique(trade_uids)
    batch = context.ingest_batch
    row_uids = np.asarray(batch.uids, dtype=np.int64)
    instruments = np.asarray(batch.instrument_ids, dtype=np.int32)
    of_forward_uid = np.isin(row_uids, forward_uids)
    forward_family = np.isin(instruments, _FORWARD_FAMILY_INSTRUMENT_IDS)
    foreign = of_forward_uid & forward_family & (instruments != VANILLA_FORWARD_NODE_ID)
    if bool(foreign.any()):
        bad = np.unique(row_uids[foreign])
        raise CompositionContractError(
            f"D102.3: {bad.size} UID(s) own a vanilla_forward row but have a "
            "FORWARD-family ingest row whose instrument is not "
            f"VANILLA_FORWARD_NODE (first 10: {bad[:10].tolist()}; instrument "
            f"id(s) {np.unique(instruments[foreign]).tolist()}). The forward "
            "source of the UID is not established; the run fails closed."
        )
    _, u_supplied, u_derived = methodology.blocks()
    derived_rows = np.isin(trade_uids, u_derived)
    if bool(derived_rows.any()):
        mapper = context.uix_mapper
        inverted = np.zeros(trade_uids.shape[0], dtype=np.bool_)
        for fx in (
            mapper.efp_effective_quads[block.effective_forward_price_uix, 3],
            mapper.edf_effective_triples[block.effective_discount_factor_uix, 2],
        ):
            has = fx != INVALID_UIX
            inverted[has] |= mapper.fx_effective_pairs[fx[has], 1] == 1
        refused_rows = inverted & derived_rows
        if bool(refused_rows.any()):
            bad = np.unique(trade_uids[refused_rows])
            raise CompositionContractError(
                f"D103.3: {bad.size} position(s) declared TAYLOR/DERIVED own a "
                "VANILLA_FORWARD row with an inverted FX leg (first 10: "
                f"{bad[:10].tolist()}). An inverted forward FX leg is admitted as "
                "FULL_REVAL only."
            )
        derived_forward = np.intersect1d(forward_uids, u_derived)
        if context.discounting_mode is not DiscountingMode.NONE:
            raise CompositionContractError(
                f"D103.4: {derived_forward.size} position(s) declared "
                "TAYLOR/DERIVED own a VANILLA_FORWARD row, and TAYLOR/DERIVED "
                "forward positions are admitted in DiscountingMode.NONE only "
                f"(got {context.discounting_mode.name}; first 10: "
                f"{derived_forward[:10].tolist()})."
            )
        other = np.isin(row_uids, derived_forward) & ~forward_family
        not_spot = other & (instruments != SPOT_ID)
        if bool(not_spot.any()):
            bad = np.unique(row_uids[not_spot])
            raise CompositionContractError(
                f"D103.4: {bad.size} position(s) declared TAYLOR/DERIVED own a "
                "VANILLA_FORWARD row and a non-forward row that is not SPOT "
                f"(first 10: {bad[:10].tolist()}; instrument id(s) "
                f"{np.unique(instruments[not_spot]).tolist()})."
            )
    refused = np.intersect1d(forward_uids, u_supplied)
    if refused.size:
        raise CompositionContractError(
            f"D102.4: {refused.size} position(s) declared TAYLOR/SUPPLIED own a "
            f"vanilla_forward row (first 10: {refused[:10].tolist()}). "
            "TAYLOR/SUPPLIED is not admitted for VANILLA_FORWARD_NODE (D103.5); "
            "the admitted routes are FULL_REVAL and TAYLOR/DERIVED under "
            "D103.3/D103.4."
        )


#: The three axis-aligned run inputs and their exact element types. Exact, not
#: "compatible": the identity digests the dtype (`digest_array`), so float32
#: base levels equal in value to the float64 ones would name a different run.
_AXIS_INPUT_DTYPES: tuple[tuple[str, type[np.generic]], ...] = (
    ("required_factor_ids", np.int32),
    ("required_shift_type_id", np.int8),
    ("base_levels", np.float64),
)


def _owned_axis_inputs(
    **arrays: object,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int8],
    npt.NDArray[np.float64],
]:
    """
    Type-check the axis-aligned inputs exactly and take owned execution copies.

    Each input must already be a 1-D, C-contiguous `numpy.ndarray` of its
    exact dtype; nothing is cast, reshaped or re-laid-out. The run then works
    on its own C-contiguous, writable copies: the caller may pass read-only
    arrays (for example straight from the read-only `RunFactorAxis`) and a
    caller buffer mutated after the call cannot reach the run. The copies are
    byte-identical, so the identity is unchanged.
    """
    owned: list[np.ndarray] = []
    for name, dtype in _AXIS_INPUT_DTYPES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise CompositionContractError(
                f"{name} must be a numpy.ndarray; got {type(value).__name__}."
            )
        if value.dtype != np.dtype(dtype):
            raise CompositionContractError(
                f"{name} must have dtype {np.dtype(dtype).name}; got {value.dtype}. "
                "Nothing is cast: the run identity digests the dtype."
            )
        if value.ndim != 1:
            raise CompositionContractError(
                f"{name} must be 1-D (one entry per run-axis row); got "
                f"{value.ndim}-D shape {value.shape}."
            )
        if not value.flags.c_contiguous:
            raise CompositionContractError(f"{name} must be C-contiguous.")
        owned.append(np.array(value, dtype=dtype, copy=True, order="C"))
    return (owned[0], owned[1], owned[2])


def _require_factor_axis(
    *,
    axis: RunFactorAxis,
    required_factor_ids: npt.NDArray[np.int32],
    required_shift_type_id: npt.NDArray[np.int8],
    base_levels: npt.NDArray[np.float64],
) -> None:
    """
    Every axis-aligned run input is bound to the ONE canonical run axis.

    The canonical run axis (D082 as amended, D085) is the valuation prefix
    (the universe map's `required_factor_ids`, row for row) followed by the
    run-derived reporting-only FX suffix. `build_market_data_matrix_from_flat_levels`
    slices the valuation prefix positionally, and the reporting conversion
    gathers its rows by axis position, so a permuted, missing or extra factor
    would read the wrong rows and still return plausible numbers. The inputs
    arrive with exact dtypes (`_owned_axis_inputs`); factor ids and shift
    types are compared ordered and element-wise, and the scenario-data arrays
    must have exactly one entry per axis row.
    """
    k_val = axis.num_valuation_factors
    k_rep = axis.factor_ids.shape[0] - k_val
    for label, canonical_arr, run_arr in (
        ("required_factor_ids", axis.factor_ids, required_factor_ids),
        ("required_shift_type_ids", axis.shift_type_ids, required_shift_type_id),
    ):
        if np.array_equal(canonical_arr, run_arr):
            continue
        if canonical_arr.shape != run_arr.shape:
            detail = (
                f"{run_arr.shape[0]} row(s) against a canonical axis of "
                f"{canonical_arr.shape[0]}"
            )
        else:
            row = int(np.flatnonzero(canonical_arr != run_arr)[0])
            detail = (
                f"row {row} is {run_arr[row].item()!r} in the run and "
                f"{canonical_arr[row].item()!r} on the canonical axis"
            )
        raise CompositionContractError(
            f"the run's {label} is not the run's canonical axis (the universe map's "
            f"{k_val} valuation row(s), then {k_rep} reporting-FX row(s)) "
            f"({detail}); the level rows would feed the wrong slots."
        )
    k = axis.factor_ids.shape[0]
    for label, values in (("base_levels", base_levels),):
        if values.shape != (k,):
            raise CompositionContractError(
                f"{label} has shape {values.shape} but the canonical "
                f"run axis has {k} row(s); scenario data must be bound to that axis."
            )


# =============================================================================
# Reporting route (D085): source currency and reporting FX are run-derived
# =============================================================================
#: Admitted instruments whose settlement (P&L) currency is fixed by the ingest
#: underlying -> settlement FX fold. Any other instrument fails explicitly.
_SOURCE_CCY_INSTRUMENT_IDS: npt.NDArray[np.int32] = np.array(
    sorted((SPOT_ID, VANILLA_FORWARD_NODE_ID, VANILLA_EUROPEAN_OPTION_NODE_ID)),
    dtype=np.int32,
)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _ReportingRoute:
    """
    Run-derived reporting route, keyed to the ascending economic UIDs.

    `fx_rf_ids[e]` is the configured FX factor that converts UID e's settlement
    currency into the reporting currency (`NO_FX_RF_REQUIRED` for an identity
    conversion) and `invert[e]` is 1 when only the reciprocal pair is quoted.
    """

    axis: RunFactorAxis
    settlement_ccy_ids: npt.NDArray[np.int32]
    fx_rf_ids: npt.NDArray[np.int32]
    invert: npt.NDArray[np.int8]


def _settlement_ccy_ids(
    *, context: ValuationContext, economic_uids: npt.NDArray[np.int64]
) -> npt.NDArray[np.int32]:
    """
    Settlement currency id of every economic UID, from compact ingest state.

    Ingest folds each row's underlying currency into its settlement currency
    through `fx_u2s` (D060): with no FX leg the settlement currency is the
    underlying's; with a leg it is the quote side of the configured pair
    (`invert = 0`) or its base side (`invert = 1`). Rows of one UID must share
    one settlement currency, or V_p would sum values in different currencies.
    """
    batch = context.ingest_batch
    static_config = context.static_config
    instruments = np.asarray(batch.instrument_ids, dtype=np.int32)
    unsupported = ~np.isin(instruments, _SOURCE_CCY_INSTRUMENT_IDS)
    if bool(unsupported.any()):
        raise CompositionContractError(
            "no settlement-currency derivation exists for instrument id(s) "
            f"{np.unique(instruments[unsupported]).tolist()}."
        )
    underlying = np.asarray(batch.underlying_ids, dtype=np.int64)
    ccy_of_rf = np.asarray(static_config.rf_id_to_ccy_id)
    known = (underlying >= 0) & (underlying < ccy_of_rf.shape[0])
    settle = np.full(underlying.shape[0], -1, dtype=np.int32)
    settle[known] = ccy_of_rf[underlying[known]]
    fx_ids = np.asarray(batch.fx_u2s_ids, dtype=np.int32)
    has_fx = fx_ids != int(NO_FX_RF_REQUIRED)
    if bool(has_fx.any()):
        matrix = np.asarray(static_config.fx_spot_matrix)
        base_ccy, quote_ccy = np.nonzero(matrix != int(NO_FX_RF_REQUIRED))
        pair_rf = matrix[base_ccy, quote_ccy]
        order = np.argsort(pair_rf)
        pair_rf, base_ccy, quote_ccy = pair_rf[order], base_ccy[order], quote_ccy[order]
        at = np.searchsorted(pair_rf, fx_ids[has_fx])
        at_clipped = np.minimum(at, pair_rf.shape[0] - 1)
        if not bool(np.array_equal(pair_rf[at_clipped], fx_ids[has_fx])):
            raise CompositionContractError(
                "an ingest FX leg names a factor that is not a configured FX pair."
            )
        inverted = np.asarray(batch.fx_u2s_inv, dtype=np.int8)[has_fx] == 1
        settle[has_fx] = np.where(inverted, base_ccy[at_clipped], quote_ccy[at_clipped])
    if bool((settle < 0).any()):
        raise CompositionContractError(
            "the settlement currency of UID(s) "
            f"{np.unique(np.asarray(batch.uids)[settle < 0])[:10].tolist()} cannot be "
            "derived: the underlying carries no configured currency."
        )
    row_uids = np.asarray(batch.uids, dtype=np.int64)
    order = np.argsort(row_uids, kind="stable")
    sorted_uids, sorted_settle = row_uids[order], settle[order]
    uids, first = np.unique(sorted_uids, return_index=True)
    per_uid = sorted_settle[first]
    counts = np.diff(np.append(first, sorted_uids.shape[0]))
    mixed = np.repeat(per_uid, counts) != sorted_settle
    if bool(mixed.any()):
        raise CompositionContractError(
            f"UID(s) {np.unique(sorted_uids[mixed])[:10].tolist()} have valuation "
            "rows settling in different currencies; one economic position has one "
            "settlement currency."
        )
    if not np.array_equal(uids, economic_uids):
        raise CompositionContractError(
            "the ingest batch's UIDs are not the compiled portfolio's economic UIDs."
        )
    return per_uid.astype(np.int32, copy=False)


def _reporting_route(
    *,
    context: ValuationContext,
    universe_map: UniverseMapping,
    economic_uids: npt.NDArray[np.int64],
    reporting_ccy: str,
) -> _ReportingRoute:
    """
    Derive the settlement -> reporting FX route and the canonical run axis.

    The FX authority is the static configuration's `fx_spot_matrix` (D060: one
    orientation per pair): the direct pair (settlement, reporting) gives
    q = x, else the reciprocal pair (reporting, settlement) gives q = 1 / x,
    else the route is refused. The run axis is the universe map's
    `required_factor_ids` followed by the reporting factors it lacks, ascending.
    """
    static_config = context.static_config
    reporting_id = static_config.ccy_name_to_id.get(reporting_ccy)
    if reporting_id is None:
        raise CompositionContractError(
            f"reporting currency {reporting_ccy!r} is not configured in the run's "
            "static configuration."
        )
    settle = _settlement_ccy_ids(context=context, economic_uids=economic_uids)
    matrix = np.asarray(static_config.fx_spot_matrix)
    none = int(NO_FX_RF_REQUIRED)
    direct = matrix[settle, int(reporting_id)]
    reciprocal = matrix[int(reporting_id), settle]
    needs = settle != int(reporting_id)
    unroutable = needs & (direct == none) & (reciprocal == none)
    if bool(unroutable.any()):
        names = np.asarray(static_config.ccy_id_to_name)
        raise CompositionContractError(
            f"no configured FX pair converts {sorted(set(names[settle[unroutable]].tolist()))} "
            f"into the reporting currency {reporting_ccy!r} (neither orientation)."
        )
    use_reciprocal = needs & (direct == none)
    fx_rf_ids = np.where(
        needs, np.where(use_reciprocal, reciprocal, direct), none
    ).astype(np.int32)
    prefix = np.asarray(universe_map.required_factor_ids, dtype=np.int32)
    suffix = np.setdiff1d(fx_rf_ids[needs], prefix).astype(np.int32)
    axis = RunFactorAxis(
        factor_ids=np.concatenate((prefix, suffix)).astype(np.int32),
        shift_type_ids=np.concatenate((
            np.asarray(universe_map.required_shift_type_ids, dtype=np.int8),
            np.asarray(static_config.rf_id_to_shift_type_id, dtype=np.int8)[suffix],
        )),
        num_valuation_factors=int(prefix.shape[0]),
    )
    return _ReportingRoute(
        axis=axis,
        settlement_ccy_ids=settle,
        fx_rf_ids=fx_rf_ids,
        invert=use_reciprocal.astype(np.int8),
    )


def _conversion_plan(
    *,
    route: _ReportingRoute,
    universe_positions: npt.NDArray[np.intp],
    static_config: RiskStaticConfig,
    reporting_ccy: str,
) -> tuple[FxConversionPlan, npt.NDArray[np.int32], tuple[str, ...]]:
    """
    The D081 conversion plan on sorted(U), plus the run-axis rows it reads.

    The reporting FX matrix holds one row per distinct FX factor in use,
    ascending by rf_id; each position points at its row, or at `INVALID_UIX`
    for an identity conversion. Returns the plan, those rows' positions on
    the run axis and their factor names (identity metadata only).
    """
    fx_rf = route.fx_rf_ids[universe_positions]
    needs = fx_rf != int(NO_FX_RF_REQUIRED)
    used = np.unique(fx_rf[needs]).astype(np.int32)
    axis_ids = route.axis.factor_ids
    sorter = np.argsort(axis_ids, kind="stable")
    axis_rows = sorter[np.searchsorted(axis_ids, used, sorter=sorter)].astype(np.int32)
    fx_row_index = np.full(fx_rf.shape[0], INVALID_UIX, dtype=np.int32)
    fx_row_index[needs] = np.searchsorted(used, fx_rf[needs]).astype(np.int32)
    names = np.asarray(static_config.ccy_id_to_name)
    plan = FxConversionPlan(
        fx_row_index=fx_row_index,
        invert=np.where(needs, route.invert[universe_positions], 0).astype(np.int8),
        settlement_ccys=tuple(
            names[route.settlement_ccy_ids[universe_positions]].tolist()
        ),
        reporting_ccy=reporting_ccy,
    )
    rf_name = {int(v): k for k, v in static_config.rel_name_to_rf_id.items()}
    return plan, axis_rows, tuple(rf_name[int(r)] for r in used.tolist())


def derive_run_factor_axis(
    *, valuation_context: ValuationContext, reporting_ccy: str
) -> RunFactorAxis:
    """
    The canonical run axis of a (valuation context, reporting currency) pair.

    Callers bind every axis-aligned input (factor ids, shift types, base
    levels, history columns) to exactly this order;
    `run_scenario_var_es` derives the same axis and refuses any other.
    """
    universe_map = build_universe_mapping(
        static_config=valuation_context.static_config,
        uix_mapper=valuation_context.uix_mapper,
        discounting_mode=valuation_context.discounting_mode,
    )
    return _reporting_route(
        context=valuation_context,
        universe_map=universe_map,
        economic_uids=np.unique(
            np.asarray(valuation_context.portfolio.trade_uids, dtype=np.int64)
        ),
        reporting_ccy=ReportingSpec(reporting_ccy=reporting_ccy).reporting_ccy,
    ).axis


def _require_taylor_block(
    *, block: TaylorBlock, uids: npt.NDArray[np.int64], axis: FactorAxis, label: str
) -> None:
    """A Taylor block prices exactly its declared partition, on the Taylor
    valuation prefix of the canonical run axis (D082 as amended).

    The calculator gathers scenario rows by position, so its compiled axis must
    be the valuation prefix row for row; reporting-only FX rows are never
    Taylor factors. Checked here, before any numerics; the calculator
    re-checks it at pricing time.
    """
    if not block.calculator.axis.matches(axis):
        raise CompositionContractError(
            f"the {label} book was compiled against a factor axis of "
            f"{block.calculator.axis.num_factors} factor(s) that is not the run's "
            f"{axis.num_factors}-factor valuation prefix, row for row. Compile the "
            "book on the valuation prefix; its rows are never re-mapped implicitly."
        )
    book_uids = block.calculator.book.position_uids
    if not np.array_equal(book_uids, uids):
        missing = np.setdiff1d(uids, book_uids)
        extra = np.setdiff1d(book_uids, uids)
        raise CompositionContractError(
            f"the {label} book does not price exactly the {label} partition: "
            f"{missing.size} declared UID(s) absent from the book (first 10: "
            f"{missing[:10].tolist()}), {extra.size} book UID(s) not declared "
            f"{label} (first 10: {extra[:10].tolist()})."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ValuationRoute:
    """The run's one valuation route: its compiled representation, its mapper
    and the universe map the run BUILT from its static configuration."""

    portfolio: CompiledPortfolio
    uix_mapper: UixMapper
    universe_map: UniverseMapping


@dataclass(frozen=True, slots=True, kw_only=True)
class _RowAggregation:
    """
    Scenario-independent map from valuation rows to economic positions.

    Built once per run (compile-once / price-many). `economic_uids` is the
    ascending unique parent UID of the rows; `row_order` groups rows by parent
    UID (stable, so each UID's rows keep their compiled order and the sum is
    deterministic); `segment_starts[e]` is the first grouped row of UID `e`.
    """

    economic_uids: npt.NDArray[np.int64]
    row_order: npt.NDArray[np.intp]
    segment_starts: npt.NDArray[np.intp]


def _row_aggregation(
    *, portfolio: CompiledPortfolio, universe: npt.NDArray[np.int64]
) -> _RowAggregation:
    """
    Reconcile the compiled representation with U and plan V_p = sum_{uid_r = p} V_r.

    `portfolio.trade_uids` is the PARENT economic UID of each valuation row;
    it is neither unique nor sorted (D045/VAL-05, D055), so it is never used as
    an axis or searched. Its unique set must be U exactly (Gate A): a valuation
    row of an undeclared UID and a declared UID without a row both fail here,
    before any value exists that could be dropped or zero-filled.
    """
    row_uids = np.asarray(portfolio.trade_uids, dtype=np.int64)
    economic_uids, inverse = np.unique(row_uids, return_inverse=True)
    unexpected = np.setdiff1d(economic_uids, universe, assume_unique=True)
    missing = np.setdiff1d(universe, economic_uids, assume_unique=True)
    if unexpected.size or missing.size:
        raise CompositionContractError(
            "the compiled portfolio's economic UIDs are not the production "
            f"universe: {unexpected.size} valued UID(s) are undeclared (first 10: "
            f"{unexpected[:10].tolist()}), {missing.size} UID(s) of U have no "
            f"valuation row (first 10: {missing[:10].tolist()}). Rows are never "
            "dropped and absent positions are never zero-filled."
        )
    counts = np.bincount(inverse.ravel(), minlength=economic_uids.shape[0])
    segment_starts = np.zeros(economic_uids.shape[0], dtype=np.intp)
    np.cumsum(counts[:-1], out=segment_starts[1:])
    return _RowAggregation(
        economic_uids=economic_uids,
        row_order=np.argsort(inverse.ravel(), kind="stable"),
        segment_starts=segment_starts,
    )


def _positions(
    economic_uids: npt.NDArray[np.int64], uids: npt.NDArray[np.int64]
) -> npt.NDArray[np.intp]:
    """Index of each UID on the (ascending, unique) economic axis; membership
    is proven, never assumed (Gate B)."""
    positions = np.searchsorted(economic_uids, uids)
    n = economic_uids.shape[0]
    if uids.size and not (
        bool((positions < n).all())
        and bool(np.array_equal(economic_uids[np.minimum(positions, n - 1)], uids))
    ):
        raise CompositionContractError("a declared UID has no valuation row.")
    return positions.astype(np.intp, copy=False)


def _certify_taylor_block(*, block: TaylorBlock, expected: TaylorBaseBinding) -> None:
    """
    P1-A gate with a RUN-DERIVED expectation (P1-B).

    Only a CERTIFIED Taylor representation reaches production P&L, and only if
    its binding is exactly the base this run derives from its own context: as-of,
    base levels, discounting mode, static configuration, valuation contract and
    the run-computed V_0 of the block's positions.
    """
    declaration = block.calculator.taylor_declaration
    if declaration is None:
        raise TaylorContractError(
            "the sensitivity book is UNCERTIFIED (no TaylorDeclaration): its "
            "order, F(p), completeness and base binding are unproven, so it "
            "cannot produce production P&L."
        )
    verify_base_binding(representation=declaration.base_binding, expected=expected)


# =============================================================================
# Identity
# =============================================================================
def _context_identity(
    *,
    context: ValuationContext,
    universe_map: UniverseMapping,
    static_config_digest: str,
    fingerprint: str,
) -> RunContextIdentity:
    return RunContextIdentity(
        as_of_ord32=int(context.uix_mapper.as_of_ord32),
        static_config_digest=static_config_digest,
        valuation_contract_fingerprint=fingerprint,
        discounting_mode=context.discounting_mode.value,
        portfolio_digest=hash_compiled_portfolio(context.portfolio),
        uix_mapper_fingerprint=context.uix_mapper.fingerprint,
        universe_map_digest=hash_universe_mapping(universe_map),
    )


# =============================================================================
# Valuation route: base state and FULL_REVAL block
# =============================================================================
def _value_positions(
    *,
    route: _ValuationRoute,
    aggregation: _RowAggregation,
    flat_levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    `(E, S)` float64 values of every economic position (ascending UID) from a
    `(K, S)` level slice.

    The ONE valuation route for the base state and for scenarios: flat levels ->
    market-data matrix -> valuation state -> `run_valuation(out=)` -> sum rows
    by parent UID.
    """
    num_scenarios = flat_levels.shape[1]
    mapper = route.uix_mapper
    matrix = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(int(mapper.as_of_ord32)),
        uix_mapper=mapper,
        universe_map=route.universe_map,
        flat_levels=flat_levels,
        num_scenarios=num_scenarios,
        discounting_mode=route.universe_map.discounting_mode,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    rows = np.empty((route.portfolio.total_trades, num_scenarios), dtype=np.float64)
    run_valuation(portfolio=route.portfolio, valuation_state=state, out=rows)
    return np.add.reduceat(
        rows[aggregation.row_order], aggregation.segment_starts, axis=0
    )


def _write_full_reval_block(
    *,
    route: _ValuationRoute,
    aggregation: _RowAggregation,
    full_positions: npt.NDArray[np.intp],
    v0_full: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    batch_size: int,
    out: npt.NDArray[np.float32],
) -> None:
    """
    Write dV = V(x_s) - V(x_0) into the FULL block.

    `v0_full` is the run's own base valuation (same route, S = 1). Scenario
    columns are valued in chunks of `batch_size` (operational only: each column
    is priced independently). The float64 difference is quantised into the
    float32 grid exactly like the Taylor blocks.
    """
    total = shocked_levels.shape[1]
    written = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        vs = _value_positions(
            route=route,
            aggregation=aggregation,
            flat_levels=np.ascontiguousarray(shocked_levels[:, start:stop]),
        )[full_positions]
        out[:, start:stop] = vs - v0_full[:, None]
        written += stop - start
    if written != total:
        raise RuntimeError(f"FULL_REVAL wrote {written} of {total} scenario columns.")


def _grid_to_production_rows(
    *, grid: PnlGrid, universe: npt.NDArray[np.int64]
) -> npt.NDArray[np.intp]:
    """Row of `universe` for each grid row; proven a bijection, never assumed."""
    m = universe.shape[0]
    rows = np.searchsorted(universe, grid.trade_uids)
    bijective = (
        grid.trade_uids.shape[0] == m
        and bool((rows < m).all())
        and bool(np.array_equal(universe[np.minimum(rows, m - 1)], grid.trade_uids))
        and bool(np.array_equal(np.sort(rows), np.arange(m)))
    )
    if not bijective:
        raise CompositionContractError(
            "the composed PnlGrid rows are not exactly the production universe."
        )
    return rows


# =============================================================================
# Entry point
# =============================================================================
def run_scenario_var_es(
    *,
    scenario_config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    required_factor_ids: npt.NDArray[np.int32],
    required_shift_type_id: npt.NDArray[np.int8],
    proxy_terms: Sequence[AuthoredProxyTerm],
    base_levels: npt.NDArray[np.float64],
    methodology: MethodologyDeclaration,
    valuation_context: ValuationContext,
    reporting: ReportingSpec,
    confidence_level: float,
    batch_size: int,
    taylor_derived: TaylorBlock | None = None,
    taylor_supplied: TaylorBlock | None = None,
    mode: ReproducibilityMode = "reproducible",
    reproduce: RunManifest | None = None,
) -> RiskRun:
    """
    Run one scenario-based VaR/ES over a mixed-methodology book.

    Parameters
    ----------
    required_factor_ids, required_shift_type_id
        The canonical run axis and each node's configured shift type (the
        target chart kappa), in axis order. The axis must equal, row for row,
        `derive_run_factor_axis(valuation_context=..., reporting_ccy=...)`: the
        valuation prefix, then the reporting-only FX suffix (D085). Exact
        dtypes int32 / int8, 1-D, C-contiguous; nothing is cast, and the run
        works on its own copies.
    proxy_terms
        The authored Proxy V1 terms (`NodeProxyTerm` / `ContainerProxyTerm`).
        The run compiles them with its own static configuration, as-of and
        full run axis (D088 §12.1, §13.8(a)); every V1 axis node without an
        authored term gets its identity row. The compiled operator's identity,
        `hash_proxy_model`, is the manifest's `shocks.proxy_model_digest`.
        Data availability never selects a model: every referenced
        `(d, tau, h)` world column must exist, or the run is refused before
        identity and numerics.
    base_levels
        Base levels x_0 on the same axis, `(K_run,)` float64, C-contiguous.
    methodology
        The explicit production universe U and its partition U_F ⊔ U_S ⊔ U_D.
    valuation_context
        The run-owned base valuation context (P1-B). The run derives its as-of,
        static-configuration and valuation-contract identity from it and
        computes every V_0 from it.
    reporting
        The reporting currency. Settlement currencies and the reporting FX
        route are derived by the run from the compiled positions (D085).
    taylor_derived
        Required iff U_D is non-empty. Its certified binding must equal the one
        the run derives; its book must price exactly U_D on the Taylor
        valuation prefix of the run axis.
    taylor_supplied
        Refused, as is a non-empty U_S, while owner policy P-1 is open.
    batch_size
        Operational only. The result is invariant to it (D041); it is
        deliberately absent from the identity.
    mode
        `"reproducible"` refuses a stochastic model with `seed=None` before any
        numerical work; `"exploratory"` permits it and marks the result.
    reproduce
        A prior manifest to reproduce against. When given, the environment and
        every supplied input are verified BEFORE execution and any mismatch
        raises; a stale or substituted snapshot never reaches the numerics.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    # The VaR/ES owner's own domain check, run before identity and numerics.
    _validate_confidence_level(confidence_level)
    # D090: a parametric world's requested and effective forecast periods must
    # be equal before the run is accepted, identified or computed. The reduced
    # h = count(H_W) is the only horizon the (d, τ, h) binder will receive.
    # D092: the crisis-multiplier form declared by the configuration must match
    # the data (scalar <-> no data vector; "per_factor" <-> data vector).
    if isinstance(scenario_config, ParametricGeneratorConfig):
        resolve_parametric_binding_horizon(scenario_config, data)
        resolve_parametric_crisis_multipliers(scenario_config, data)
    reproducible = _is_reproducible(scenario_config)
    if mode == "reproducible" and not reproducible:
        raise NonReproducibleRunError(
            "A reproducible run requires an explicit seed for a stochastic model; "
            "pass mode='exploratory' to run with seed=None (result marked "
            "non-reproducible)."
        )

    # ---- composition contract: structure before identity and numerics -------
    # Exact dtype / 1-D / C-contiguous, then run-owned copies used downstream.
    (
        required_factor_ids,
        required_shift_type_id,
        base_levels,
    ) = _owned_axis_inputs(
        required_factor_ids=required_factor_ids,
        required_shift_type_id=required_shift_type_id,
        base_levels=base_levels,
    )
    universe = methodology.production_position_uids
    u_full, u_supplied, u_derived = methodology.blocks()
    _require_block_wiring(
        methodology=methodology,
        taylor_supplied=taylor_supplied,
        taylor_derived=taylor_derived,
    )
    context = valuation_context
    _require_taylor_capable_spot(
        context=context, taylor_uids=np.concatenate((u_supplied, u_derived))
    )
    _require_forward_capability(context=context, methodology=methodology)
    aggregation = _row_aggregation(portfolio=context.portfolio, universe=universe)
    universe_positions = _positions(aggregation.economic_uids, universe)
    full_positions = _positions(aggregation.economic_uids, u_full)
    universe_map = build_universe_mapping(
        static_config=context.static_config,
        uix_mapper=context.uix_mapper,
        discounting_mode=context.discounting_mode,
    )
    # D085: settlement currencies, reporting FX factors and the canonical run
    # axis are derived here, once; the caller supplies only the reporting ccy.
    route = _reporting_route(
        context=context,
        universe_map=universe_map,
        economic_uids=aggregation.economic_uids,
        reporting_ccy=reporting.reporting_ccy,
    )
    _require_factor_axis(
        axis=route.axis,
        required_factor_ids=required_factor_ids,
        required_shift_type_id=required_shift_type_id,
        base_levels=base_levels,
    )
    # ---- D088: the run-bound Proxy V1 operator over the FULL run axis ---------
    # M = f(authored terms, static configuration, as-of, run axis) (§12.1). The
    # axis is the valuation prefix plus the reporting-FX suffix (§13.8(a)): a
    # suffix node with a V1 chart is a compiled row like any other node; the
    # reporting conversion below only reads its scenario level.
    as_of_ord32 = context.uix_mapper.as_of_ord32
    if as_of_ord32 is None:
        raise CompositionContractError(
            "the valuation context's mapper carries no as-of; the proxy model is "
            "compiled at the run as-of (D088 §12.2)."
        )
    proxy_model = compile_proxy_model(
        authored_terms=proxy_terms,
        static_config=context.static_config,
        as_of_date=ord_to_date(int(as_of_ord32)),
        run_axis_rf_ids=route.axis.factor_ids,
    )
    # Execution operator: V1 rows from the model, native ABSOLUTE_BASE rows as
    # one unit term on their own column, NO_SHIFT rows with no term (§13.4).
    # kappa per row is the target definition, never a term chart.
    operator = lower_axis_operator(
        model=proxy_model,
        axis_rf_ids=route.axis.factor_ids,
        axis_shift_type_ids=route.axis.shift_type_ids,
    )
    # Every requested (d, tau) bound to the world's (d, tau, h) before identity
    # and before any scenario is generated; a missing column or a horizon
    # mismatch fails here. h is the world's (D088 §12.5).
    world_binding = bind_operator_to_world(operator, scenario_config, data)
    plan, fx_axis_rows, fx_factor_names = _conversion_plan(
        route=route,
        universe_positions=universe_positions,
        static_config=context.static_config,
        reporting_ccy=reporting.reporting_ccy,
    )
    # Valuation and Taylor read the valuation prefix only (D082 as amended).
    k_val = route.axis.num_valuation_factors
    valuation_factor_ids = required_factor_ids[:k_val]
    taylor_axis: FactorAxis | None = None
    if taylor_derived is not None:
        taylor_axis = FactorAxis.from_keys(
            pl.Series("factor_id", valuation_factor_ids.tolist(), dtype=pl.Int32)
        )
        _require_taylor_block(
            block=taylor_derived,
            uids=u_derived,
            axis=taylor_axis,
            label="TAYLOR/DERIVED",
        )

    # ---- identity: every context field derived by the run itself (P1-B) ------
    static_digest = static_config_content_digest(context.static_config)
    fingerprint = valuation_contract_fingerprint()
    composition = CompositionIdentity(
        methodology=MethodologyIdentity(partition_digest=methodology.digest()),
        context=_context_identity(
            context=context,
            universe_map=universe_map,
            static_config_digest=static_digest,
            fingerprint=fingerprint,
        ),
    )
    manifest = build_run_manifest(
        scenario_config=scenario_config,
        data=data,
        proxy_model=proxy_model,
        base_levels=base_levels,
        sensitivities_digest=_NO_TAYLOR_REPRESENTATION_DIGEST
        if taylor_derived is None
        else hash_compiled_book(taylor_derived.calculator.book),
        reporting_currency=reporting.reporting_ccy,
        fx_digest=digest_reporting_plan(
            reporting_ccy=reporting.reporting_ccy,
            settlement_ccys=plan.settlement_ccys,
            fx_factor_names=fx_factor_names,
            fx_factor_rows=fx_axis_rows,
        ),
        confidence_level=confidence_level,
        composition=composition,
    )
    identity = manifest.identity()

    # ---- fail fast BEFORE numerics when reproducing a prior run --------------
    if reproduce is not None:
        verify_environment(reproduce)
        verify_run_inputs(
            reproduce,
            scenario_config=scenario_config,
            data=data,
            proxy_model=proxy_model,
            base_levels=base_levels,
            confidence_level=confidence_level,
            reporting_currency=reporting.reporting_ccy,
            **(
                {}
                if taylor_derived is None
                else {"book": taylor_derived.calculator.book}
            ),
        )
        if manifest.pnl.composition != reproduce.pnl.composition:
            raise InputMismatchError(
                "run context or methodology mismatch: the partition, the as-of, "
                "the static configuration, the valuation contract or the compiled "
                "representation differ from the manifest."
            )
        if identity != reproduce.identity():
            # every verifiable input matched, so the residue is the reporting
            # plan or the environment: name it rather than compute a different run
            raise RuntimeError(
                "run identity differs from the manifest being reproduced although "
                "all verifiable inputs match; check the reporting plan."
            )

    # ---- base state: V_0 = V(x_0) for EVERY position, once, on the one route --
    valuation_route = _ValuationRoute(
        portfolio=context.portfolio,
        uix_mapper=context.uix_mapper,
        universe_map=universe_map,
    )
    v0_economic = _value_positions(
        route=valuation_route,
        aggregation=aggregation,
        flat_levels=np.ascontiguousarray(base_levels[:k_val, None], dtype=np.float64),
    )[:, 0]
    base_settlement_values = np.ascontiguousarray(v0_economic[universe_positions])

    # ---- P1-A gate against the RUN-DERIVED binding, before scenario numerics --
    if taylor_derived is not None:
        v0_derived = v0_economic[_positions(aggregation.economic_uids, u_derived)]
        _certify_taylor_block(
            block=taylor_derived,
            expected=TaylorBaseBinding(
                as_of_ord32=int(context.uix_mapper.as_of_ord32),
                base_levels_digest=base_levels_digest(
                    factor_ids=np.asarray(valuation_factor_ids, dtype=np.int32),
                    base_levels=np.asarray(base_levels[:k_val], dtype=np.float64),
                ),
                discounting_mode=context.discounting_mode.value,
                static_config_digest=static_digest,
                valuation_contract_fingerprint=fingerprint,
                base_settlement_values_digest=base_settlement_values_digest(
                    position_uids=u_derived, values=v0_derived
                ),
            ),
        )

    # ---- execute: scenarios -> ONE canonical shocked-level buffer ------------
    # The scenario world belongs to the run; a batch is only an execution
    # partition of it. The batches must tile [0, S) exactly once, in order:
    # a repeated, skipped or reordered batch would change the world while the
    # identity (built from the configuration) still names the original (PR-06).
    generator = build_scenario_generator(scenario_config, data=data)
    total_scenarios = int(generator.total_scenarios)
    k = base_levels.shape[0]
    chunks: list[npt.NDArray[np.float64]] = []
    written = 0
    # The world streams exactly the bound (d, tau) columns, in request order;
    # the operator's terms index them through world_binding.term_column_uix.
    # Loadings are the compiled float64 betas; nothing else weights a column.
    for batch in generator.iter_batches(
        driver_rf_ids=world_binding.request_driver_rf_ids,
        chart_ids=world_binding.request_chart_ids,
        batch_size=batch_size,
    ):
        count = int(batch.shocks.shape[0])
        if (
            batch.scenario_offset != written
            or batch.scenario_count_total != total_scenarios
            or written + count > total_scenarios
        ):
            raise RuntimeError(
                f"scenario batch [{batch.scenario_offset}, "
                f"{batch.scenario_offset + count}) of a declared total "
                f"{batch.scenario_count_total} does not continue the run's "
                f"{total_scenarios}-scenario stream at scenario {written}."
            )
        out = np.empty((k, count), dtype=np.float64, order="C")
        apply_linear_terms_to_base_levels(
            base_levels=base_levels,
            row_term_offsets=operator.row_term_offsets,
            term_column_uix=world_binding.term_column_uix,
            term_loadings=operator.term_loadings,
            target_shift_types=operator.target_shift_type_ids,
            shocks=batch.shocks,
            out=out,
        )
        chunks.append(out)
        written += count
    if written != total_scenarios:
        raise RuntimeError(
            f"the scenario stream ended after {written} of {total_scenarios} scenarios."
        )
    shocked_levels = np.ascontiguousarray(np.hstack(chunks), dtype=np.float64)
    num_scenarios = shocked_levels.shape[1]

    # ---- settlement P&L: one grid, disjoint blocks [U_F | U_S | U_D] ---------
    grid = allocate_pnl_grid(
        block_position_uids=[u_full, u_supplied, u_derived], num_scenarios=num_scenarios
    )
    if u_full.size:
        _write_full_reval_block(
            route=valuation_route,
            aggregation=aggregation,
            full_positions=full_positions,
            v0_full=v0_economic[full_positions],
            shocked_levels=shocked_levels[:k_val],
            batch_size=batch_size,
            out=grid.block_view(_FULL_BLOCK),
        )
    provenance = None
    if taylor_derived is not None and taylor_axis is not None:
        provenance = taylor_derived.calculator.run_into_grid(
            taylor_axis.levels(
                base_levels=base_levels[:k_val], shocked_levels=shocked_levels[:k_val]
            ),
            grid=grid,
            block_index=_DERIVED_BLOCK,
        )

    # ---- canonical economic axis sorted(U): one scatter, f32 -> f64 ----------
    production_rows = _grid_to_production_rows(grid=grid, universe=universe)
    m = universe.shape[0]
    settlement_pnl = np.empty((m, num_scenarios), dtype=np.float64, order="C")
    settlement_pnl[production_rows] = grid.pnl

    # ---- reporting currency: the ONE D081 conversion for every methodology ---
    converted = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl,
        fx_levels=np.ascontiguousarray(shocked_levels[fx_axis_rows], dtype=np.float64),
        plan=plan,
        base_settlement_values=base_settlement_values,
        base_fx_levels=np.ascontiguousarray(
            base_levels[fx_axis_rows], dtype=np.float64
        ),
    )

    # ---- VaR / ES: positions then portfolio, one canonical call ---------------
    groups = np.ascontiguousarray(
        np.vstack([converted.converted_pnl, converted.portfolio_pnl[None, :]]),
        dtype=np.float64,
    )
    metrics = compute_var_es(groups, confidence_level)

    position_uids = np.array(universe, dtype=np.int64, copy=True)
    position_uids.setflags(write=False)
    return RiskRun(
        position_uids=position_uids,
        var=np.asarray(metrics.var, dtype=np.float64),
        es=np.asarray(metrics.es, dtype=np.float64),
        reporting_ccy=reporting.reporting_ccy,
        manifest=manifest,
        identity=identity,
        reproducible=reproducible,
        pnl_provenance=provenance,
    )
