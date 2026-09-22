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

Composes ONLY existing production functions (the D036 scenario seam,
`apply_shocks_to_base_levels`, market-data assembly, `run_valuation`, the
certified `SensitivityPnlCalculator`, `PnlGrid`, the FX conversion and
`compute_var_es`). Its own arithmetic is limited to summing valuation rows into
their parent economic UID (D055). The manifest is built from the very objects
handed to the computation, before any numerical work; it never enters a
generator, a kernel or a Numba signature (D001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.market_data.assemble import (
    UniverseMapping,
    build_market_data_matrix_from_flat_levels,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.reporting.fx_conversion import (
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    FullRevalBlock,
    MethodologyDeclaration,
    NonReproducibleRunError,
    ReportingSpec,
    ReproducibilityMode,
    RiskRun,
    TaylorBlock,
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
    FullRevalIdentity,
    MethodologyIdentity,
    RunManifest,
    TaylorBlockIdentity,
)
from riskquant.solutions.scenario_models.api import build_scenario_generator
from riskquant.solutions.scenario_models.schemas import ScenarioModelConfig
from riskquant.solutions.sensitivity_pnl.api import FactorAxis
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorContractError,
    base_levels_digest,
    base_settlement_values_digest,
    valuation_contract_fingerprint,
    verify_base_binding,
)
from riskquant.universe import UixMapper
from riskquant.utils.business_days import ord_to_date
from riskquant.valuation.engine import run_valuation

__all__ = ["run_scenario_var_es"]


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
    full_reval: FullRevalBlock | None,
    taylor_supplied: TaylorBlock | None,
    taylor_derived: TaylorBlock | None,
) -> None:
    """Each methodology block is supplied iff its partition is non-empty."""
    u_full, u_supplied, u_derived = methodology.blocks()
    if taylor_supplied is not None or u_supplied.size:
        raise CompositionContractError(
            f"TAYLOR/SUPPLIED is not admissible: owner policy P-1 is unresolved "
            f"({u_supplied.size} position(s) declared SUPPLIED). A supplied "
            "representation is never priced as DERIVED or FULL_REVAL instead."
        )
    for label, uids, block in (
        ("FULL_REVAL", u_full, full_reval),
        ("TAYLOR/DERIVED", u_derived, taylor_derived),
    ):
        if uids.size and block is None:
            raise CompositionContractError(
                f"{uids.size} position(s) are declared {label} but no {label} "
                "block was supplied."
            )
        if block is not None and not uids.size:
            raise CompositionContractError(
                f"a {label} block was supplied but no position is declared "
                f"{label}; an unused methodology input is a wiring defect."
            )


def _require_reporting_alignment(
    *, reporting: ReportingSpec, universe: npt.NDArray[np.int64]
) -> None:
    """The reporting metadata is keyed to U, in canonical ascending order."""
    if not np.array_equal(reporting.position_uids, universe):
        raise CompositionContractError(
            "ReportingSpec.position_uids is not the production universe in "
            "ascending order; settlement currencies would be read for the wrong "
            "positions."
        )


def _require_factor_axis(
    *,
    universe_map: UniverseMapping,
    required_factor_ids: npt.NDArray[np.int32],
    required_shift_type_id: npt.NDArray[np.int8],
) -> None:
    """
    The run's shock axis IS the market-data layout axis, row for row.

    `build_market_data_matrix_from_flat_levels` slices flat level rows by the
    universe map's positions, so a permuted, missing or extra factor would
    price every slot from the wrong row and still return plausible PVs. Equality
    is ordered and element-wise; lengths or sets are never enough.
    """
    for label, layout, run_axis in (
        ("required_factor_ids", universe_map.required_factor_ids, required_factor_ids),
        (
            "required_shift_type_ids",
            universe_map.required_shift_type_ids,
            required_shift_type_id,
        ),
    ):
        layout_arr, run_arr = np.asarray(layout), np.asarray(run_axis)
        if np.array_equal(layout_arr, run_arr):
            continue
        if layout_arr.shape != run_arr.shape:
            detail = (
                f"{run_arr.shape[0]} row(s) against a layout of {layout_arr.shape[0]}"
            )
        else:
            row = int(np.flatnonzero(layout_arr != run_arr)[0])
            detail = (
                f"row {row} is {run_arr[row].item()!r} in the run and "
                f"{layout_arr[row].item()!r} in the universe map"
            )
        raise CompositionContractError(
            f"the run's {label} is not the FULL_REVAL universe map's ({detail}); "
            "the flat level rows would feed the wrong market-data slots."
        )


def _require_taylor_block(
    *, block: TaylorBlock, uids: npt.NDArray[np.int64], axis: FactorAxis, label: str
) -> None:
    """A Taylor block prices exactly its declared partition, on the run axis.

    The calculator gathers scenario rows by position, so its compiled axis must
    be the run's axis row for row. Checked here, before any numerics; the
    calculator re-checks it at pricing time.
    """
    if not block.calculator.axis.matches(axis):
        raise CompositionContractError(
            f"the {label} book was compiled against a factor axis of "
            f"{block.calculator.axis.num_factors} factor(s) that is not the run's "
            f"{axis.num_factors}-factor axis, row for row. Compile the book on the "
            "run axis; its rows are never re-mapped implicitly."
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
class _RowAggregation:
    """
    Scenario-independent map from valuation rows to economic positions.

    Built once per run (compile-once / price-many). `row_order` groups rows by
    parent UID (stable, so each UID's rows keep their compiled order and the
    sum is deterministic); `segment_starts[e]` is the first grouped row of
    economic UID `e`; `full_positions` indexes U_F on the economic axis.
    """

    row_order: npt.NDArray[np.intp]
    segment_starts: npt.NDArray[np.intp]
    full_positions: npt.NDArray[np.intp]


def _full_reval_row_aggregation(
    *,
    portfolio: CompiledPortfolio,
    universe: npt.NDArray[np.int64],
    full_uids: npt.NDArray[np.int64],
) -> _RowAggregation:
    """
    Reconcile the compiled representation with U and plan V_p = sum_{uid_r = p} V_r.

    `portfolio.trade_uids` is the PARENT economic UID of each valuation row;
    it is neither unique nor sorted (D045/VAL-05, D055), so it is never used as
    an axis or searched. Its unique set must be U exactly: a valuation row of
    an undeclared UID and a declared UID without a row both fail here, before
    any value exists that could be dropped or zero-filled.
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
    positions = np.searchsorted(economic_uids, full_uids)
    covered = bool((positions < economic_uids.shape[0]).all()) and bool(
        np.array_equal(
            economic_uids[np.minimum(positions, economic_uids.shape[0] - 1)], full_uids
        )
    )
    if not covered:
        raise CompositionContractError(
            "a FULL_REVAL UID has no valuation row in the compiled portfolio."
        )
    counts = np.bincount(inverse.ravel(), minlength=economic_uids.shape[0])
    segment_starts = np.zeros(economic_uids.shape[0], dtype=np.intp)
    np.cumsum(counts[:-1], out=segment_starts[1:])
    return _RowAggregation(
        row_order=np.argsort(inverse.ravel(), kind="stable"),
        segment_starts=segment_starts,
        full_positions=positions.astype(np.intp, copy=False),
    )


def _certify_taylor_block(
    *,
    block: TaylorBlock,
    required_factor_ids: npt.NDArray[np.int32],
    base_levels: npt.NDArray[np.float64],
) -> TaylorBaseBinding:
    """
    P1-A: only a CERTIFIED Taylor representation reaches production P&L, and
    only at the exact base it was certified at. The levels and V_0 digests are
    re-derived from the run's own inputs, so q_0 (read from the same base
    levels) is bound too. Returns the certified binding.
    """
    declaration = block.calculator.taylor_declaration
    if declaration is None:
        raise TaylorContractError(
            "the sensitivity book is UNCERTIFIED (no TaylorDeclaration): its "
            "order, F(p), completeness and base binding are unproven, so it "
            "cannot produce production P&L."
        )
    verify_base_binding(
        representation=declaration.base_binding, expected=block.expected_binding
    )
    if (
        base_levels_digest(
            factor_ids=np.asarray(required_factor_ids, dtype=np.int32),
            base_levels=np.asarray(base_levels, dtype=np.float64),
        )
        != declaration.base_binding.base_levels_digest
    ):
        raise TaylorContractError(
            "the run's base levels are not the base the Taylor representation "
            "was produced at (base_levels_digest mismatch)."
        )
    if (
        base_settlement_values_digest(
            position_uids=block.calculator.book.position_uids,
            values=block.base_settlement_values,
        )
        != declaration.base_binding.base_settlement_values_digest
    ):
        raise TaylorContractError(
            "V_0 is not the base value the Taylor representation was bound to "
            "(base_settlement_values_digest mismatch)."
        )
    return declaration.base_binding


def _safe_as_of_ord32(as_of_ord32: int | None) -> int | None:
    """Safely extract the as_of_ord32 value, casting to int only if it exists."""
    return int(as_of_ord32) if as_of_ord32 is not None else None


def _require_coherent_base(
    *, full_reval: FullRevalBlock, binding: TaylorBaseBinding, fingerprint: str
) -> None:
    """In a mixed run, FULL_REVAL and TAYLOR/DERIVED value ONE base state."""

    # Safely extract the value, casting to int only if it exists
    full_as_of = full_reval.uix_mapper.as_of_ord32
    safe_full_as_of = _safe_as_of_ord32(full_as_of)

    mismatched = [
        name
        for name, full_value, taylor_value in (
            (
                "as_of_ord32",
                safe_full_as_of,
                binding.as_of_ord32,
            ),
            (
                "discounting_mode",
                full_reval.universe_map.discounting_mode.value,
                binding.discounting_mode,
            ),
            (
                "valuation_contract_fingerprint",
                fingerprint,
                binding.valuation_contract_fingerprint,
            ),
        )
        if full_value != taylor_value
    ]
    if mismatched:
        raise CompositionContractError(
            f"FULL_REVAL and TAYLOR/DERIVED would value different bases: "
            f"mismatched {mismatched}. One run has one base state (D081 §10)."
        )


# =============================================================================
# Identity
# =============================================================================
def _taylor_block_identity(block: TaylorBlock) -> TaylorBlockIdentity:
    """Stated binding plus the digest of the V_0 values actually supplied."""
    b = block.expected_binding
    return TaylorBlockIdentity(
        as_of_ord32=int(b.as_of_ord32),
        base_levels_digest=b.base_levels_digest,
        discounting_mode=b.discounting_mode,
        static_config_digest=b.static_config_digest,
        valuation_contract_fingerprint=b.valuation_contract_fingerprint,
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=block.calculator.book.position_uids,
            values=block.base_settlement_values,
        ),
    )


def _full_reval_identity(
    full_reval: FullRevalBlock, *, fingerprint: str
) -> FullRevalIdentity:
    return FullRevalIdentity(
        portfolio_digest=hash_compiled_portfolio(full_reval.portfolio),
        uix_mapper_fingerprint=full_reval.uix_mapper.fingerprint,
        universe_map_digest=hash_universe_mapping(full_reval.universe_map),
        as_of_ord32=int(full_reval.uix_mapper.as_of_ord32),
        discounting_mode=full_reval.universe_map.discounting_mode.value,
        valuation_contract_fingerprint=fingerprint,
    )


# =============================================================================
# FULL_REVAL block
# =============================================================================
def _value_positions(
    *,
    full_reval: FullRevalBlock,
    aggregation: _RowAggregation,
    flat_levels: npt.NDArray[np.float64],
    uix_mapper: UixMapper,
) -> npt.NDArray[np.float64]:
    """
    `(|U_F|, S)` float64 FULL_REVAL position values from a `(K, S)` level slice.

    The ONE valuation route for base and scenarios: flat levels -> market-data
    matrix -> valuation state -> `run_valuation(out=)` -> sum rows by parent UID
    -> select U_F.
    """
    num_scenarios = flat_levels.shape[1]
    matrix = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(int(uix_mapper.as_of_ord32)),
        uix_mapper=uix_mapper,
        universe_map=full_reval.universe_map,
        flat_levels=flat_levels,
        num_scenarios=num_scenarios,
        discounting_mode=full_reval.universe_map.discounting_mode,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=uix_mapper)
    rows = np.empty(
        (full_reval.portfolio.total_trades, num_scenarios), dtype=np.float64
    )
    run_valuation(portfolio=full_reval.portfolio, valuation_state=state, out=rows)
    economic = np.add.reduceat(
        rows[aggregation.row_order], aggregation.segment_starts, axis=0
    )
    return economic[aggregation.full_positions]


def _write_full_reval_block(
    *,
    full_reval: FullRevalBlock,
    aggregation: _RowAggregation,
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    batch_size: int,
    out: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """
    Write dV = V(x_s) - V(x_0) into the FULL block; return V_0 = V(x_0).

    V_0 is valued ONCE at S = 1 on the same route as the scenarios. Scenario
    columns are valued in chunks of `batch_size` (operational only: each
    column is priced independently). The float64 difference is quantised into
    the float32 grid exactly like the Taylor blocks.
    """
    mapper = full_reval.uix_mapper
    v0 = _value_positions(
        full_reval=full_reval,
        aggregation=aggregation,
        flat_levels=np.ascontiguousarray(base_levels[:, None], dtype=np.float64),
        uix_mapper=mapper,
    )[:, 0]
    total = shocked_levels.shape[1]
    written = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        vs = _value_positions(
            full_reval=full_reval,
            aggregation=aggregation,
            flat_levels=np.ascontiguousarray(shocked_levels[:, start:stop]),
            uix_mapper=mapper,
        )
        out[:, start:stop] = vs - v0[:, None]
        written += stop - start
    if written != total:
        raise RuntimeError(f"FULL_REVAL wrote {written} of {total} scenario columns.")
    return v0


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
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
    base_levels: npt.NDArray[np.float64],
    methodology: MethodologyDeclaration,
    reporting: ReportingSpec,
    confidence_level: float,
    batch_size: int,
    full_reval: FullRevalBlock | None = None,
    taylor_derived: TaylorBlock | None = None,
    taylor_supplied: TaylorBlock | None = None,
    mode: ReproducibilityMode = "reproducible",
    reproduce: RunManifest | None = None,
) -> RiskRun:
    """
    Run one scenario-based VaR/ES over a mixed-methodology book.

    Parameters
    ----------
    required_factor_ids, required_shift_type_id, source_factor_ids, shock_scale
        The shock universe on the required-factor axis, in axis order. These are
        result-defining and are part of the identity (D040) — they are NOT part
        of the scenario config.
    base_levels
        Base levels x_0 on the same axis, `(K,)` float64.
    methodology
        The explicit production universe U and its partition U_F ⊔ U_S ⊔ U_D.
    reporting
        Reporting metadata keyed to U (ascending).
    full_reval
        Required iff U_F is non-empty. Its universe map must carry exactly the
        run's factor axis, in order.
    taylor_derived
        Required iff U_D is non-empty. Its book must price exactly U_D.
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
    reproducible = _is_reproducible(scenario_config)
    if mode == "reproducible" and not reproducible:
        raise NonReproducibleRunError(
            "A reproducible run requires an explicit seed for a stochastic model; "
            "pass mode='exploratory' to run with seed=None (result marked "
            "non-reproducible)."
        )

    # ---- composition contract: structure before identity and numerics -------
    universe = methodology.production_position_uids
    u_full, u_supplied, u_derived = methodology.blocks()
    _require_block_wiring(
        methodology=methodology,
        full_reval=full_reval,
        taylor_supplied=taylor_supplied,
        taylor_derived=taylor_derived,
    )
    _require_reporting_alignment(reporting=reporting, universe=universe)
    aggregation: _RowAggregation | None = None
    fingerprint = ""
    if full_reval is not None:
        _require_factor_axis(
            universe_map=full_reval.universe_map,
            required_factor_ids=required_factor_ids,
            required_shift_type_id=required_shift_type_id,
        )
        aggregation = _full_reval_row_aggregation(
            portfolio=full_reval.portfolio, universe=universe, full_uids=u_full
        )
        fingerprint = valuation_contract_fingerprint()
    run_axis: FactorAxis | None = None
    if taylor_derived is not None:
        run_axis = FactorAxis.from_keys(
            pl.Series("factor_id", required_factor_ids.tolist(), dtype=pl.Int32)
        )
        _require_taylor_block(
            block=taylor_derived, uids=u_derived, axis=run_axis, label="TAYLOR/DERIVED"
        )

    # ---- identity, from the exact objects the computation will consume -------
    composition = CompositionIdentity(
        methodology=MethodologyIdentity(partition_digest=methodology.digest()),
        full_reval=None
        if full_reval is None
        else _full_reval_identity(full_reval, fingerprint=fingerprint),
        taylor_supplied=None,
        taylor_derived=None
        if taylor_derived is None
        else _taylor_block_identity(taylor_derived),
    )
    manifest = build_run_manifest(
        scenario_config=scenario_config,
        data=data,
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=source_factor_ids,
        shock_scale=shock_scale,
        base_levels=base_levels,
        sensitivities_digest=_NO_TAYLOR_REPRESENTATION_DIGEST
        if taylor_derived is None
        else hash_compiled_book(taylor_derived.calculator.book),
        reporting_currency=reporting.reporting_ccy,
        fx_digest=digest_reporting_plan(
            reporting_ccy=reporting.reporting_ccy,
            settlement_ccys=reporting.settlement_ccys,
            fx_factor_names=reporting.fx_factor_names,
            fx_factor_rows=reporting.fx_factor_rows,
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
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            shock_scale=shock_scale,
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
                "methodology composition mismatch: the partition, the FULL_REVAL "
                "inputs or a Taylor binding differ from the manifest."
            )
        if identity != reproduce.identity():
            # every verifiable input matched, so the residue is the reporting
            # plan or the environment: name it rather than compute a different run
            raise RuntimeError(
                "run identity differs from the manifest being reproduced although "
                "all verifiable inputs match; check the reporting plan."
            )

    # ---- P1-A gate and cross-methodology base coherence, before numerics -----
    if taylor_derived is not None:
        binding = _certify_taylor_block(
            block=taylor_derived,
            required_factor_ids=required_factor_ids,
            base_levels=base_levels,
        )
        if full_reval is not None:
            _require_coherent_base(
                full_reval=full_reval, binding=binding, fingerprint=fingerprint
            )

    # ---- execute: scenarios -> ONE canonical shocked-level buffer ------------
    generator = build_scenario_generator(scenario_config, data=data)
    k = base_levels.shape[0]
    chunks: list[npt.NDArray[np.float64]] = []
    for batch in generator.iter_batches(
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=source_factor_ids,
        shock_scale=shock_scale,
        batch_size=batch_size,
    ):
        out = np.empty((k, batch.shocks.shape[0]), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=out)
        chunks.append(out)
    shocked_levels = np.ascontiguousarray(np.hstack(chunks), dtype=np.float64)
    num_scenarios = shocked_levels.shape[1]

    # ---- settlement P&L: one grid, disjoint blocks [U_F | U_S | U_D] ---------
    grid = allocate_pnl_grid(
        block_position_uids=[u_full, u_supplied, u_derived], num_scenarios=num_scenarios
    )
    v0_full = np.empty(0, dtype=np.float64)
    if full_reval is not None and aggregation is not None:
        v0_full = _write_full_reval_block(
            full_reval=full_reval,
            aggregation=aggregation,
            base_levels=base_levels,
            shocked_levels=shocked_levels,
            batch_size=batch_size,
            out=grid.block_view(_FULL_BLOCK),
        )
    provenance = None
    v0_derived = np.empty(0, dtype=np.float64)
    if taylor_derived is not None and run_axis is not None:
        provenance = taylor_derived.calculator.run_into_grid(
            run_axis.levels(base_levels=base_levels, shocked_levels=shocked_levels),
            grid=grid,
            block_index=_DERIVED_BLOCK,
        )
        v0_derived = taylor_derived.base_settlement_values

    # ---- canonical economic axis sorted(U): one scatter, f32 -> f64 ----------
    production_rows = _grid_to_production_rows(grid=grid, universe=universe)
    m = universe.shape[0]
    settlement_pnl = np.empty((m, num_scenarios), dtype=np.float64, order="C")
    settlement_pnl[production_rows] = grid.pnl
    base_settlement_values = np.empty(m, dtype=np.float64)
    base_settlement_values[production_rows] = np.concatenate([
        v0_full,
        np.empty(0, dtype=np.float64),
        v0_derived,
    ])

    # ---- reporting currency: the ONE D081 conversion for every methodology ---
    plan = build_fx_conversion_plan(
        settlement_ccys=list(reporting.settlement_ccys),
        reporting_ccy=reporting.reporting_ccy,
        fx_factor_names=list(reporting.fx_factor_names),
    )
    converted = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl,
        fx_levels=np.ascontiguousarray(
            shocked_levels[reporting.fx_factor_rows], dtype=np.float64
        ),
        plan=plan,
        base_settlement_values=base_settlement_values,
        base_fx_levels=np.ascontiguousarray(
            base_levels[reporting.fx_factor_rows], dtype=np.float64
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
