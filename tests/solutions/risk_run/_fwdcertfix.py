# tests/solutions/risk_run/_fwdcertfix.py
"""
Production-scale VANILLA_FORWARD_NODE certification world.

Consumer: `scripts/release/forward_node/certify_forward_node_production_scale.py`.
`scripts/demos/e2e/_forward_world.py` holds fixed 7-name level tables and one
2-pillar curve, so it cannot reach 250 rf ids and 140 axis points without a
change. This module builds a separate world with production entry points only:

    raw Polars frame
      -> compile_raw_df_into_ingest_batch   (dispatcher, identity gate, D101/D065)
      -> build_uix_mapper                   (run as-of, D067)
      -> ValuationContext                   (compiled VanillaForwardBlock)
      -> derive_run_factor_axis             (valuation prefix + reporting suffix)

Dimensions (the SPOT release dimensions, `spot_production_scale_certifcate.json`):

    K_NODES   250  history columns: 246 test-owned pillars, USDEUR and the 3
                   relative rf ids BRENT [M+0], BRENT [M+1], WTI [M+0]
    K_AXIS    140  run-axis points: 136 exact pillars, 3 relative names, USDEUR
    M         200  forward positions, one row each (SPOT release M = 200)
    T         600  daily observations (historical S = 599)
    S        2000  scenarios of each stochastic model

Static configuration: the canonical tables plus 6 test-owned USD LABEL + PRICE
curves with 41 pillars each. Curve charts cycle LOG_RETURN, SIMPLE_RETURN,
DIFFERENCE, so the axis covers all three V1 charts. Pillar rf ids are sparse.

Book (DiscountingMode.NONE, reporting currency USD). UIDs are sparse; the raw
frame order is a permutation, so ingest order is not ascending. Position i of
the generation order is TAYLOR/DERIVED when i is odd and FULL_REVAL when i is
even. FX legs (USDEUR quotes EUR per USD; every underlying is USD-quoted):

    FULL_REVAL      (settle, strike) cycles (USD,USD) (EUR,USD) (USD,EUR) (EUR,EUR)
                    (USD,EUR) is the inverted strike leg, FULL_REVAL only (D103.3)
    TAYLOR/DERIVED  (settle, strike) cycles (USD,USD) (EUR,USD) (EUR,EUR)

Each leg links the discount curve of its own currency, so ingest admits the
row under D066. Exact NODE rows omit `delivery_date` on every third position,
so ingest admits them under D101.3. DiscountingMode.NONE gives DF = 1 and the
oracle has no DF input, so the numeric checks on this world do not cover D066
or D101.3: this world exercises both decisions at admission only. Relative NODE
rows carry a delivery date inside the resolved contract period (D101.4, D065).

`ORACLE_ROWS` is written from the raw generation data, never from compiled
state. The TAYLOR/DERIVED book is produced on the U_D sub-batch at the run
base and certified against the binding the run derives (P1-A).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_TYPE_NAME_TO_ID_REGISTRY,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RunFactorAxis,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.book_adapter import (
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

AS_OF: Final[dt.date] = CANONICAL_AS_OF_DATE  # 2026-01-15
AS_OF_ORD32: Final[int] = int(date_to_ord32(AS_OF))
REPORTING_CCY: Final[str] = "USD"
DISCOUNTING: Final[DiscountingMode] = DiscountingMode.NONE
CONFIDENCE_LEVEL: Final[float] = 0.99
BATCH_SIZE: Final[int] = 256

K_NODES: Final[int] = 250
K_AXIS: Final[int] = 140
M: Final[int] = 200
T: Final[int] = 600
S: Final[int] = 2_000
#: Position count of the SPOT release world (`_certfix.M`); M must not be smaller.
SPOT_RELEASE_M: Final[int] = 200

FX_NAME: Final[str] = "USDEUR"
USDEUR_BASE: Final[float] = 0.9137
RELATIVE_NAMES: Final[tuple[str, ...]] = (
    "BRENT_CRUDE_FWD[M+0]",
    "BRENT_CRUDE_FWD[M+1]",
    "WTI_CRUDE_FWD[M+0]",
)
#: Delivery and settlement of the relative names: inside the contract period
#: that each name resolves to at AS_OF (D065; dates of `_fwdfix`).
_RELATIVE_DATES: Final[MappingProxyType[str, tuple[dt.date, dt.date]]] = (
    MappingProxyType({
        "BRENT_CRUDE_FWD[M+0]": (dt.date(2026, 2, 16), dt.date(2026, 2, 18)),
        "BRENT_CRUDE_FWD[M+1]": (dt.date(2026, 3, 16), dt.date(2026, 3, 18)),
        "WTI_CRUDE_FWD[M+0]": (dt.date(2026, 2, 16), dt.date(2026, 2, 18)),
    })
)
_EXACT_DELIVERY: Final[dt.date] = dt.date(2026, 3, 16)
_EXACT_SETTLEMENT: Final[dt.date] = dt.date(2026, 3, 18)

_NUM_CURVES: Final[int] = 6
_PILLARS_PER_CURVE: Final[int] = 41
_CURVE_CHARTS: Final[tuple[str, ...]] = ("LOG_RETURN", "SIMPLE_RETURN", "DIFFERENCE")
_DISCOUNT_CURVE: Final[MappingProxyType[str, str]] = MappingProxyType({
    "USD": "USD_SOFR_OIS",
    "EUR": "EUR_ESTR_OIS",
})
_FULL_FX: Final[tuple[tuple[str, str], ...]] = (
    ("USD", "USD"),
    ("EUR", "USD"),
    ("USD", "EUR"),
    ("EUR", "EUR"),
)
_DERIVED_FX: Final[tuple[tuple[str, str], ...]] = (
    ("USD", "USD"),
    ("EUR", "USD"),
    ("EUR", "EUR"),
)
_SEED: Final[int] = 20261007

IDENTITY, DIRECT, INVERTED = "identity", "direct", "inverted"

F64: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class OracleRow:
    """Raw economics of one forward position, as the trade record states it.

    `fx_asset` and `fx_strike` are the orientation of each leg into the
    settlement currency; `fx_report` is settlement -> reporting.
    """

    uid: int
    underlying: str
    quantity: float
    strike: float
    fx_asset: str
    fx_strike: str
    fx_report: str
    derived: bool


def _pillars() -> tuple[tuple[int, str, str, tuple[tuple[int, str], ...]], ...]:
    """(curve_id, curve_name, chart, ((rf_id, tenor), ...)) of the test curves."""
    curves = []
    for c in range(_NUM_CURVES):
        points = tuple(
            (30_000 + 7 * (c * _PILLARS_PER_CURVE + p), f"[{p + 1}M]")
            for p in range(_PILLARS_PER_CURVE)
        )
        curves.append((
            250_020_000 + 1_000 * c,
            f"RQCERT_FWD_{c}",
            _CURVE_CHARTS[c % len(_CURVE_CHARTS)],
            points,
        ))
    return tuple(curves)


CURVES: Final = _pillars()


def static_config_dfs() -> dict[str, pl.DataFrame]:
    """Canonical tables plus the test-owned LABEL + PRICE curves."""
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm, cp = dfs["curve_meta"], dfs["curve_point_meta"]
    meta: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for curve_id, name, chart, pts in CURVES:
        row: dict[str, object] = dict.fromkeys(cm.columns)
        row.update(
            curve_id=curve_id,
            curve_name=name,
            quote_ccy="USD",
            asset_class="COMMODITY",
            asset_subclass="CM_OIL",
            quote_type="PRICE",
            unit_scale="DECIMAL",
            shift_type=chart,
            x_axis_kind="LABEL",
            x_interp_method="LINEAR",
            x_extrap_method="FLAT",
            discount_basis="NOT_A_DISCOUNT_CURVE",
            is_default_discount_for_ccy=False,
        )
        meta.append(row)
        points += [
            {"rf_id": rf, "curve_id": curve_id, "tenor_token": t} for rf, t in pts
        ]
    dfs["curve_meta"] = pl.concat([cm, pl.DataFrame(meta, schema=cm.schema)])
    dfs["curve_point_meta"] = pl.concat([cp, pl.DataFrame(points, schema=cp.schema)])
    return dfs


def chart_by_name() -> dict[str, str]:
    """Configured chart of every history factor, read from the raw tables."""
    dfs = static_config_dfs()
    node = dict(
        zip(dfs["node_meta"]["node_name"], dfs["node_meta"]["shift_type"], strict=True)
    )
    curve = dict(
        zip(
            dfs["curve_meta"]["curve_name"],
            dfs["curve_meta"]["shift_type"],
            strict=True,
        )
    )
    names = [FX_NAME, *RELATIVE_NAMES]
    names += [f"{name}{t}" for _, name, _, pts in CURVES for _, t in pts]
    return {n: str(node.get(n) or curve[n.split("[")[0]]) for n in names}


def _orientation(from_ccy: str, to_ccy: str) -> str:
    if from_ccy == to_ccy:
        return IDENTITY
    if (from_ccy, to_ccy) == ("USD", "EUR"):
        return DIRECT
    if (from_ccy, to_ccy) == ("EUR", "USD"):
        return INVERTED
    raise ValueError(f"no USDEUR route {from_ccy} -> {to_ccy}")


def _levels_and_history(
    rng: np.random.Generator, names: list[str]
) -> tuple[dict[str, float], dict[str, F64]]:
    """Daily log random walk per name; the base level is the as-of (last) value."""
    hist: dict[str, F64] = {}
    for n in names:
        start, vol = (
            (USDEUR_BASE, 0.004)
            if n == FX_NAME
            else (float(rng.uniform(40.0, 160.0)), float(rng.uniform(0.005, 0.03)))
        )
        steps = rng.normal(0.0, vol, T - 1)
        walk = start * np.exp(np.concatenate(([0.0], np.cumsum(steps))))
        hist[n] = np.ascontiguousarray(walk, dtype=np.float64)
    return {n: float(h[-1]) for n, h in hist.items()}, hist


def _book(
    rng: np.random.Generator, exact: list[str], base: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[int, OracleRow]]:
    """M raw forward rows (permuted) and the oracle view of each UID."""
    underlyings = list(RELATIVE_NAMES) + list(
        rng.choice(
            np.array(exact), size=K_AXIS - 1 - len(RELATIVE_NAMES), replace=False
        )
    )
    chosen = underlyings + [
        underlyings[int(j)]
        for j in rng.integers(0, len(underlyings), M - len(underlyings))
    ]
    uids = rng.choice(np.arange(1, 5_000_000, dtype=np.int64), size=M, replace=False)
    rows: list[dict[str, Any]] = []
    oracle: dict[int, OracleRow] = {}
    for i, name in enumerate(chosen):
        derived = i % 2 == 1
        table = _DERIVED_FX if derived else _FULL_FX
        settle, strike_ccy = table[(i // 2) % len(table)]
        qty = float(np.round(rng.normal(0.0, 100.0), 2)) or 1.0
        fwd_in_strike_ccy = base[name] * (USDEUR_BASE if strike_ccy == "EUR" else 1.0)
        strike = float(np.round(fwd_in_strike_ccy * (1.0 + rng.normal(0.0, 0.05)), 2))
        if name in _RELATIVE_DATES:
            delivery: dt.date | None = _RELATIVE_DATES[name][0]
            settlement = _RELATIVE_DATES[name][1]
        else:
            delivery = None if i % 3 == 0 else _EXACT_DELIVERY
            settlement = _EXACT_SETTLEMENT
        uid = int(uids[i])
        rows.append({
            "uid": uid,
            "instrument": "VANILLA_FORWARD_NODE",
            "quantity": qty,
            "underlying": name,
            "settlement_ccy": settle,
            "strike_price": strike,
            "strike_ccy": strike_ccy,
            "delivery_date": delivery,
            "settlement_date": settlement,
            "under_discount_curve": _DISCOUNT_CURVE["USD"],
            "strike_discount_curve": _DISCOUNT_CURVE[strike_ccy],
        })
        oracle[uid] = OracleRow(
            uid=uid,
            underlying=name,
            quantity=qty,
            strike=strike,
            fx_asset=_orientation("USD", settle),
            fx_strike=_orientation(strike_ccy, settle),
            fx_report=_orientation(settle, REPORTING_CCY),
            derived=derived,
        )
    order = rng.permutation(M)
    return [rows[int(j)] for j in order], oracle


def positions_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Polars frame of the raw rows; null where a row has no delivery date."""
    columns = list(rows[0])
    return pl.DataFrame(
        [{c: r[c] for c in columns} for r in rows],
        schema_overrides={
            "uid": pl.Int64,
            "delivery_date": pl.Date,
            "settlement_date": pl.Date,
        },
        infer_schema_length=None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CertWorld:
    """Every production object of the certification run, plus the raw record."""

    rsc: RiskStaticConfig
    rows: tuple[dict[str, Any], ...]
    oracle_rows: MappingProxyType[int, OracleRow]
    context: ValuationContext
    axis: RunFactorAxis
    names: tuple[str, ...]  # factor name of each run-axis row
    base: F64  # base level of each run-axis row
    base_by_name: MappingProxyType[str, float]
    history_by_name: MappingProxyType[str, F64]
    chart_by_name: MappingProxyType[str, str]
    history: RiskFactorMatrix
    parametric: ParametricRiskData
    uids: npt.NDArray[np.int64]  # sorted production universe U
    full_uids: npt.NDArray[np.int64]
    derived_uids: npt.NDArray[np.int64]
    taylor: TaylorBlock

    def row(self, name: str) -> int:
        return self.names.index(name)


def _history_matrix(
    rsc: RiskStaticConfig, hist: dict[str, F64]
) -> tuple[RiskFactorMatrix, npt.NDArray[np.int32], list[str]]:
    rf = dict(rsc.rel_name_to_rf_id)
    pairs = sorted((int(rf[n]), n) for n in hist)
    ids = np.array([p[0] for p in pairs], dtype=np.int32)
    names = [p[1] for p in pairs]
    values = np.asfortranarray(np.column_stack([hist[n] for n in names]))
    matrix = RiskFactorMatrix(
        dates=np.arange(AS_OF_ORD32 - T + 1, AS_OF_ORD32 + 1, dtype=np.int32),
        factor_ids=ids,
        values=values,
    )
    return matrix, ids, names


def _parametric(
    ids: npt.NDArray[np.int32],
    names: list[str],
    charts: dict[str, str],
    base: dict[str, float],
) -> ParametricRiskData:
    """Student-t world over all history ids. Vols are in chart units."""
    rng = np.random.default_rng(_SEED + 1)
    k = ids.shape[0]
    a = rng.normal(0.0, 1.0, (k, k))
    cov = a @ a.T / k + np.eye(k) * 0.05
    inv_sd = 1.0 / np.sqrt(np.diag(cov))
    corr = cov * np.outer(inv_sd, inv_sd)
    vols = np.array(
        [0.02 * base[n] if charts[n] == "DIFFERENCE" else 0.02 for n in names],
        dtype=np.float64,
    )
    return ParametricRiskData(
        factor_ids=ids,
        cholesky_matrix=np.ascontiguousarray(np.linalg.cholesky(corr)),
        volatilities=np.ascontiguousarray(vols),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.array(
            [SHIFT_TYPE_NAME_TO_ID_REGISTRY[charts[n]] for n in names], dtype=np.int8
        ),
    )


def _taylor_block(
    rsc: RiskStaticConfig,
    rows: list[dict[str, Any]],
    axis: RunFactorAxis,
    base_by_name: dict[str, float],
) -> TaylorBlock:
    """Producer side: sensitize U_D at the run base and certify the book."""
    batch = compile_raw_df_into_ingest_batch(
        raw_df=positions_frame(rows), static_config=rsc, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch], as_of_ord32=AS_OF_ORD32)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=rsc, uix_mapper=mapper, discounting_mode=DISCOUNTING
    )
    name_of = {int(v): k for k, v in rsc.rel_name_to_rf_id.items()}
    flat = np.array(
        [[base_by_name[name_of[int(f)]]] for f in layout.required_factor_ids],
        dtype=np.float64,
    )
    market = build_market_data_matrix_from_flat_levels(
        discounting_mode=DISCOUNTING,
        as_of_date=AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=np.ascontiguousarray(flat),
        num_scenarios=1,
    )
    values = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(matrix=market, uix_mapper=mapper),
        out=values,
    )
    sens = run_sensitivities(portfolio=portfolio, uix_mapper=mapper, market_data=market)
    k_val = axis.num_valuation_factors
    prefix = np.ascontiguousarray(axis.factor_ids[:k_val], dtype=np.int32)
    prefix_base = np.array([base_by_name[name_of[int(f)]] for f in prefix])
    trade_uids = np.asarray(portfolio.trade_uids, dtype=np.int64)
    order = np.argsort(trade_uids, kind="stable")
    binding = TaylorBaseBinding(
        as_of_ord32=AS_OF_ORD32,
        base_levels_digest=base_levels_digest(
            factor_ids=prefix, base_levels=prefix_base
        ),
        discounting_mode=DISCOUNTING.value,
        static_config_digest=static_config_content_digest(rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=trade_uids[order], values=values[order, 0]
        ),
    )
    spec = build_sensitivity_book_spec(sensitivities=sens, required_factor_ids=prefix)
    spec = dataclasses.replace(
        spec,
        taylor_declaration=derive_taylor_declaration(
            sensitivities=sens,
            order=TaylorOrder.SECOND_ORDER_FULL,
            base_binding=binding,
        ),
    )
    return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


def build_world() -> CertWorld:
    """Build the certification world. Raises when a dimension is not met."""
    rng = np.random.default_rng(_SEED)
    rsc = compile_risk_static_config_from_dict_of_dfs(static_config_dfs())
    charts = chart_by_name()
    exact = [f"{name}{t}" for _, name, _, pts in CURVES for _, t in pts]
    base_by_name, hist = _levels_and_history(rng, [FX_NAME, *RELATIVE_NAMES, *exact])
    rows, oracle = _book(rng, exact, base_by_name)
    batch = compile_raw_df_into_ingest_batch(
        raw_df=positions_frame(rows), static_config=rsc, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch], as_of_ord32=AS_OF_ORD32)
    context = ValuationContext(
        static_config=rsc,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=DISCOUNTING,
    )
    axis = derive_run_factor_axis(
        valuation_context=context, reporting_ccy=REPORTING_CCY
    )
    name_of = {int(v): k for k, v in rsc.rel_name_to_rf_id.items()}
    names = tuple(name_of[int(f)] for f in axis.factor_ids)
    matrix, ids, hist_names = _history_matrix(rsc, hist)
    dims = (len(names), int(ids.shape[0]), len(oracle))
    if dims != (K_AXIS, K_NODES, M) or M < SPOT_RELEASE_M:
        raise RuntimeError(
            f"certification world dimensions: axis {len(names)} (need {K_AXIS}), "
            f"history {ids.shape[0]} (need {K_NODES}), positions {len(oracle)} "
            f"(need {M} >= {SPOT_RELEASE_M})."
        )
    uids = np.array(sorted(oracle), dtype=np.int64)
    derived = np.array([u for u in uids.tolist() if oracle[u].derived], dtype=np.int64)
    full = np.setdiff1d(uids, derived)
    derived_rows = [r for r in rows if oracle[int(r["uid"])].derived]
    return CertWorld(
        rsc=rsc,
        rows=tuple(rows),
        oracle_rows=MappingProxyType(oracle),
        context=context,
        axis=axis,
        names=names,
        base=np.array([base_by_name[n] for n in names], dtype=np.float64),
        base_by_name=MappingProxyType(base_by_name),
        history_by_name=MappingProxyType(hist),
        chart_by_name=MappingProxyType(charts),
        history=matrix,
        parametric=_parametric(ids, hist_names, charts, base_by_name),
        uids=uids,
        full_uids=full,
        derived_uids=derived,
        taylor=_taylor_block(rsc, derived_rows, axis, base_by_name),
    )


#: Scenario models of the run, horizon 1. Stochastic models draw S scenarios.
MODEL_CONFIGS: Final[MappingProxyType[str, dict[str, Any]]] = MappingProxyType({
    "historical": {"model": "historical", "horizon": 1},
    "filtered_historical": {"model": "filtered_historical", "horizon": 1},
    "sbb": {
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": S,
        "expected_block_size": 20.0,
        "seed": 7,
    },
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": S,
        "seed": 7,
    },
    "parametric": {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": S,
        "distribution": {"kind": "student_t", "df": 5.0},
        "seed": 7,
    },
})


def inputs(
    world: CertWorld, model: str, *, batch_size: int = BATCH_SIZE
) -> dict[str, Any]:
    """Keyword arguments of `run_scenario_var_es` for one model."""
    return {
        "scenario_config": parse_scenario_model_config(MODEL_CONFIGS[model]),
        "data": world.parametric if model == "parametric" else world.history,
        "required_factor_ids": world.axis.factor_ids.copy(),
        "required_shift_type_id": world.axis.shift_type_ids.copy(),
        "proxy_terms": (),
        "base_levels": world.base.copy(),
        "methodology": MethodologyDeclaration(
            production_position_uids=world.uids,
            full_reval_uids=world.full_uids,
            taylor_supplied_uids=np.empty(0, dtype=np.int64),
            taylor_derived_uids=world.derived_uids,
        ),
        "valuation_context": world.context,
        "taylor_derived": world.taylor,
        "reporting": ReportingSpec(reporting_ccy=REPORTING_CCY),
        "confidence_level": CONFIDENCE_LEVEL,
        "batch_size": batch_size,
    }
