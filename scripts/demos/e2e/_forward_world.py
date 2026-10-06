# scripts/demos/e2e/_forward_world.py
"""
Shared helper for the VANILLA_FORWARD_NODE executable guides (`10_`, `11_`)
and `bench_forward_scaling.py`.

Not a demo. No `main()`. The leading underscore is the repository's "not a
script" convention. Presentation and the `Checks` ledger come from
`_spot_world` (they are product-neutral); everything forward-specific lives
here and calls production APIs only:

    Polars positions frame
      -> compile_raw_df_into_ingest_batch   (ingest: names die, uids and rf ids live)
      -> build_uix_mapper                   (identity graph, as of the run date)
      -> ValuationContext                   (portfolio compile: VanillaForwardBlock)
      -> derive_run_factor_axis             (the run's factor axis)
      -> run_valuation                      (V0 and scenario values)

Static configuration
--------------------
The SPOT world has no PRICE curves, so it cannot hold a forward underlying.
This world uses the production canonical tables
(`riskquant.ingest_fixtures`, which carry the relative forward curves
BRENT_CRUDE_FWD and WTI_CRUDE_FWD, OFFSET_M + CALENDAR_PERIOD, and the FX node
USDEUR) and adds ONE demo-owned exact-NODE curve:

    DEMO_GAS_FWD   quote USD, PRICE, LABEL x-axis, pillars [1M] and [3M]

FX orientation (printed by the chapters): USDEUR quotes EUR per USD.
    USD -> EUR  multiply by USDEUR          direct    (invert = 0)
    EUR -> USD  divide by USDEUR (1 / x)    inverted  (invert = 1)
A forward whose leg is inverted is admitted in FULL_REVAL and refused in
TAYLOR/DERIVED (D103.3). The main book therefore uses direct legs only.

Everything is deterministic: fixed as-of date, fixed row order, and a LOCAL
`np.random.default_rng(seed)` for the synthetic history.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
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
from riskquant.portfolio.engine import build_uix_mapper
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import ValuationContext
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

AS_OF: dt.date = CANONICAL_AS_OF_DATE  # 2026-01-15, fixed
HISTORY_SEED = 20260115

#: The demo-owned exact-NODE curve: (curve_id, name, ((rf_id, tenor), ...)).
DEMO_CURVE = (250_013_000, "DEMO_GAS_FWD", ((14_003, "[1M]"), (14_009, "[3M]")))

#: Base levels by factor NAME (the human-readable edge).
LEVELS: dict[str, float] = {
    "USDEUR": 0.9137,
    "AAPL_EQ": 191.25,
    "BRENT_CRUDE_FWD[M+0]": 81.37,
    "BRENT_CRUDE_FWD[M+1]": 80.92,
    "WTI_CRUDE_FWD[M+0]": 77.18,
    "DEMO_GAS_FWD[1M]": 100.00,
    "DEMO_GAS_FWD[3M]": 104.50,
}
#: Daily volatility of the synthetic history, by factor name.
VOLS: dict[str, float] = {
    "USDEUR": 0.004,
    "AAPL_EQ": 0.015,
    "BRENT_CRUDE_FWD[M+0]": 0.021,
    "BRENT_CRUDE_FWD[M+1]": 0.019,
    "WTI_CRUDE_FWD[M+0]": 0.024,
    "DEMO_GAS_FWD[1M]": 0.018,
    "DEMO_GAS_FWD[3M]": 0.016,
}
DISCOUNT_CURVE = {"USD": "USD_SOFR_OIS", "EUR": "EUR_ESTR_OIS"}
#: Quote currency of each forward underlying (for the FX-orientation tables).
UNDERLYING_CCY = {"BRENT": "USD", "WTI": "USD", "DEMO_GAS": "USD"}


# ---- static configuration ----------------------------------------------------------


def static_config() -> RiskStaticConfig:
    """Canonical tables plus the demo-owned LABEL + PRICE curve, compiled by
    the production compiler (validation included)."""
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm, cp = dfs["curve_meta"], dfs["curve_point_meta"]
    curve_id, name, points = DEMO_CURVE
    row: dict[str, object] = dict.fromkeys(cm.columns)
    row.update(
        curve_id=curve_id,
        curve_name=name,
        quote_ccy="USD",
        asset_class="COMMODITY",
        asset_subclass="CM_NATGAS",
        quote_type="PRICE",
        unit_scale="DECIMAL",
        shift_type="LOG_RETURN",
        x_axis_kind="LABEL",
        x_interp_method="LINEAR",
        x_extrap_method="FLAT",
        discount_basis="NOT_A_DISCOUNT_CURVE",
        is_default_discount_for_ccy=False,
    )
    dfs["curve_meta"] = pl.concat([cm, pl.DataFrame([row], schema=cm.schema)])
    rows = [{"rf_id": rf, "curve_id": curve_id, "tenor_token": t} for rf, t in points]
    dfs["curve_point_meta"] = pl.concat([cp, pl.DataFrame(rows, schema=cp.schema)])
    return compile_risk_static_config_from_dict_of_dfs(dfs)


def resolved_contract(rsc: RiskStaticConfig, relative_name: str) -> str:
    """The absolute contract a relative NODE name resolves to at AS_OF (D077.4)."""
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=AS_OF
    )
    return str(mapping.get_abs_name_for_rel_name(relative_name))


# ---- positions ---------------------------------------------------------------------


def forward(
    uid: int,
    underlying: str,
    qty: float,
    strike: float,
    *,
    strike_ccy: str = "USD",
    settle_ccy: str = "USD",
    delivery: dt.date | None,
    settlement: dt.date,
) -> dict[str, Any]:
    """One VANILLA_FORWARD_NODE ingest row (the human-readable edge)."""
    return {
        "uid": uid,
        "instrument": "VANILLA_FORWARD_NODE",
        "quantity": qty,
        "underlying": underlying,
        "settlement_ccy": settle_ccy,
        "strike_price": strike,
        "strike_ccy": strike_ccy,
        "delivery_date": delivery,
        "settlement_date": settlement,
        "under_discount_curve": DISCOUNT_CURVE[UNDERLYING_CCY[_family(underlying)]],
        "strike_discount_curve": DISCOUNT_CURVE[strike_ccy],
    }


def spot(uid: int, underlying: str, qty: float, settle_ccy: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "SPOT",
        "quantity": qty,
        "underlying": underlying,
        "settlement_ccy": settle_ccy,
    }


def positions_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Polars frame; columns a row does not use are null."""
    columns = sorted({k for r in rows for k in r}, key=lambda c: (c != "uid", c))
    return pl.DataFrame(
        [{c: r.get(c) for c in columns} for r in rows],
        schema_overrides={"delivery_date": pl.Date, "settlement_date": pl.Date},
        infer_schema_length=None,
    )


# ---- world -------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardWorld:
    """Every production object one forward book needs."""

    rsc: RiskStaticConfig
    rows: tuple[dict[str, Any], ...]
    context: ValuationContext
    axis: Any  # RunFactorAxis
    names: tuple[str, ...]  # factor name of each run-axis row
    base: npt.NDArray[np.float64]  # base level of each run-axis row
    data: RiskFactorMatrix  # synthetic daily history on the run axis
    uids: npt.NDArray[np.int64]  # sorted production universe
    reporting_ccy: str

    def level(self, name: str) -> float:
        return float(self.base[self.names.index(name)])

    @property
    def valuation_factor_ids(self) -> npt.NDArray[np.int32]:
        k = self.axis.num_valuation_factors
        return np.asarray(self.axis.factor_ids[:k], dtype=np.int32)


def build_world(
    rows: list[dict[str, Any]],
    *,
    reporting_ccy: str = "USD",
    discounting_mode: DiscountingMode = DiscountingMode.NONE,
    num_observations: int = 81,
    rsc: RiskStaticConfig | None = None,
) -> ForwardWorld:
    rsc = static_config() if rsc is None else rsc
    batch = compile_raw_df_into_ingest_batch(
        raw_df=positions_frame(rows), static_config=rsc, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(AS_OF)))
    context = ValuationContext(
        static_config=rsc,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=discounting_mode,
    )
    axis = derive_run_factor_axis(
        valuation_context=context, reporting_ccy=reporting_ccy
    )
    name_of = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    names = tuple(name_of[int(f)] for f in axis.factor_ids)
    base = np.array([LEVELS[n] for n in names], dtype=np.float64)
    return ForwardWorld(
        rsc=rsc,
        rows=tuple(rows),
        context=context,
        axis=axis,
        names=names,
        base=base,
        data=history(rsc, axis.factor_ids, num_observations),
        uids=np.array(sorted({int(r["uid"]) for r in rows}), dtype=np.int64),
        reporting_ccy=reporting_ccy,
    )


def history(
    rsc: RiskStaticConfig, factor_ids: npt.NDArray[np.int32], num_observations: int
) -> RiskFactorMatrix:
    """Synthetic daily levels (log random walk from the base), keyed by rf id,
    ending on AS_OF. Local RNG: no global random state is touched."""
    rng = np.random.default_rng(HISTORY_SEED)
    name_of = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    ids = np.sort(np.asarray(factor_ids, dtype=np.int32))
    columns = []
    for f in ids:
        name = name_of[int(f)]
        steps = rng.normal(0.0, VOLS[name], num_observations - 1)
        columns.append(LEVELS[name] * np.exp(np.concatenate(([0.0], np.cumsum(steps)))))
    as_of = int(date_to_ord32(AS_OF))
    return RiskFactorMatrix(
        dates=np.arange(as_of - num_observations + 1, as_of + 1, dtype=np.int32),
        factor_ids=ids,
        values=np.asfortranarray(np.column_stack(columns)),
    )


# ---- production valuation ------------------------------------------------------------


def value_by_uid(
    world: ForwardWorld, levels_by_name: dict[str, npt.NDArray[np.float64]]
) -> dict[int, npt.NDArray[np.float64]]:
    """Production valuation (`run_valuation`) of every row at the given levels
    (one column per scenario), summed per UID, in the settlement currency."""
    mapper = world.context.uix_mapper
    layout = build_universe_mapping(
        static_config=world.rsc,
        uix_mapper=mapper,
        discounting_mode=world.context.discounting_mode,
    )
    name_of = {v: k for k, v in world.rsc.rel_name_to_rf_id.items()}
    flat = np.ascontiguousarray(
        np.stack([levels_by_name[name_of[int(f)]] for f in layout.required_factor_ids])
    )
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=world.context.discounting_mode,
        as_of_date=AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=flat,
        num_scenarios=flat.shape[1],
    )
    portfolio = world.context.portfolio
    out = np.empty((portfolio.total_trades, flat.shape[1]), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(matrix=matrix, uix_mapper=mapper),
        out=out,
    )
    uids = np.asarray(portfolio.trade_uids, dtype=np.int64)
    return {int(u): out[uids == u].sum(axis=0) for u in np.unique(uids)}


def base_value_by_uid(world: ForwardWorld) -> dict[int, float]:
    columns = {n: np.array([world.level(n)]) for n in world.names}
    return {u: float(v[0]) for u, v in value_by_uid(world, columns).items()}


# ---- independent hand formula (NONE: every discount factor is 1) ----------------------


def fx_leg(from_ccy: str, to_ccy: str) -> tuple[str, str]:
    """(orientation, description) of converting `from_ccy` into `to_ccy` with
    the only FX node of this world, USDEUR (EUR per USD)."""
    if from_ccy == to_ccy:
        return "identity", "no FX (X = 1)"
    if (from_ccy, to_ccy) == ("USD", "EUR"):
        return "direct", "X = USDEUR (invert = 0)"
    if (from_ccy, to_ccy) == ("EUR", "USD"):
        return "inverted", "X = 1 / USDEUR (invert = 1)"
    raise ValueError(f"no FX route {from_ccy} -> {to_ccy} in this demo world")


def fx_value(orientation: str, usdeur: float) -> float:
    return {"identity": 1.0, "direct": usdeur, "inverted": 1.0 / usdeur}[orientation]


def hand_value(row: dict[str, Any], levels: dict[str, float]) -> float:
    """V = Q * (F * X_a - K * X_k) for a forward; V = Q * S * X for a SPOT.
    Written out here, independent of every production helper."""
    usdeur = levels["USDEUR"]
    if row["instrument"] == "SPOT":
        x = fx_value(fx_leg("USD", row["settlement_ccy"])[0], usdeur)
        return row["quantity"] * levels[row["underlying"]] * x
    und_ccy = UNDERLYING_CCY[_family(row["underlying"])]
    x_a = fx_value(fx_leg(und_ccy, row["settlement_ccy"])[0], usdeur)
    x_k = fx_value(fx_leg(row["strike_ccy"], row["settlement_ccy"])[0], usdeur)
    return row["quantity"] * (
        levels[row["underlying"]] * x_a - row["strike_price"] * x_k
    )


def _family(underlying: str) -> str:
    stem = underlying.split("[")[0]
    return stem.replace("_CRUDE_FWD", "").replace("_FWD", "")


# ---- the main teaching book (chapters 10 and 11) --------------------------------------
#: F1/F2 exact NODE with/without delivery (same node), F3 relative NODE, F4 EUR-settled
#: forward (direct USDEUR legs), S1 SPOT. UIDs are sparse on purpose.
D_FEB, D_MAR = dt.date(2026, 2, 16), dt.date(2026, 3, 16)

MAIN_BOOK: list[dict[str, Any]] = [
    forward(
        7_301,
        "DEMO_GAS_FWD[1M]",
        120.0,
        97.0,
        delivery=D_MAR,
        settlement=dt.date(2026, 3, 18),
    ),
    forward(
        12,
        "DEMO_GAS_FWD[1M]",
        -75.0,
        99.0,
        delivery=None,
        settlement=dt.date(2026, 3, 18),
    ),
    forward(
        90_001,
        "BRENT_CRUDE_FWD[M+0]",
        60.0,
        80.0,
        delivery=D_FEB,
        settlement=dt.date(2026, 2, 18),
    ),
    forward(
        4_444,
        "WTI_CRUDE_FWD[M+0]",
        -90.0,
        76.0,
        settle_ccy="EUR",
        delivery=D_FEB,
        settlement=dt.date(2026, 2, 18),
    ),
    spot(2_026, "AAPL_EQ", 15.0, "USD"),
]
LABELS: dict[int, str] = {7_301: "F1", 12: "F2", 90_001: "F3", 4_444: "F4", 2_026: "S1"}
