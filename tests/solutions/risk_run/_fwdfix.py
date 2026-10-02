# tests/solutions/risk_run/_fwdfix.py
"""
WS-B production world for VANILLA_FORWARD_NODE FULL_REVAL evidence.

Everything here is PRODUCTION input and production entry points:

    raw polars frame
      -> compile_raw_df_into_ingest_batch   (dispatcher, identity gate, D101/D065)
      -> build_uix_mapper                   (EFP quads, EDF triples, FX pairs)
      -> ValuationContext                   (canonical compiler: VanillaForwardBlock)
      -> derive_run_factor_axis             (valuation prefix + reporting suffix)
      -> run_scenario_var_es                (D102 admission, FULL_REVAL, D081, VaR/ES)

Configuration: the canonical static configuration plus two test-owned LABEL +
PRICE curves (the canonical tables have no exact-NODE price curve; precedent
`tests/market_data/remediation/test_exact_node_forward_delivery.py`). Their
charts are SIMPLE_RETURN and DIFFERENCE, so the book covers all three charts.

Book (reporting currency EUR; ingest order NOT ascending; forward rows not all
adjacent; DiscountingMode.NONE):

    uid    slot  instrument  underlying              Q      K     K ccy  settle
    90001  F3    FWD NODE    BRENT_CRUDE_FWD[M+0]   +60    80.0  USD    EUR
    2026   S1    SPOT        AAPL_EQ                +15     -     -      EUR
    7301   F1    FWD NODE    TEST_EXACT_SIMPLE[1M]  +120   97.0  USD    USD  delivery D1
    815    F5    FWD NODE    BRENT_CRUDE_FWD[M+1]   +50    74.0  EUR    USD  (inverted K leg)
    333    S2    SPOT        AAPL_EQ                -8      -     -      USD
    4444   F4    FWD NODE    WTI_CRUDE_FWD[M+0]     -90    76.0  USD    EUR
    12     F2    FWD NODE    TEST_EXACT_DIFF[3M]    -75    44.0  USD    USD  no delivery

F1 and F2 are exact NODE (identity FX); F2 omits the delivery date (D101.3).
F3 and F4 are relative NODE with direct USDEUR on both legs; F4's forward level
moves its EUR value. F5 has an inverted strike leg (EUR -> USD = 1 / USDEUR).

The history is synthetic (seeded) and keyed by factor id. On day pair
(D2_DAY, D2_DAY + 1) only TEST_EXACT_DIFF[3M] moves (detector D2).
"""

from __future__ import annotations

import datetime as dt
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.engine import build_uix_mapper
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.utils.dates import date_to_ord32

AS_OF = CANONICAL_AS_OF_DATE  # 2026-01-15
REPORTING_CCY = "EUR"
HISTORY_DAYS = 81  # S = 80 scenarios at horizon 1
D2_DAY = 40

# ---- test-owned exact-NODE curves (sparse point ids, not consecutive) -----------------
_EXACT_CURVES = (
    # (curve_id, name, shift type, ((rf_id, tenor), ...))
    (
        250_013_000,
        "TEST_EXACT_SIMPLE",
        "SIMPLE_RETURN",
        ((14_003, "[1M]"), (14_009, "[3M]")),
    ),
    (250_014_000, "TEST_EXACT_DIFF", "DIFFERENCE", ((15_011, "[3M]"),)),
)

F1, F2, F3, F4, F5, S1, S2 = 7_301, 12, 90_001, 4_444, 815, 2_026, 333
FORWARDS = (F1, F2, F3, F4, F5)
SPOTS = (S1, S2)

D1 = dt.date(2026, 3, 16)


def _fwd(
    uid: int,
    underlying: str,
    qty: float,
    strike: float,
    strike_ccy: str,
    settle_ccy: str,
    delivery: dt.date | None,
    settlement: dt.date,
) -> dict[str, Any]:
    curve = {"USD": "USD_SOFR_OIS", "EUR": "EUR_ESTR_OIS"}
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
        "under_discount_curve": curve["USD"],
        "strike_discount_curve": curve[strike_ccy],
    }


def _spot(uid: int, underlying: str, qty: float, settle_ccy: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "SPOT",
        "quantity": qty,
        "underlying": underlying,
        "settlement_ccy": settle_ccy,
    }


ROWS: tuple[dict[str, Any], ...] = (
    _fwd(
        F3,
        "BRENT_CRUDE_FWD[M+0]",
        60.0,
        80.0,
        "USD",
        "EUR",
        dt.date(2026, 2, 16),
        dt.date(2026, 2, 18),
    ),
    _spot(S1, "AAPL_EQ", 15.0, "EUR"),
    _fwd(
        F1, "TEST_EXACT_SIMPLE[1M]", 120.0, 97.0, "USD", "USD", D1, dt.date(2026, 3, 18)
    ),
    _fwd(
        F5,
        "BRENT_CRUDE_FWD[M+1]",
        50.0,
        74.0,
        "EUR",
        "USD",
        dt.date(2026, 3, 16),
        dt.date(2026, 3, 18),
    ),
    _spot(S2, "AAPL_EQ", -8.0, "USD"),
    _fwd(
        F4,
        "WTI_CRUDE_FWD[M+0]",
        -90.0,
        76.0,
        "USD",
        "EUR",
        dt.date(2026, 2, 16),
        dt.date(2026, 2, 18),
    ),
    _fwd(
        F2, "TEST_EXACT_DIFF[3M]", -75.0, 44.0, "USD", "USD", None, dt.date(2026, 6, 17)
    ),
)

#: Oracle view of each row, written from the raw rows above (never read from the
#: compiled state): underlying factor NAME and the orientation of each FX leg.
#: USDEUR quotes EUR per USD, so USD -> EUR is direct and EUR -> USD is inverted.
ORACLE_ROWS = MappingProxyType({
    F1: (
        "forward",
        "TEST_EXACT_SIMPLE[1M]",
        120.0,
        97.0,
        "identity",
        "identity",
        "USD",
    ),
    F2: ("forward", "TEST_EXACT_DIFF[3M]", -75.0, 44.0, "identity", "identity", "USD"),
    F3: ("forward", "BRENT_CRUDE_FWD[M+0]", 60.0, 80.0, "direct", "direct", "EUR"),
    F4: ("forward", "WTI_CRUDE_FWD[M+0]", -90.0, 76.0, "direct", "direct", "EUR"),
    F5: ("forward", "BRENT_CRUDE_FWD[M+1]", 50.0, 74.0, "identity", "inverted", "USD"),
    S1: ("spot", "AAPL_EQ", 15.0, 0.0, "direct", "identity", "EUR"),
    S2: ("spot", "AAPL_EQ", -8.0, 0.0, "identity", "identity", "USD"),
})
#: Settlement -> reporting (EUR) rate orientation.
REPORTING_RATE = MappingProxyType({"USD": "direct", "EUR": "identity"})
FX_NAME = "USDEUR"

BASE_LEVELS = MappingProxyType({
    "USDEUR": 0.9137,
    "AAPL_EQ": 191.25,
    "BRENT_CRUDE_FWD[M+0]": 81.37,
    "BRENT_CRUDE_FWD[M+1]": 80.92,
    "WTI_CRUDE_FWD[M+0]": 77.18,
    "TEST_EXACT_SIMPLE[1M]": 100.0,
    "TEST_EXACT_DIFF[3M]": 42.5,
})
#: Daily volatility of the synthetic history, by name (distinct per factor).
_VOL = MappingProxyType({
    "USDEUR": 0.004,
    "AAPL_EQ": 0.015,
    "BRENT_CRUDE_FWD[M+0]": 0.021,
    "BRENT_CRUDE_FWD[M+1]": 0.017,
    "WTI_CRUDE_FWD[M+0]": 0.026,
    "TEST_EXACT_SIMPLE[1M]": 0.012,
    "TEST_EXACT_DIFF[3M]": 0.009,
})


def static_config_dfs() -> dict[str, pl.DataFrame]:
    """Canonical tables plus the two test-owned exact-NODE price curves."""
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm, cp = dfs["curve_meta"], dfs["curve_point_meta"]
    meta, points = [], []
    for curve_id, name, shift, pts in _EXACT_CURVES:
        row: dict[str, object] = dict.fromkeys(cm.columns)
        row.update(
            curve_id=curve_id,
            curve_name=name,
            quote_ccy="USD",
            asset_class="COMMODITY",
            asset_subclass="CM_OIL",
            quote_type="PRICE",
            unit_scale="DECIMAL",
            shift_type=shift,
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


def shift_type_by_name() -> dict[str, str]:
    """Chart of every book factor, read from the raw configuration tables."""
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
    return {n: node.get(n) or curve[n.split("[")[0]] for n in BASE_LEVELS}


def history(*, seed: int = 20260115) -> dict[str, np.ndarray]:
    """Synthetic daily levels by NAME; only TEST_EXACT_DIFF moves on D2_DAY."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, x0 in BASE_LEVELS.items():
        steps = rng.normal(0.0, _VOL[name], HISTORY_DAYS - 1)
        if name != "TEST_EXACT_DIFF[3M]":
            steps[D2_DAY] = 0.0
        else:
            steps[D2_DAY] = 0.031
        out[name] = x0 * np.exp(np.concatenate(([0.0], np.cumsum(steps))))
    return out


class World:
    """The book on the real production path, bound to its canonical run axis."""

    def __init__(self, rows: tuple[dict[str, Any], ...] = ROWS) -> None:
        self.rsc = compile_risk_static_config_from_dict_of_dfs(static_config_dfs())
        self.as_of = int(date_to_ord32(AS_OF))
        self.rows = rows
        self.batch = compile_raw_df_into_ingest_batch(
            raw_df=pl.from_pandas(pd.DataFrame(list(rows))),
            static_config=self.rsc,
            as_of_date=AS_OF,
        )
        self.mapper = build_uix_mapper([self.batch], as_of_ord32=self.as_of)
        self.context = ValuationContext(
            static_config=self.rsc,
            ingest_batch=self.batch,
            uix_mapper=self.mapper,
            discounting_mode=DiscountingMode.NONE,
        )
        self.axis = derive_run_factor_axis(
            valuation_context=self.context, reporting_ccy=REPORTING_CCY
        )
        self.rf = dict(self.rsc.rel_name_to_rf_id)
        name_of = {v: k for k, v in self.rf.items()}
        self.names = [name_of[int(f)] for f in self.axis.factor_ids]
        self.base = np.array([BASE_LEVELS[n] for n in self.names], dtype=np.float64)
        self.hist = history()
        self.uids = np.array(sorted(r["uid"] for r in rows), dtype=np.int64)

    def row(self, name: str) -> int:
        return self.names.index(name)

    def data(self, hist: dict[str, np.ndarray] | None = None) -> RiskFactorMatrix:
        hist = self.hist if hist is None else hist
        ids = np.sort(self.axis.factor_ids)
        name_of = {v: k for k, v in self.rf.items()}
        values = np.column_stack([hist[name_of[int(f)]] for f in ids])
        return RiskFactorMatrix(
            dates=np.arange(
                self.as_of - HISTORY_DAYS + 1, self.as_of + 1, dtype=np.int32
            ),
            factor_ids=ids.astype(np.int32),
            values=np.asfortranarray(values),
        )

    def inputs(self, **over: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "scenario_config": parse_scenario_model_config({
                "model": "historical",
                "horizon": 1,
            }),
            "data": self.data(),
            "required_factor_ids": self.axis.factor_ids.copy(),
            "required_shift_type_id": self.axis.shift_type_ids.copy(),
            "proxy_terms": (),
            "base_levels": self.base.copy(),
            "methodology": MethodologyDeclaration(
                production_position_uids=self.uids,
                full_reval_uids=self.uids,
                taylor_supplied_uids=np.empty(0, dtype=np.int64),
                taylor_derived_uids=np.empty(0, dtype=np.int64),
            ),
            "valuation_context": self.context,
            "reporting": ReportingSpec(reporting_ccy=REPORTING_CCY),
            "confidence_level": 0.9,
            "batch_size": 7,
        }
        kw.update(over)
        return kw
