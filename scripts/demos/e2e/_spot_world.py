# scripts/demos/e2e/_spot_world.py
"""
Shared helper for the numbered SPOT executable-guide demos (`00_` .. `07_`).

Not a demo. No `main()`. The leading underscore is the repository's "not a
script" convention (`scripts/run_all_scripts.py`, `tests/e2e`).

What lives here
---------------
1. Narrative output: ASCII-only, <= 78 columns, readable in Windows
   PowerShell and in redirected text files. No ANSI colour, no box glyphs.
2. `Checks`: every claim a demo prints is an assertion. A failed check raises
   `DemoCheckError` immediately (FAIL FAST); `expect_rejection` accepts ONLY
   the exact expected exception type and lets anything else propagate.
3. World construction: the repetitive, production-API setup that turns
   human-readable tables into the typed objects `run_scenario_var_es` needs.
   `00_spot_getting_started.py` performs the same construction inline, step
   by step; `07_spot_production_example.py` performs the run-input assembly
   inline, as an integration writes it. Every function here calls production
   APIs directly and adds no mathematics of its own.
4. `scenario_pnl_anatomy`: recomposes the SAME core building blocks
   `run_scenario_var_es` composes (generator -> shocks -> levels -> valuation /
   Taylor -> float32 grid -> D081 conversion). It exists to make the scenario
   P&L DISTRIBUTION visible, because a `RiskRun` carries only VaR/ES. Every
   demo that uses it proves the recomposition bitwise-equal to the production
   `RiskRun` before it trusts a single printed number.

What does NOT live here
-----------------------
The production call itself. Every demo calls
`riskquant.solutions.risk_run.api.run_scenario_var_es` visibly.
"""

from __future__ import annotations

import dataclasses
import json
import os
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.market_data.assemble import (
    UniverseMapping,
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_df,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.reporting.fx_conversion import (
    ReportingCurrencyPnl,
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID, ShiftType
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RiskRun,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import build_scenario_generator
from riskquant.solutions.scenario_models.schemas import ScenarioModelConfig
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityPnlCalculator,
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
from riskquant.universe import UixMapper
from riskquant.utils.business_days import ord_to_date
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

# =============================================================================
# 1. NARRATIVE OUTPUT
# =============================================================================

WIDTH: int = 78
_RULE_HEAVY: str = "=" * WIDTH
_RULE_LIGHT: str = "-" * WIDTH
_STATE: dict[str, bool] = {"after_check": False}


def _gap() -> None:
    """One blank line between a run of [PASS] lines and following prose."""
    if _STATE["after_check"]:
        print()
        _STATE["after_check"] = False


def title(text: str) -> None:
    """Demo-level title block."""
    print()
    print(_RULE_HEAVY)
    for line in textwrap.wrap(text.upper(), WIDTH - 4):
        print(f"  {line}")
    print(_RULE_HEAVY)
    print()


def step(label: str, text: str) -> None:
    """One chapter of the guided tour, e.g. `step("STEP 3", "VALUE THE BASE")`."""
    _gap()
    print()
    print(_RULE_LIGHT)
    print(f"{label} - {text.upper()}")
    print(_RULE_LIGHT)
    print()


def para(text: str, *, indent: int = 0) -> None:
    """Wrap one explanatory paragraph to the console width."""
    _gap()
    pad = " " * indent
    for block in textwrap.dedent(text).strip().split("\n\n"):
        joined = " ".join(block.split())
        for line in textwrap.wrap(joined, WIDTH - indent):
            print(f"{pad}{line}")
        print()


def formula(*lines: str) -> None:
    """Print a displayed formula block, indented, verbatim."""
    _gap()
    for line in lines:
        print(f"      {line}")
    print()


def kv(pairs: Sequence[tuple[str, object]], *, indent: int = 4) -> None:
    """Aligned `label : value` lines; long values wrap with a hanging indent."""
    _gap()
    width = max(len(k) for k, _ in pairs)
    lead_len = indent + width + 3
    for key, value in pairs:
        lines = textwrap.wrap(str(value), max(WIDTH - lead_len, 20)) or [""]
        print(f"{' ' * indent}{key:<{width}} : {lines[0]}")
        for line in lines[1:]:
            print(" " * lead_len + line)
    print()


def num(value: float, digits: int = 4) -> str:
    """Fixed-point with thousands separators; negative zero prints as zero."""
    v = float(value)
    return f"{(0.0 if v == 0.0 else v):,.{digits}f}"


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    indent: int = 4,
    align: str | None = None,
) -> None:
    """Plain ASCII table. `align` is one char per column: `l` or `r`."""
    _gap()
    body = [[str(cell) for cell in row] for row in rows]
    cols = len(headers)
    align = align or ("l" + "r" * (cols - 1))
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in body)) if body else len(headers[c])
        for c in range(cols)
    ]

    def fmt_row(cells: Sequence[str]) -> str:
        parts = [
            f"{cell:<{widths[c]}}" if align[c] == "l" else f"{cell:>{widths[c]}}"
            for c, cell in enumerate(cells)
        ]
        return (" " * indent + "  ".join(parts)).rstrip()

    print(fmt_row(headers))
    print(" " * indent + "  ".join("-" * w for w in widths))
    for row in body:
        print(fmt_row(row))
    print()


# =============================================================================
# 2. CHECKS (every printed claim is an assertion)
# =============================================================================


_ASCII_MAP: dict[str, str] = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2294": "U",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00d7": "x",
    "\u2026": "...",
    "\u00a7": "section ",
}


def ascii_text(text: str) -> str:
    """Production messages may contain typographic glyphs; the console
    output of this suite is ASCII-only (Windows PowerShell code pages)."""
    out = "".join(_ASCII_MAP.get(ch, ch) for ch in text)
    return out.encode("ascii", "replace").decode("ascii")


def _field(name: str, text: str) -> None:
    """`           name      : text`, wrapped with a hanging indent."""
    lead = f"           {name:<9} : "
    lines = textwrap.wrap(text, WIDTH - len(lead)) or [""]
    print(lead + lines[0])
    for line in lines[1:]:
        print(" " * len(lead) + line)


class DemoCheckError(AssertionError):
    """A demo claim did not hold. Raised immediately: the demo stops."""


E = TypeVar("E", bound=BaseException)


class Checks:
    """Assertion ledger. `check` raises on the first failure (FAIL FAST)."""

    def __init__(self) -> None:
        self.passed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if not bool(condition):
            raise DemoCheckError(f"[FAIL] {label}" + (f" | {detail}" if detail else ""))
        self.passed.append(label)
        for i, line in enumerate(textwrap.wrap(label, WIDTH - 11)):
            print(f"    {'[PASS]' if i == 0 else '      '} {line}")
        _STATE["after_check"] = True
        if detail:
            for line in textwrap.wrap(detail, WIDTH - 11):
                print(f"           {line}")

    def expect_rejection(
        self,
        label: str,
        expected: type[E],
        call: Callable[[], object],
        *,
        reason: str,
        stage: str,
    ) -> E:
        """
        Invoke a real API with a deliberately invalid input.

        Only `expected` (exact type, not a subclass and not a sibling) counts
        as PASS. Any other exception propagates unchanged, and a call that
        returns normally is a FAIL: the contract did not protect the run.
        """
        try:
            call()
        except expected as exc:
            if type(exc) is not expected:
                raise DemoCheckError(
                    f"[FAIL] {label} | expected exactly {expected.__name__}, got "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            first = ascii_text(" ".join(str(exc).split()))
            self.passed.append(label)
            for i, line in enumerate(textwrap.wrap(label, WIDTH - 11)):
                print(f"    {'[PASS]' if i == 0 else '      '} {line}")
            first = first if len(first) <= 320 else first[:317] + "..."
            _field("exception", expected.__name__)
            _field("raised by", expected.__module__)
            _field("stage", stage)
            _field("why", reason)
            _field("message", first)
            print()
            return exc
        raise DemoCheckError(f"[FAIL] {label} | the invalid input was ACCEPTED.")

    def done(self) -> None:
        _STATE["after_check"] = False
        print()
        print(_RULE_HEAVY)
        print(f"  {len(self.passed)} checks passed.")
        print("  ALL CHECKS PASSED")
        print(_RULE_HEAVY)


def write_evidence(name: str, payload: Mapping[str, Any]) -> Path | None:
    """
    Optional machine-readable evidence.

    Written only when `RQ_DEMO_EVIDENCE_DIR` is set, so a normal console run
    leaves no files behind. Floats are stored as IEEE-754 hex (bitwise
    round-trip), mirroring `RiskRun.to_json`.
    """
    root = os.environ.get("RQ_DEMO_EVIDENCE_DIR")
    if not root:
        return None

    def encode(value: object) -> object:
        if isinstance(value, np.ndarray):
            return encode(value.tolist())
        if isinstance(value, float):
            return float(value).hex()
        if isinstance(value, (np.floating,)):
            return float(value).hex()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, Mapping):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    path = Path(root) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encode(dict(payload)), indent=2, sort_keys=True))
    return path


# =============================================================================
# 3. STATIC CONFIGURATION (edge: human-readable tables -> RiskStaticConfig)
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorDef:
    """One scalar risk-factor node, as it appears in the `node_meta` sheet."""

    node_id: int
    name: str
    quote_ccy: str
    asset_class: str
    asset_subclass: str
    quote_type: str
    shift_type: str
    fx_base_ccy: str | None = None


def equity(node_id: int, name: str, ccy: str, shift: str = "LOG_RETURN") -> FactorDef:
    return FactorDef(
        node_id=node_id,
        name=name,
        quote_ccy=ccy,
        asset_class="EQUITY",
        asset_subclass="EQ_SINGLE_NAME",
        quote_type="PRICE",
        shift_type=shift,
    )


def commodity(
    node_id: int, name: str, ccy: str, shift: str = "LOG_RETURN"
) -> FactorDef:
    return FactorDef(
        node_id=node_id,
        name=name,
        quote_ccy=ccy,
        asset_class="COMMODITY",
        asset_subclass="CM_OIL",
        quote_type="PRICE",
        shift_type=shift,
    )


def fx_rate(node_id: int, pair: str, shift: str = "LOG_RETURN") -> FactorDef:
    """`pair="EURUSD"` multiplies EUR -> USD (FX authority: `portfolio.fx`)."""
    return FactorDef(
        node_id=node_id,
        name=pair,
        quote_ccy=pair[3:],
        asset_class="FX",
        asset_subclass="FX_G10",
        quote_type="FX_RATE",
        shift_type=shift,
        fx_base_ccy=pair[:3],
    )


#: One default OIS curve per currency. The static-config validator requires a
#: discount curve for every active quote currency, even for SPOT (which reads
#: no discount factor). Curve rf_ids sit far above node ids.
_CURVE_TEMPLATE: dict[str, tuple[int, int, str]] = {
    "EUR": (250_000_000, 1_000, "EUR_ESTR_OIS"),
    "USD": (250_001_000, 2_000, "USD_SOFR_OIS"),
    "GBP": (250_002_000, 3_000, "GBP_SONIA_OIS"),
    "CHF": (250_003_000, 4_000, "CHF_SARON_OIS"),
    "JPY": (250_004_000, 5_000, "JPY_TONA_OIS"),
}


def compile_static_config(
    factors: Sequence[FactorDef], currencies: Sequence[str]
) -> RiskStaticConfig:
    """Build the seven static-config sheets and compile them (production API)."""
    ccys = list(currencies)
    for ccy in ccys:
        if ccy not in _CURVE_TEMPLATE:
            raise ValueError(f"no demo discount-curve template for {ccy!r}")
    curves = [_CURVE_TEMPLATE[c] for c in ccys]
    sheets: dict[str, pl.DataFrame] = {
        "ccy": pl.DataFrame({
            "ccy_name": ccys,
            "ccy_id": list(range(1, len(ccys) + 1)),
        }),
        "holiday_calendars": pl.DataFrame({
            "holiday_calendar_name": ["NL", "US"],
            "holiday_date": ["2027-01-01", "2027-01-01"],
        }),
        "node_meta": pl.DataFrame(
            {
                "node_id": [f.node_id for f in factors],
                "node_name": [f.name for f in factors],
                "quote_ccy": [f.quote_ccy for f in factors],
                "asset_class": [f.asset_class for f in factors],
                "asset_subclass": [f.asset_subclass for f in factors],
                "quote_type": [f.quote_type for f in factors],
                "unit_scale": ["DECIMAL"] * len(factors),
                "shift_type": [f.shift_type for f in factors],
                "fx_base_ccy": [f.fx_base_ccy for f in factors],
            },
            schema_overrides={"fx_base_ccy": pl.String},
        ),
        "curve_meta": pl.DataFrame(
            {
                "curve_id": [c[0] for c in curves],
                "curve_name": [c[2] for c in curves],
                "quote_ccy": ccys,
                "asset_class": ["IR"] * len(ccys),
                "asset_subclass": ["IR_OIS"] * len(ccys),
                "quote_type": ["RATE"] * len(ccys),
                "unit_scale": ["PERCENT"] * len(ccys),
                "shift_type": ["DIFFERENCE"] * len(ccys),
                "x_axis_kind": ["LABEL"] * len(ccys),
                "x_interp_method": ["LINEAR"] * len(ccys),
                "x_extrap_method": ["FLAT"] * len(ccys),
                "roll_method": [None] * len(ccys),
                "anchor_offset": [None] * len(ccys),
                "roll_month": [None] * len(ccys),
                "roll_day": [None] * len(ccys),
                "fh_bh_mmdd": [None] * len(ccys),
                "discount_basis": ["OIS"] * len(ccys),
                "is_default_discount_for_ccy": [True] * len(ccys),
            },
            schema_overrides={
                "roll_method": pl.String,
                "anchor_offset": pl.Int32,
                "roll_month": pl.Int32,
                "roll_day": pl.Int32,
                "fh_bh_mmdd": pl.String,
            },
        ),
        "curve_point_meta": pl.DataFrame({
            "rf_id": [c[1] for c in curves],
            "curve_id": [c[0] for c in curves],
            "tenor_token": ["[3M]"] * len(ccys),
        }),
        "surface_meta": _EMPTY_SURFACE_META,
        "surface_point_meta": _EMPTY_SURFACE_POINT_META,
    }
    return compile_risk_static_config_from_dict_of_dfs(sheets)


_EMPTY_SURFACE_META = pl.DataFrame({
    "surface_id": pl.Series([], dtype=pl.Int64),
    "surface_name": pl.Series([], dtype=pl.String),
    "quote_ccy": pl.Series([], dtype=pl.String),
    "asset_class": pl.Series([], dtype=pl.String),
    "asset_subclass": pl.Series([], dtype=pl.String),
    "quote_type": pl.Series([], dtype=pl.String),
    "unit_scale": pl.Series([], dtype=pl.String),
    "shift_type": pl.Series([], dtype=pl.String),
    "x_axis_kind": pl.Series([], dtype=pl.String),
    "x_interp_method": pl.Series([], dtype=pl.String),
    "x_extrap_method": pl.Series([], dtype=pl.String),
    "roll_method": pl.Series([], dtype=pl.String),
    "anchor_offset": pl.Series([], dtype=pl.Int32),
    "roll_month": pl.Series([], dtype=pl.Int32),
    "roll_day": pl.Series([], dtype=pl.Int32),
    "fh_bh_mmdd": pl.Series([], dtype=pl.String),
    "y_axis_kind": pl.Series([], dtype=pl.String),
    "y_interp_method": pl.Series([], dtype=pl.String),
    "y_extrap_method": pl.Series([], dtype=pl.String),
})

_EMPTY_SURFACE_POINT_META = pl.DataFrame({
    "rf_id": pl.Series([], dtype=pl.Int64),
    "surface_id": pl.Series([], dtype=pl.Int64),
    "x_tenor_token": pl.Series([], dtype=pl.String),
    "y_label": pl.Series([], dtype=pl.String),
})


def author_proxy_terms(
    proxies: Sequence[tuple[str, str, ShiftType, float]] = (),
) -> tuple[NodeProxyTerm, ...]:
    """
    `(target_name, driver_name, driver_chart, loading)` rows -> Proxy V1 terms.

    Each row states one term of X_target = sum loading * Y(driver, driver_chart, h)
    (D088). The driver chart is written by the author; it is never taken from
    the target's configured shift type. A factor with no row keeps its
    identity row (target, its own chart, 1.0), compiled by the run.
    """
    return tuple(
        NodeProxyTerm(
            target_name=target,
            driver_name=driver,
            driver_chart=chart,
            loading=float(loading),
        )
        for target, driver, chart, loading in proxies
    )


# =============================================================================
# 4. THE VALUATION WORLD
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotWorld:
    """
    Every typed object one SPOT book needs, built from human-readable inputs.

    Field groups
    ------------
    edge        : `positions` (the raw frame), `levels` (name -> level)
    identity    : `rsc` (names <-> rf_ids), `ingest_batch` (uids),
                  `uix_mapper` (rf_id <-> uix, as-of), `universe_map` (axis)
    valuation   : `portfolio`, `market_data` (S = 1), `v0_by_uid`
    run axis    : `required_factor_ids` .. `base_levels`, one row per factor of
                  the canonical run axis (D082 amended, D085): the valuation
                  universe (`num_valuation_factors` rows, universe-map order),
                  then any reporting-only FX factor `reporting_ccy` needs.
    proxy model : `proxy_terms`, the authored Proxy V1 terms (D088); the run
                  compiles them over its full axis.
    """

    rsc: RiskStaticConfig
    proxy_terms: tuple[NodeProxyTerm, ...]
    positions: pd.DataFrame
    levels: Mapping[str, float]
    as_of: date
    ingest_batch: IngestBatch
    uix_mapper: UixMapper
    portfolio: CompiledPortfolio
    universe_map: UniverseMapping
    snapshot: RiskFactorLevelSnapshot
    market_data: MarketDataMatrix
    v0_by_uid: Mapping[int, float]
    required_factor_ids: npt.NDArray[np.int32]
    required_shift_type_id: npt.NDArray[np.int8]
    base_levels: npt.NDArray[np.float64]
    num_valuation_factors: int
    reporting_ccy: str

    @property
    def valuation_factor_ids(self) -> npt.NDArray[np.int32]:
        """The valuation prefix of the run axis (also the Taylor axis)."""
        return self.required_factor_ids[: self.num_valuation_factors]

    @property
    def reporting_factor_ids(self) -> npt.NDArray[np.int32]:
        """The reporting-only FX suffix of the run axis (possibly empty)."""
        return self.required_factor_ids[self.num_valuation_factors :]

    # ---- name helpers (edge only; nothing below enters a kernel) -----------
    def name_of(self, rf_id: int) -> str:
        return {int(v): k for k, v in self.rsc.rel_name_to_rf_id.items()}[int(rf_id)]

    def rf(self, name: str) -> int:
        return int(self.rsc.rel_name_to_rf_id[name])

    def row(self, name: str) -> int:
        """Row of `name` on the run axis."""
        hits = np.flatnonzero(self.required_factor_ids == self.rf(name))
        if hits.size != 1:
            raise KeyError(f"{name} is not on the run axis")
        return int(hits[0])

    def axis_names(self) -> list[str]:
        return [self.name_of(int(r)) for r in self.required_factor_ids]

    @property
    def uids(self) -> npt.NDArray[np.int64]:
        """Production universe U, ascending."""
        return np.sort(np.asarray(self.positions["uid"], dtype=np.int64))

    def position(self, uid: int) -> pd.Series:
        return self.positions.set_index("uid").loc[int(uid)]


def build_world(
    *,
    rsc: RiskStaticConfig,
    proxy_terms: tuple[NodeProxyTerm, ...],
    positions: pd.DataFrame,
    levels: Mapping[str, float],
    as_of: date,
    reporting_ccy: str = "USD",
) -> SpotWorld:
    """
    positions + levels -> the production valuation world (see `01_...`).

        compile_raw_df_into_ingest_batch  : names die, uids and rf_ids live
        build_uix_mapper                  : rf_id <-> uix, as-of (D067)
        compile_portfolio_from_ingest_batch : SpotBlock SoA
        build_universe_mapping            : the valuation universe (V0)
        build_market_data_matrix_from_snapshot + run_valuation : V0
        derive_run_factor_axis            : THE run axis for `reporting_ccy`

    The scenario law of every run-axis factor is the Proxy V1 model: the
    authored `proxy_terms`, compiled by the run over its full axis (D088).
    """
    as_of_ord = int(date_to_ord32(as_of))
    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(positions), static_config=rsc
    )
    uix_mapper = build_uix_mapper([ingest_batch], as_of_ord32=as_of_ord)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch, uix_mapper=uix_mapper
    )
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=uix_mapper
    )
    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=as_of, levels_dict=dict(levels), static_config=rsc
    )
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=as_of,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        snapshot=snapshot,
        num_scenarios=1,
    )
    state = assemble_valuation_state(matrix=market_data, uix_mapper=uix_mapper)
    out = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    v0_by_uid = _sum_rows_by_uid(portfolio.trade_uids, out[:, 0])

    # The run owns the axis (D082 as amended, D085): scenario data is bound to
    # exactly the axis `run_scenario_var_es` derives for this reporting ccy.
    run_axis = derive_run_factor_axis(
        valuation_context=ValuationContext(
            static_config=rsc,
            ingest_batch=ingest_batch,
            uix_mapper=uix_mapper,
            discounting_mode=DiscountingMode.NONE,
        ),
        reporting_ccy=reporting_ccy,
    )
    # The run axis is a read-only descriptor; the world owns its scenario-data
    # arrays bound to it (the scenario building blocks take writable arrays).
    required = np.array(run_axis.factor_ids, copy=True)
    base_levels = extract_levels_from_snapshot(required_ids=required, snapshot=snapshot)
    return SpotWorld(
        rsc=rsc,
        proxy_terms=tuple(proxy_terms),
        positions=positions.reset_index(drop=True),
        levels=dict(levels),
        as_of=as_of,
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
        portfolio=portfolio,
        universe_map=universe_map,
        snapshot=snapshot,
        market_data=market_data,
        v0_by_uid=v0_by_uid,
        required_factor_ids=required,
        required_shift_type_id=np.array(run_axis.shift_type_ids, copy=True),
        base_levels=base_levels,
        num_valuation_factors=run_axis.num_valuation_factors,
        reporting_ccy=reporting_ccy,
    )


def _sum_rows_by_uid(
    trade_uids: npt.NDArray[np.int64], values: npt.NDArray[np.float64]
) -> dict[int, float]:
    """SPOT: one valuation row per economic uid. Asserted, never assumed."""
    uids = np.asarray(trade_uids, dtype=np.int64)
    if np.unique(uids).shape[0] != uids.shape[0]:
        raise ValueError("a SPOT uid has more than one valuation row")
    return {
        int(u): float(v) for u, v in zip(uids.tolist(), values.tolist(), strict=True)
    }


def history_matrix(
    rsc: RiskStaticConfig,
    *,
    dates: Sequence[str] | pd.DatetimeIndex,
    names: Sequence[str],
    values: npt.ArrayLike,
) -> RiskFactorMatrix:
    """Wide (T, N) history (names on columns) -> canonical RiskFactorMatrix."""
    frame = pd.DataFrame(
        np.asarray(values, dtype=np.float64),
        index=pd.to_datetime(list(dates)),
        columns=list(names),
    )
    return build_risk_factor_matrix_from_df(df=frame, static_config=rsc)


def parametric_data_from_history(matrix: RiskFactorMatrix) -> ParametricRiskData:
    """
    Sample vols and correlation-Cholesky of the matrix's 1-day log returns.

    `ParametricGenerator` scales correlated unit draws by `volatilities`, so it
    takes the Cholesky factor of the CORRELATION matrix, not the covariance.
    The matrix's `factor_ids` are already sorted ascending, as required.
    """
    returns = np.diff(np.log(np.asarray(matrix.values, dtype=np.float64)), axis=0)
    vols = np.ascontiguousarray(np.std(returns, axis=0, ddof=1), dtype=np.float64)
    corr = np.corrcoef(returns, rowvar=False)
    chol = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)
    k = matrix.factor_ids.shape[0]
    return ParametricRiskData(
        factor_ids=np.ascontiguousarray(matrix.factor_ids, dtype=np.int32),
        cholesky_matrix=chol,
        volatilities=vols,
        # The statistics are those of consecutive business-day closes of the
        # history, so every column is the law of a one-business-day log return
        # (D090 period (1, BUSINESS_DAY); D091 chart LOG_RETURN per column).
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(k, SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


# =============================================================================
# 5. RUN INPUTS
# =============================================================================


def valuation_context(world: SpotWorld) -> ValuationContext:
    """The run-owned base valuation context (P1-B, N1)."""
    return ValuationContext(
        static_config=world.rsc,
        ingest_batch=world.ingest_batch,
        uix_mapper=world.uix_mapper,
        discounting_mode=DiscountingMode.NONE,
    )


def reporting_spec(world: SpotWorld, *, reporting_ccy: str | None = None) -> ReportingSpec:
    """
    The caller's reporting choice: the reporting currency only (D085).

    The run derives each position's settlement currency and its reporting FX
    route itself. The world's scenario data is bound to the run axis of the
    currency it was built for; a currency that needs another axis is refused
    here with the rebuild instruction (the run would refuse it anyway).
    """
    ccy = world.reporting_ccy if reporting_ccy is None else reporting_ccy
    if ccy != world.reporting_ccy:
        axis = derive_run_factor_axis(
            valuation_context=valuation_context(world), reporting_ccy=ccy
        )
        if not np.array_equal(axis.factor_ids, world.required_factor_ids):
            raise ValueError(
                f"this world is bound to the {world.reporting_ccy} run axis; "
                f"{ccy} reporting needs another axis: rebuild with reporting_ccy={ccy!r}."
            )
    return ReportingSpec(reporting_ccy=ccy)


def restrict_sensitivities(
    sens: BlockSensitivities, uids: Sequence[int], *, keep_cross: bool
) -> BlockSensitivities:
    """Keep only `uids`' rows (and cross rows only if the order keeps them)."""
    keep = np.isin(sens.position_uids, np.asarray(uids, dtype=np.int64))
    keep_x = (
        np.isin(sens.cross_position_uids, np.asarray(uids, dtype=np.int64)) & keep_cross
    )
    fields: dict[str, Any] = {}
    for f in dataclasses.fields(sens):
        value = getattr(sens, f.name)
        if isinstance(value, np.ndarray):
            fields[f.name] = (
                value[keep_x] if f.name.startswith("cross_") else value[keep]
            )
    return dataclasses.replace(sens, **fields)


def run_base_binding(world: SpotWorld, uids: Sequence[int]) -> TaylorBaseBinding:
    """The exact base a Taylor book of `uids` must be certified at (P1-B)."""
    book_uids = np.sort(np.asarray(uids, dtype=np.int64))
    v0 = np.array([world.v0_by_uid[int(u)] for u in book_uids], dtype=np.float64)
    return TaylorBaseBinding(
        as_of_ord32=int(world.uix_mapper.as_of_ord32),
        base_levels_digest=base_levels_digest(
            factor_ids=world.valuation_factor_ids,
            base_levels=world.base_levels[: world.num_valuation_factors],
        ),
        discounting_mode=DiscountingMode.NONE.value,
        static_config_digest=static_config_content_digest(world.rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book_uids, values=v0
        ),
    )


def sensitivities_for(
    world: SpotWorld, uids: Sequence[int]
) -> BlockSensitivities:
    """
    Production sensitizer output for `uids`, at the world's base.

    Sensitised on the compiled sub-portfolio of `uids` ONLY (U_D-scoped). A
    position outside the Taylor partition never reaches the sensitizer, so an
    inverted-FX leg that is FULL_REVAL only (D084) does not block a Taylor book
    for the rest of the book. Sensitivities are rf_id-keyed (D012); the book
    aligns to the run axis by identity, not by the sub-portfolio's mapper.
    """
    keep = world.positions["uid"].isin([int(u) for u in uids])
    rows = world.positions.loc[keep].reset_index(drop=True)
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(rows), static_config=world.rsc
    )
    mapper = build_uix_mapper(
        [batch], as_of_ord32=int(world.uix_mapper.as_of_ord32)
    )
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE,
        static_config=world.rsc,
        uix_mapper=mapper,
    )
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=world.as_of,
        uix_mapper=mapper,
        universe_map=universe_map,
        snapshot=world.snapshot,
        num_scenarios=1,
    )
    return run_sensitivities(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data
    )


def certified_taylor_block(
    world: SpotWorld,
    uids: Sequence[int],
    *,
    order: TaylorOrder = TaylorOrder.SECOND_ORDER_FULL,
    binding: TaylorBaseBinding | None = None,
) -> TaylorBlock:
    """
    TAYLOR/DERIVED: production sensitizer output for `uids` (U_D-scoped, see
    `sensitivities_for`), certified at the run base and compiled on the
    valuation prefix of the run axis (F-1, D082 as amended). `01_` does this
    inline; `07_` writes it as an integration would.
    """
    sens = sensitivities_for(world, uids)
    sub = restrict_sensitivities(
        sens, uids, keep_cross=order is TaylorOrder.SECOND_ORDER_FULL
    )
    spec = build_sensitivity_book_spec(
        sensitivities=sub, required_factor_ids=world.valuation_factor_ids
    )
    declaration = derive_taylor_declaration(
        sensitivities=sub,
        order=order,
        base_binding=binding or run_base_binding(world, uids),
    )
    return TaylorBlock(
        calculator=SensitivityPnlCalculator.compile(
            dataclasses.replace(spec, taylor_declaration=declaration)
        )
    )


def methodology(
    *, full: Sequence[int] = (), derived: Sequence[int] = ()
) -> MethodologyDeclaration:
    """U = FULL + DERIVED (disjoint), TAYLOR/SUPPLIED empty (P-1 open)."""
    universe = sorted(int(u) for u in (*full, *derived))
    return MethodologyDeclaration(
        production_position_uids=np.asarray(universe, dtype=np.int64),
        full_reval_uids=np.asarray(sorted(full), dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.asarray(sorted(derived), dtype=np.int64),
    )


# =============================================================================
# 6. SCENARIO P&L ANATOMY (distribution visibility; proven == RiskRun)
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioAnatomy:
    """Every intermediate the production run builds and does not return."""

    uids: npt.NDArray[np.int64]
    shocked_levels: npt.NDArray[np.float64]  # (K, S) canonical shocked levels
    v0: npt.NDArray[np.float64]  # (M,) base settlement values
    settlement_pnl: npt.NDArray[np.float64]  # (M, S) dV after the f32 grid
    reporting: ReportingCurrencyPnl  # (M, S) dW and (S,) portfolio
    var: npt.NDArray[np.float64]
    es: npt.NDArray[np.float64]

    @property
    def num_scenarios(self) -> int:
        return int(self.shocked_levels.shape[1])

    def row(self, uid: int) -> int:
        return int(np.searchsorted(self.uids, uid))


def shocked_levels_for(
    world: SpotWorld,
    *,
    scenario_config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    batch_size: int,
) -> npt.NDArray[np.float64]:
    """
    (K, S) canonical shocked levels through the D088 building blocks, in the
    order `run_scenario_var_es` composes them: compile the authored terms over
    the full axis, lower onto it, bind (d, tau) to the world's (d, tau, h),
    stream the bound world columns, then X = sum beta Y and T^-1_kappa.
    """
    model = compile_proxy_model(
        authored_terms=world.proxy_terms,
        static_config=world.rsc,
        as_of_date=world.as_of,
        run_axis_rf_ids=world.required_factor_ids,
    )
    operator = lower_axis_operator(
        model=model,
        axis_rf_ids=world.required_factor_ids,
        axis_shift_type_ids=world.required_shift_type_id,
    )
    binding = bind_operator_to_world(operator, scenario_config, data)
    generator = build_scenario_generator(scenario_config, data=data)
    k = world.base_levels.shape[0]
    chunks: list[npt.NDArray[np.float64]] = []
    for batch in generator.iter_batches(driver_rf_ids=binding.request_driver_rf_ids,
        chart_ids=binding.request_chart_ids,
        batch_size=batch_size,
    ):
        out = np.empty((k, batch.shocks.shape[0]), dtype=np.float64, order="C")
        apply_linear_terms_to_base_levels(
            base_levels=world.base_levels,
            row_term_offsets=operator.row_term_offsets,
            term_column_uix=binding.term_column_uix,
            term_loadings=operator.term_loadings,
            target_shift_types=operator.target_shift_type_ids,
            shocks=batch.shocks,
            out=out,
        )
        chunks.append(out)
    return np.ascontiguousarray(np.hstack(chunks), dtype=np.float64)


def revalue(
    world: SpotWorld, flat_levels: npt.NDArray[np.float64]
) -> dict[int, npt.NDArray[np.float64]]:
    """uid -> (S,) settlement value at each column of a (K, S) run-axis slice.

    Valuation reads the valuation prefix only (D082 as amended)."""
    s = flat_levels.shape[1]
    matrix = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(int(world.uix_mapper.as_of_ord32)),
        uix_mapper=world.uix_mapper,
        universe_map=world.universe_map,
        flat_levels=np.ascontiguousarray(
            flat_levels[: world.num_valuation_factors], dtype=np.float64
        ),
        num_scenarios=s,
        discounting_mode=world.universe_map.discounting_mode,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=world.uix_mapper)
    rows = np.empty((world.portfolio.total_trades, s), dtype=np.float64)
    run_valuation(portfolio=world.portfolio, valuation_state=state, out=rows)
    uids = np.asarray(world.portfolio.trade_uids, dtype=np.int64)
    if np.unique(uids).shape[0] != uids.shape[0]:
        raise ValueError("a SPOT uid has more than one valuation row")
    return {int(u): rows[i] for i, u in enumerate(uids.tolist())}


def scenario_pnl_anatomy(
    world: SpotWorld,
    *,
    scenario_config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    method: MethodologyDeclaration,
    taylor_derived: TaylorBlock | None,
    reporting: ReportingSpec,
    confidence_level: float,
    batch_size: int,
) -> ScenarioAnatomy:
    """
    Recompose the production pipeline from its public building blocks.

    Same functions, same order, same dtypes as `run_scenario_var_es`:
    FULL_REVAL dV = V(x_s) - V(x_0) and TAYLOR/DERIVED dV = calculator P&L are
    both quantised through the float32 settlement grid (PnlGrid contract),
    then ONE D081 conversion, then ONE `compute_var_es`. A demo that uses this
    MUST assert its VaR/ES bitwise-equal to the RiskRun before printing any
    distribution statistic derived from it.

    The reporting plan is rebuilt INDEPENDENTLY of the run's derived route
    (D085): settlement currencies from the positions frame (the trade record),
    FX routes from the public FX authority over every FX factor on the run
    axis. Agreement with the RiskRun is therefore also a cross-check of the
    run's own derivation.
    """
    uids = method.production_position_uids
    shocked = shocked_levels_for(
        world, scenario_config=scenario_config, data=data, batch_size=batch_size
    )
    s = shocked.shape[1]
    v0 = np.array([world.v0_by_uid[int(u)] for u in uids], dtype=np.float64)
    grid = np.empty((uids.shape[0], s), dtype=np.float32)
    full = set(method.full_reval_uids.tolist())
    if full:
        vs = revalue(world, shocked)
        for i, u in enumerate(uids.tolist()):
            if u in full:
                grid[i] = vs[u] - world.v0_by_uid[u]
    if taylor_derived is not None:
        k = world.num_valuation_factors
        axis = FactorAxis.from_keys(
            pl.Series("factor_id", world.valuation_factor_ids.tolist(), dtype=pl.Int32)
        )
        result = taylor_derived.calculator.run(
            axis.levels(base_levels=world.base_levels[:k], shocked_levels=shocked[:k])
        )
        book_uids = result.pnl_matrix.trade_uids.tolist()
        for j, u in enumerate(book_uids):
            grid[int(np.searchsorted(uids, u))] = result.pnl_matrix.pnl[j]
    settlement = np.ascontiguousarray(grid, dtype=np.float64)
    frame = world.positions.set_index("uid")
    fx_rows = fx_axis_rows(world)
    plan = build_fx_conversion_plan(
        settlement_ccys=[str(frame.loc[int(u), "settlement_ccy"]) for u in uids],
        reporting_ccy=reporting.reporting_ccy,
        fx_factor_names=[world.name_of(int(world.required_factor_ids[r])) for r in fx_rows],
    )
    converted = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement,
        fx_levels=np.ascontiguousarray(shocked[fx_rows], np.float64),
        plan=plan,
        base_settlement_values=v0,
        base_fx_levels=np.ascontiguousarray(world.base_levels[fx_rows], np.float64),
    )
    groups = np.ascontiguousarray(
        np.vstack([converted.converted_pnl, converted.portfolio_pnl[None, :]]),
        dtype=np.float64,
    )
    metrics = compute_var_es(groups, confidence_level)
    return ScenarioAnatomy(
        uids=np.asarray(uids, dtype=np.int64),
        shocked_levels=shocked,
        v0=v0,
        settlement_pnl=settlement,
        reporting=converted,
        var=np.asarray(metrics.var, dtype=np.float64),
        es=np.asarray(metrics.es, dtype=np.float64),
    )


def fx_axis_rows(world: SpotWorld) -> npt.NDArray[np.intp]:
    """Rows of the run axis that carry a configured FX pair (prefix or suffix)."""
    matrix = np.asarray(world.rsc.fx_spot_matrix)
    fx_ids = np.unique(matrix[matrix != int(NO_FX_RF_REQUIRED)])
    return np.flatnonzero(np.isin(world.required_factor_ids, fx_ids)).astype(np.intp)


def anatomy_matches_run(anatomy: ScenarioAnatomy, run: RiskRun) -> bool:
    """Bitwise: same universe, same VaR, same ES."""
    return (
        np.array_equal(anatomy.uids, run.position_uids)
        and np.array_equal(anatomy.var, run.var)
        and np.array_equal(anatomy.es, run.es)
    )


# =============================================================================
# 7. SHARED FIXTURES
# =============================================================================
#
# Two kinds, as the suite plan requires:
#
#   TINY       3 factors, exactly representable (dyadic) numbers, DIFFERENCE
#              shifts. Every level, shock, value, float32 grid entry and FX
#              product is exact in binary floating point, so a hand oracle can
#              be compared with RiskQuant by EXACT equality, not a tolerance.
#   REALISTIC  ~55 factors, 3 currencies, LOG_RETURN shifts, a proxied factor,
#              ~3 years of seeded synthetic history, 120 positions with sparse,
#              unsorted UIDs. Used for model comparison, batching,
#              reproducibility and the production-style example.

TINY_AS_OF: date = date(2027, 1, 4)

#: Static universe of the tiny world. DIFFERENCE shifts: a historical shock is
#: x_t - x_{t-1}, applied as base + shock, so dyadic inputs stay exact.
TINY_FACTORS: tuple[FactorDef, ...] = (
    fx_rate(1, "EURUSD", "DIFFERENCE"),
    equity(2, "AAPL_EQ", "USD", "DIFFERENCE"),
    equity(3, "SAP_EQ", "EUR", "DIFFERENCE"),
)
TINY_CCYS: tuple[str, ...] = ("EUR", "USD")

#: Base levels. 1.25 = 5/4 is exact in binary.
TINY_LEVELS: dict[str, float] = {"EURUSD": 1.25, "AAPL_EQ": 200.0, "SAP_EQ": 120.0}

TINY_HISTORY_DATES: tuple[str, ...] = (
    "2026-12-28",
    "2026-12-29",
    "2026-12-30",
    "2026-12-31",
    "2027-01-01",
)
TINY_HISTORY_NAMES: tuple[str, ...] = ("EURUSD", "AAPL_EQ", "SAP_EQ")
#: Last row equals the base levels. 1-day differences are +-0.0625 (FX),
#: +4/+4/-6/+2 (AAPL) and +2/-2/+4/-2 (SAP).
TINY_HISTORY_VALUES: tuple[tuple[float, float, float], ...] = (
    (1.2500, 196.0, 118.0),
    (1.3125, 200.0, 120.0),
    (1.2500, 204.0, 118.0),
    (1.1875, 198.0, 122.0),
    (1.2500, 200.0, 120.0),
)

#: The three-position teaching book.
#:   101  +10 AAPL  (USD asset, USD settled)           V = q * P
#:   102   -4 SAP   (EUR asset, EUR settled)           V = q * P        [EUR]
#:   103   +8 SAP   (EUR asset, USD settled: quanto)   V = q * P * EURUSD
TINY_POSITIONS: pd.DataFrame = pd.DataFrame({
    "uid": [101, 102, 103],
    "instrument": ["SPOT", "SPOT", "SPOT"],
    "quantity": [10.0, -4.0, 8.0],
    "underlying": ["AAPL_EQ", "SAP_EQ", "SAP_EQ"],
    "settlement_ccy": ["USD", "EUR", "USD"],
})


@dataclass(frozen=True, slots=True, kw_only=True)
class TinyFixture:
    world: SpotWorld
    history: RiskFactorMatrix


def tiny_fixture(
    positions: pd.DataFrame | None = None,
    *,
    levels: Mapping[str, float] | None = None,
    history_values: Sequence[Sequence[float]] | None = None,
    reporting_ccy: str = "USD",
) -> TinyFixture:
    """The tiny dyadic world, optionally with another positions frame."""
    rsc = compile_static_config(TINY_FACTORS, TINY_CCYS)
    world = build_world(
        rsc=rsc,
        proxy_terms=author_proxy_terms(),  # every factor keeps its identity row
        positions=TINY_POSITIONS if positions is None else positions,
        levels=TINY_LEVELS if levels is None else levels,
        as_of=TINY_AS_OF,
        reporting_ccy=reporting_ccy,
    )
    history = history_matrix(
        rsc,
        dates=TINY_HISTORY_DATES,
        names=TINY_HISTORY_NAMES,
        values=TINY_HISTORY_VALUES if history_values is None else history_values,
    )
    return TinyFixture(world=world, history=history)


# ---- realistic ---------------------------------------------------------------

REALISTIC_AS_OF: date = date(2027, 1, 4)
REALISTIC_CCYS: tuple[str, ...] = ("EUR", "GBP", "USD")
_N_USD_EQ, _N_EUR_EQ, _N_GBP_EQ = 30, 12, 8


def _realistic_factors() -> tuple[FactorDef, ...]:
    factors: list[FactorDef] = [fx_rate(1, "EURUSD"), fx_rate(2, "GBPUSD")]
    nid = 10
    for prefix, ccy, n in (
        ("US", "USD", _N_USD_EQ),
        ("EU", "EUR", _N_EUR_EQ),
        ("UK", "GBP", _N_GBP_EQ),
    ):
        for i in range(n):
            factors.append(equity(nid, f"{prefix}_EQ_{i:02d}", ccy))
            nid += 1
    factors += [
        commodity(90, "BRENT_CRUDE", "USD"),
        commodity(91, "WTI_CRUDE", "USD"),
        commodity(92, "GOLD", "USD"),
    ]
    return tuple(factors)


@dataclass(frozen=True, slots=True, kw_only=True)
class RealisticFixture:
    world: SpotWorld
    history: RiskFactorMatrix
    parametric: ParametricRiskData
    num_observations: int


def realistic_fixture(
    *,
    num_positions: int = 120,
    num_observations: int = 751,
    seed: int = 20270104,
    positions: pd.DataFrame | None = None,
    reporting_ccy: str = "USD",
) -> RealisticFixture:
    """
    Production-shaped world, fully deterministic from `seed` (a LOCAL
    `np.random.default_rng`; no global random state is touched).

    * 55 factors: EURUSD, GBPUSD, 50 equities in USD/EUR/GBP, 3 commodities.
    * WTI_CRUDE carries NO history and is proxied to BRENT_CRUDE (scale 0.95).
    * History: one-factor-plus-sector correlated GBM, business days.
    * `positions` replaces the generated book (same world, same history).
    * Positions: sparse UIDs ingested in NON-ascending order; mostly settled
      in their quote currency; ~30% of EUR/GBP assets settled in USD (a
      direct quanto FX read).
    """
    rng = np.random.default_rng(seed)
    factors = _realistic_factors()
    rsc = compile_static_config(factors, REALISTIC_CCYS)
    # WTI_CRUDE has no history. Its scenario log return is modelled as 0.95 x
    # BRENT_CRUDE's log return over the world horizon: one authored Proxy V1
    # term, driver chart LOG_RETURN written explicitly (D088).
    proxies = author_proxy_terms([("WTI_CRUDE", "BRENT_CRUDE", ShiftType.LOG_RETURN, 0.95)])

    hist_names = [f.name for f in factors if f.name != "WTI_CRUDE"]
    n = len(hist_names)
    market = rng.standard_normal(num_observations - 1)
    loadings = np.array(
        [0.2 if nm.endswith("USD") else 0.6 for nm in hist_names], dtype=np.float64
    )
    vols = np.array([
        0.005 if nm.endswith("USD") else rng.uniform(0.010, 0.022) for nm in hist_names
    ])
    idio = rng.standard_normal((num_observations - 1, n))
    z = loadings[None, :] * market[:, None] + np.sqrt(1 - loadings**2)[None, :] * idio
    start = np.array([
        1.08
        if nm == "EURUSD"
        else 1.27
        if nm == "GBPUSD"
        else 82.0
        if nm == "BRENT_CRUDE"
        else 2050.0
        if nm == "GOLD"
        else rng.uniform(20.0, 400.0)
        for nm in hist_names
    ])
    paths = start[None, :] * np.exp(
        np.vstack([np.zeros((1, n)), np.cumsum(z * vols[None, :], axis=0)])
    )
    dates = pd.bdate_range(end="2026-12-31", periods=num_observations)
    history = history_matrix(rsc, dates=dates, names=hist_names, values=paths)

    levels = {nm: float(paths[-1, j]) for j, nm in enumerate(hist_names)}
    levels["WTI_CRUDE"] = round(levels["BRENT_CRUDE"] * 0.93, 2)

    underlyings = [f.name for f in factors if f.asset_class != "FX"]
    quote = {f.name: f.quote_ccy for f in factors}
    picks = rng.integers(0, len(underlyings), num_positions)
    under = [underlyings[i] for i in picks]
    settle: list[str] = []
    for u in under:
        q = quote[u]
        # Cross settlement only where the FX leg is DIRECT (EURUSD / GBPUSD
        # multiply EUR/GBP -> USD). A USD asset settled in EUR would read the
        # INVERTED leg 1/EURUSD, which is FULL_REVAL only (D084, chapter 03).
        # A direct-only book lets every chapter declare ANY generated position
        # TAYLOR/DERIVED.
        cross = rng.uniform() < 0.30
        settle.append("USD" if cross and q in ("EUR", "GBP") else q)
    uid_pool = np.sort(
        rng.choice(np.arange(10_000, 999_999), num_positions, replace=False)
    )
    order = rng.permutation(num_positions)
    quantities = np.round(rng.normal(0.0, 400.0, num_positions), 0)
    # Ingest refuses zero / dust quantities (FAIL FAST); a rounded draw of 0
    # becomes 1 unit so the generated book is always admissible.
    quantities[quantities == 0.0] = 1.0
    generated = pd.DataFrame({
        "uid": uid_pool[order].astype(np.int64).tolist(),
        "instrument": ["SPOT"] * num_positions,
        "quantity": quantities[order].tolist(),
        "underlying": [under[i] for i in order],
        "settlement_ccy": [settle[i] for i in order],
    })
    if positions is not None:
        positions = positions.copy()
    else:
        positions = generated
    world = build_world(
        rsc=rsc,
        proxy_terms=proxies,
        positions=positions,
        levels=levels,
        as_of=REALISTIC_AS_OF,
        reporting_ccy=reporting_ccy,
    )
    return RealisticFixture(
        world=world,
        history=history,
        parametric=parametric_data_from_history(history),
        num_observations=num_observations,
    )
