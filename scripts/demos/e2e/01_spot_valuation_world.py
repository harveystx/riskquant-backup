# scripts/demos/e2e/01_spot_valuation_world.py
"""
RiskQuant SPOT -- chapter 01: the base valuation world.

Makes the economic world visible before any scenario exists:

    names  --(static config)-->  rf_id      (int32 factor identity)
    rows   --(ingest)--------->  uid        (int64 economic position identity)
    rf_id  --(UixMapper)------>  uix        (int32 dense array index, per as-of)
    uid    --(compiler)------->  SpotBlock  (SoA: qty, effective_spot_price_uix)
    mapper --(universe map)--->  valuation axis (the run axis's prefix)
    levels --(market data)---->  prices, FX legs -> folded effective prices
    SpotBlock x effective prices  -->  V0 per position, settlement currency

Three identities never mix: `rf_id` names a risk factor, `uid` names an
economic position, `uix` is only an array index valid for ONE mapper. Strings
die at ingest; nothing after `compile_raw_df_into_ingest_batch` reads a name.

The book adds a fourth position to the chapter-00 book: 104 holds 1,000 EUR
as a SPOT on the FX rate EURUSD itself, settled in USD. EURUSD therefore plays
three roles in one world: an underlying (104), a pricing FX leg (the quanto
103) and a reporting FX rate (102 is EUR-settled; reporting is USD).

Run:  python scripts/demos/e2e/01_spot_valuation_world.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np
import pandas as pd

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.constants import INVALID_UIX
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import ValuationContext
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

POSITIONS: pd.DataFrame = pd.concat(
    [
        sw.TINY_POSITIONS,
        pd.DataFrame({
            "uid": [104],
            "instrument": ["SPOT"],
            "quantity": [1000.0],
            "underlying": ["EURUSD"],
            "settlement_ccy": ["USD"],
        }),
    ],
    ignore_index=True,
)
REPORTING_CCY: str = "USD"


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 01 the base valuation world")

    # ---------------------------------------------------------------- A
    sw.step("A", "Factor identity: names become rf_ids")
    rsc = sw.compile_static_config(sw.TINY_FACTORS, sw.TINY_CCYS)
    sw.para(
        """
        `compile_risk_static_config_from_dict_of_dfs` validates the static
        sheets and assigns each risk factor a stable integer rf_id. The rf_id
        is the factor's identity for the rest of the pipeline. The OIS curve
        points exist because the validator requires a discount curve per
        active currency; SPOT reads no discount factor.
        """
    )
    sw.table(
        ["name", "rf_id", "asset class", "quote ccy", "fx base", "shift type"],
        [
            (f.name, rsc.rel_name_to_rf_id[f.name], f.asset_class, f.quote_ccy, f.fx_base_ccy or "-", f.shift_type)
            for f in sw.TINY_FACTORS
        ],
        align="lrllll",
    )
    ok.check("every factor name resolves to exactly one rf_id",
             len({rsc.rel_name_to_rf_id[f.name] for f in sw.TINY_FACTORS}) == len(sw.TINY_FACTORS))
    sw.para(
        """
        FX naming follows the FX authority (`portfolio.fx`): the pair "EURUSD"
        multiplies EUR -> USD. Quote currency USD, base currency EUR.
        """
    )

    # ---------------------------------------------------------------- B
    sw.step("B", "Position identity: ingest")
    sw.table(
        ["uid", "instrument", "quantity", "underlying", "settlement_ccy"],
        POSITIONS.itertuples(index=False),
        align="rlrll",
    )
    ingest_batch = compile_raw_df_into_ingest_batch(raw_df=POSITIONS.copy(), static_config=rsc)
    sw.para(
        """
        `compile_raw_df_into_ingest_batch` validates every row and emits a
        dense SoA batch. The underlying name becomes its rf_id. When the
        underlying's quote currency differs from the settlement currency, the
        batch records the FX leg that converts one into the other
        (`fx_u2s_ids`: rf_id of the pair, `fx_u2s_inv`: 1 if inverted).
        """
    )
    sw.table(
        ["uid", "qty", "underlying_id", "fx_u2s_id", "fx_u2s_inv"],
        zip(ingest_batch.uids.tolist(), ingest_batch.qty.tolist(), ingest_batch.underlying_ids.tolist(),
            ingest_batch.fx_u2s_ids.tolist(), ingest_batch.fx_u2s_inv.tolist(), strict=True),
    )
    eurusd = rsc.rel_name_to_rf_id["EURUSD"]
    ok.check("uids survive ingest unchanged (int64)",
             ingest_batch.uids.tolist() == POSITIONS["uid"].tolist() and ingest_batch.uids.dtype == np.int64)
    ok.check("underlying names resolved to rf_ids",
             ingest_batch.underlying_ids.tolist() == [rsc.rel_name_to_rf_id[n] for n in POSITIONS["underlying"]])
    ok.check("only the quanto 103 (EUR asset, USD settled) carries an FX leg: EURUSD, direct",
             int(ingest_batch.fx_u2s_ids[2]) == eurusd and int(ingest_batch.fx_u2s_inv[2]) == 0
             and all(int(x) < 0 for i, x in enumerate(ingest_batch.fx_u2s_ids.tolist()) if i != 2),
             "negative fx_u2s_id = no FX leg required (quote ccy == settlement ccy)")

    # ---------------------------------------------------------------- C
    sw.step("C", "Array identity: the UixMapper")
    uix_mapper = build_uix_mapper([ingest_batch], as_of_ord32=int(date_to_ord32(sw.TINY_AS_OF)))
    sw.para(
        """
        `build_uix_mapper` collects every rf_id the book reads and assigns it
        a dense int32 index (uix) by position in `market_id_universe`. It
        also fixes the as-of (D067) and carries a fingerprint: arrays keyed
        by one mapper are refused by a run built on another.
        `spot_effective_pairs` is one row per distinct (price, FX leg): the
        FX column is INVALID_UIX (-999) when no conversion is needed.
        """
    )
    sw.kv([
        ("as_of_ord32", f"{uix_mapper.as_of_ord32} (days since 1970-01-01 = {sw.TINY_AS_OF})"),
        ("market_id_universe", uix_mapper.market_id_universe.tolist()),
        ("fx_effective_pairs", uix_mapper.fx_effective_pairs.tolist()),
        ("spot_effective_pairs", uix_mapper.spot_effective_pairs.tolist()),
        ("fingerprint", uix_mapper.fingerprint[:16] + "..."),
    ])
    ok.check("uix is the index into market_id_universe (sorted rf_ids)",
             uix_mapper.market_id_universe.tolist() == sorted(rsc.rel_name_to_rf_id[f.name] for f in sw.TINY_FACTORS))
    ok.check("exactly one effective pair uses an FX leg (SAP x EURUSD for 103)",
             int((uix_mapper.spot_effective_pairs[:, 1] != INVALID_UIX).sum()) == 1)

    # ---------------------------------------------------------------- D
    sw.step("D", "The compiled SpotBlock")
    portfolio = compile_portfolio_from_ingest_batch(ingest_batch=ingest_batch, uix_mapper=uix_mapper)
    block = portfolio.spot
    sw.para(
        """
        The compiler turns the batch into a struct-of-arrays block. A SPOT
        row needs exactly two fields: its quantity and the row of the folded
        effective-price table it reads. There is no per-trade object and no
        `.price()` method: pricing is one vectorised kernel over the block.
        """
    )
    sw.table(
        ["trade_uid", "qty", "effective_spot_price_uix"],
        zip(block.trade_uids.tolist(), block.qty.tolist(), block.effective_spot_price_uix.tolist(), strict=True),
    )
    ok.check("one SpotBlock row per economic uid (SPOT has no legs)",
             block.trade_uids.tolist() == POSITIONS["uid"].tolist() and portfolio.total_trades == 4)
    ok.check("block arrays are float64 qty / int32 uix, C-contiguous",
             block.qty.dtype == np.float64 and block.effective_spot_price_uix.dtype == np.int32
             and block.qty.flags.c_contiguous)

    # ---------------------------------------------------------------- E
    sw.step("E", "The valuation axis and the run axis")
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=uix_mapper
    )
    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=sw.TINY_AS_OF, levels_dict=sw.TINY_LEVELS, static_config=rsc
    )
    axis = universe_map.required_factor_ids
    base_levels = extract_levels_from_snapshot(required_ids=axis, snapshot=snapshot)
    names = {int(v): k for k, v in rsc.rel_name_to_rf_id.items()}
    sw.para(
        """
        `build_universe_mapping` lays out the factors the book's PRICERS read:
        FX spots first, then scalar nodes. This is the valuation axis. The
        run axis (D082, D085) is this axis followed by any FX factor that only
        the reporting conversion needs; `derive_run_factor_axis` returns it.
        Base levels, shocks, shocked levels and the reporting FX rows are
        indexed by the run axis, the Taylor book by its valuation prefix.
        `run_scenario_var_es` refuses any other axis row for row, because a
        permuted float row is otherwise indistinguishable from the right one.
        """
    )
    sw.table(
        ["row", "factor", "rf_id", "shift id", "base level x0"],
        [(i, names[int(r)], int(r), int(universe_map.required_shift_type_ids[i]), sw.num(base_levels[i]))
         for i, r in enumerate(axis)],
        align="rlrrr",
    )
    ok.check("axis dtypes: rf_ids int32, shift ids int8, levels float64",
             axis.dtype == np.int32 and universe_map.required_shift_type_ids.dtype == np.int8
             and base_levels.dtype == np.float64)
    ok.check("base levels are the snapshot's, in axis order",
             base_levels.tolist() == [sw.TINY_LEVELS[names[int(r)]] for r in axis])
    run_axis = derive_run_factor_axis(
        valuation_context=ValuationContext(
            static_config=rsc,
            ingest_batch=ingest_batch,
            uix_mapper=uix_mapper,
            discounting_mode=DiscountingMode.NONE,
        ),
        reporting_ccy=REPORTING_CCY,
    )
    ok.check("USD reporting adds no row: the run axis IS the valuation axis here",
             np.array_equal(run_axis.factor_ids, axis) and run_axis.num_valuation_factors == axis.shape[0],
             "the EUR-settled 102 needs EURUSD for reporting, and the pricers already read it")

    # ---------------------------------------------------------------- F
    sw.step("F", "Market data and the folded effective prices")
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=sw.TINY_AS_OF,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        snapshot=snapshot,
        num_scenarios=1,
    )
    state = assemble_valuation_state(matrix=market_data, uix_mapper=uix_mapper)
    sw.para(
        """
        `MarketDataMatrix` holds raw grids keyed by uix (prices, FX legs;
        S = 1 here because this is the base state). `assemble_valuation_state`
        folds each (price, FX leg) pair into one settlement-currency price,
        so the SPOT kernel is a single gather and multiply. The folded table
        is internal execution state; it is shown once here for understanding.
        """
    )
    eff = state.effective_spot_prices[:, 0]
    sw.table(
        ["eff uix", "pair (price rf, fx)", "effective price", "hand"],
        [
            (0, "EURUSD, none", sw.num(eff[0]), "1.25"),
            (1, "AAPL_EQ, none", sw.num(eff[1]), "200"),
            (2, "SAP_EQ, none", sw.num(eff[2]), "120"),
            (3, "SAP_EQ x EURUSD", sw.num(eff[3]), "120 x 1.25 = 150"),
        ],
        align="rlrl",
    )
    ok.check("folded effective prices equal the hand values exactly",
             eff.tolist() == [1.25, 200.0, 120.0, 150.0])

    # ---------------------------------------------------------------- G
    sw.step("G", "Base valuation V0")
    out = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    v0 = dict(zip(portfolio.trade_uids.tolist(), out[:, 0].tolist(), strict=True))
    hand = {101: 10.0 * 200.0, 102: -4.0 * 120.0, 103: 8.0 * 120.0 * 1.25, 104: 1000.0 * 1.25}
    settle = POSITIONS.set_index("uid")["settlement_ccy"].to_dict()
    q0 = {"USD": 1.0, "EUR": 1.25}  # settlement -> USD at base: EUR x EURUSD
    sw.para(
        """
        `run_valuation(portfolio, valuation_state, out)` writes into a caller
        allocated (N, S) float64 buffer: zero allocation in the hot path. V0
        is in each position's OWN settlement currency. RiskQuant never sums
        V0 across currencies. The reporting-currency value W0 = V0 * q0 below
        is computed by hand, for orientation only: production reports P&L,
        and does that conversion itself (chapter 10).
        """
    )
    sw.table(
        ["uid", "hand formula", "V0", "ccy", "q0", "W0 (USD, hand)"],
        [
            (u, f, sw.num(v0[u], 2), settle[u], sw.num(q0[settle[u]], 4), sw.num(v0[u] * q0[settle[u]], 2))
            for u, f in ((101, "10 x 200"), (102, "-4 x 120"), (103, "8 x 120 x 1.25"), (104, "1000 x 1.25"))
        ],
        align="rlrlrr",
    )
    w0_total = sum(v0[u] * q0[settle[u]] for u in v0)
    sw.kv([
        ("reporting currency", REPORTING_CCY),
        ("settlement currencies", sorted(set(settle.values()))),
        ("base portfolio value (USD, hand)", sw.num(w0_total, 2)),
    ])
    ok.check("V0 equals q * P (* FX leg) exactly for all four positions", v0 == hand)
    ok.check("the run-level helper `build_world` produces the same V0 (used from 02 on)",
             sw.tiny_fixture(POSITIONS).world.v0_by_uid == v0)

    # ---------------------------------------------------------------- H
    sw.step("H", "The three roles of one FX factor")
    sw.table(
        ["role", "position", "where it acts"],
        [
            ("underlying", "104", "the pricer: V = 1000 x EURUSD"),
            ("pricing FX leg", "103", "the pricer: V = 8 x SAP x EURUSD"),
            ("reporting FX", "102", "reporting: W = V(EUR) x EURUSD"),
        ],
        align="lll",
    )
    sw.para(
        """
        The first two roles are VALUATION: the factor is read by the pricer,
        so it appears in the position's sensitivities and in a Taylor book.
        The third role is REPORTING CONVERSION: it is applied after P&L, by
        `convert_scenario_pnl_to_reporting_currency`, never as a Taylor
        factor (D081). 102's sensitivities contain SAP_EQ only. Had no pricer
        read EURUSD, it would still enter the run axis, as a reporting-only
        row after the valuation prefix (D085, chapter 10).
        """
    )
    ok.check("EURUSD is on the axis because the book's pricing reads it",
             int(axis[0]) == eurusd)
    ok.done()


if __name__ == "__main__":
    main()
