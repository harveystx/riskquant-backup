# tests/market_data/remediation/test_production_path_certification.py
"""
REAL PRODUCTION PRODUCER PATH certification against the independent oracle.

The expected value of every consumed state cell and every PV comes from
`md_oracle` (economic names and quotes only). The observed value comes from the
production chain in `hostile_fixture.run_spot_book`. Nothing is hand-built.

Mutation discipline (D059): every mutation first proves it moved the consumed
production output, then proves the oracle rejects the moved output.
"""

from __future__ import annotations

import gc

import hostile_fixture as hf
import md_oracle as oracle
import numpy as np
import pandas as pd
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.dates import date_to_ord32

# Tolerance fixed BEFORE observation: the only arithmetic between quote and PV
# is one gather, at most one reciprocal and two multiplies in float64.
RTOL = 1e-13


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())


def _market(levels):
    return oracle.EconomicMarket(
        levels=levels,
        factor_ccy=hf.FACTOR_CCY,
        fx_quotes={pair: levels[name] for pair, name in hf.FX_QUOTES.items()},
    )


def _expected(book, levels, n):
    mkt = _market(levels)
    esp, pv = {}, {}
    for r in book.itertuples(index=False):
        esp[r.uid] = oracle.effective_spot(mkt, r.underlying, r.settlement_ccy, n)
        pv[r.uid] = oracle.spot_pv(
            mkt, qty=r.quantity, underlying=r.underlying, settle=r.settlement_ccy, n=n
        )
    return esp, pv


def _observed(portfolio, state, out):
    block = portfolio.spot
    esp = {
        int(u): state.effective_spot_prices[k]
        for u, k in zip(
            block.trade_uids.tolist(),
            block.effective_spot_price_uix.tolist(),
            strict=True,
        )
    }
    pv = {int(u): out[i] for i, u in enumerate(portfolio.trade_uids.tolist())}
    return esp, pv


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def test_hostile_universe_really_interleaves_fx(rsc):
    _, m, *_ = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    kinds = [
        "FX" if names[int(i)] in hf.FX_QUOTES.values() else "PX"
        for i in m.market_id_universe
    ]
    assert kinds[0] == "PX" and "PX" in kinds[kinds.index("FX") :], kinds
    assert set(m.fx_effective_pairs[:, 1].tolist()) == {0, 1}


def test_production_state_and_pv_reconcile_to_oracle(rsc):
    p, m, matrix, state, out = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    exp_esp, exp_pv = _expected(hf.SPOT_BOOK, hf.LEVELS, hf.S)
    obs_esp, obs_pv = _observed(p, state, out)
    assert set(obs_pv) == set(exp_pv)
    for uid in exp_pv:
        assert _rel(obs_esp[uid], exp_esp[uid]) <= RTOL, uid
        assert _rel(obs_pv[uid], exp_pv[uid]) <= RTOL, uid
    assert (
        state.effective_spot_prices.dtype == np.float64
        and state.effective_spot_prices.flags.c_contiguous
    )
    assert matrix.fx_spots_effective.shape == (m.fx_effective_pairs.shape[0], hf.S)
    assert np.all(np.isfinite(state.effective_spot_prices))


def test_production_path_is_bitwise_deterministic(rsc):
    a = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )[4]
    b = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )[4]
    np.testing.assert_array_equal(a, b)


def test_every_fx_effective_row_is_consumed(rsc):
    _, m, *_ = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    referenced = set(
        m.spot_effective_pairs[:, 1][m.spot_effective_pairs[:, 1] >= 0].tolist()
    )
    assert referenced == set(range(m.fx_effective_pairs.shape[0]))


def _perturb(levels, name, fn):
    out = {k: v.copy() for k, v in levels.items()}
    out[name] = fn(out[name])
    return out


MUTATIONS = {
    "fx_level_USDEUR": (
        lambda lv: _perturb(lv, "USDEUR", lambda x: x * 1.03),
        [101, 104],
    ),
    "fx_level_EURGBP": (
        lambda lv: _perturb(lv, "EURGBP", lambda x: x * 0.97),
        [102, 105],
    ),
    "fx_direction_USDCHF_inverted": (
        lambda lv: _perturb(lv, "USDCHF", lambda x: 1.0 / x),
        [103],
    ),
    "fx_row_substitution_EURGBP_as_USDEUR": (
        lambda lv: _perturb(lv, "USDEUR", lambda x: lv["EURGBP"].copy()),
        [101, 104],
    ),
    "price_level_EQ_EUR": (lambda lv: _perturb(lv, "EQ_EUR", lambda x: x + 0.5), [105]),
    "scenario_axis_reversed_EQ_GBP": (
        lambda lv: _perturb(lv, "EQ_GBP", lambda x: x[::-1].copy()),
        [102],
    ),
    "negative_spread_sign": (
        lambda lv: _perturb(lv, "PWR_SPREAD", lambda x: -x),
        [106],
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_is_consumed_and_rejected_by_oracle(rsc, name):
    mutate, expected_moved = MUTATIONS[name]
    base = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    mutated_levels = mutate(hf.LEVELS)
    mut = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=mutated_levels, num_scenarios=hf.S
    )
    uids = base[0].trade_uids.tolist()
    moved = sorted(
        u for i, u in enumerate(uids) if not np.array_equal(base[4][i], mut[4][i])
    )
    assert moved == sorted(expected_moved), (
        f"D059: mutation must touch exactly its consumed rows; moved={moved}"
    )
    _, exp_pv = _expected(hf.SPOT_BOOK, hf.LEVELS, hf.S)
    for i, u in enumerate(uids):
        if u in moved:
            assert _rel(mut[4][i], exp_pv[u]) > 1e-6, (name, u)
        else:
            assert _rel(mut[4][i], exp_pv[u]) <= RTOL


def test_control_mutation_of_unreferenced_factor_is_vacuous(rsc):
    base = hf.run_spot_book(
        static_config=rsc,
        book=hf.SPOT_BOOK.iloc[:2],
        levels=hf.LEVELS,
        num_scenarios=hf.S,
    )
    mut = hf.run_spot_book(
        static_config=rsc,
        book=hf.SPOT_BOOK.iloc[:2],
        levels=_perturb(hf.LEVELS, "USDCHF", lambda x: x * 9.0),
        num_scenarios=hf.S,
    )
    np.testing.assert_array_equal(base[4], mut[4])


# ---------------------------------------------------------------- memory poisoning (MD-01b / NEW-01)
POISON = (7.77e11, -4.242e9, 123456.789)


@pytest.mark.parametrize("poison", POISON)
@pytest.mark.parametrize("s", [1, 2000])
def test_fx_rate_underlying_is_owned_not_inherited_from_allocator(rsc, poison, s):
    book = hf.SPOT_BOOK[hf.SPOT_BOOK.uid.isin([102, 104])].reset_index(drop=True)
    levels = {k: np.resize(v, s) for k, v in hf.LEVELS.items()}
    junk = [np.full((8, s), poison) for _ in range(32)]
    del junk
    gc.collect()
    p, _, _, _, out = hf.run_spot_book(
        static_config=rsc, book=book, levels=levels, num_scenarios=s
    )
    _, exp_pv = _expected(book, levels, s)
    for i, uid in enumerate(p.trade_uids.tolist()):
        assert _rel(out[i], exp_pv[uid]) <= RTOL
        assert not np.any(
            np.isclose(out[i], poison * book.set_index("uid").loc[uid].quantity)
        )


@pytest.mark.parametrize("poison", POISON)
def test_container_underlying_cannot_reach_the_fold_even_when_ingest_is_bypassed(
    rsc, poison
):
    import dataclasses

    batch = compile_raw_df_into_ingest_batch(
        raw_df=hf.SPOT_BOOK.iloc[[0]], static_config=rsc, copy=True
    )
    und = batch.underlying_ids.copy()
    und[0] = rsc.container_name_to_id["USD_OIS"]
    bypass = dataclasses.replace(batch, underlying_ids=np.ascontiguousarray(und))
    mapper = build_uix_mapper([bypass], as_of_ord32=int(date_to_ord32(hf.AS_OF)))
    compile_portfolio_from_ingest_batch(ingest_batch=bypass, uix_mapper=mapper)
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    junk = [np.full((4, 1), poison) for _ in range(32)]
    del junk
    with pytest.raises(ValueError, match="D072"):
        build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            flat_levels=np.full((um.required_factor_ids.shape[0], 1), 1.0),
            num_scenarios=1,
        )


def test_container_underlying_is_rejected_at_ingest(rsc):
    from riskquant.exceptions import PrettyKeyError

    with pytest.raises(PrettyKeyError):
        compile_raw_df_into_ingest_batch(
            raw_df=pd.DataFrame([
                dict(
                    uid=1,
                    instrument="SPOT",
                    quantity=1.0,
                    underlying="USD_OIS",
                    settlement_ccy="USD",
                )
            ]),
            static_config=rsc,
        )
