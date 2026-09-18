# tests/market_data/remediation/test_history_unit_contract.py
"""
AUD-06: the history route has the D069 unit contract.

    source frame (declared unit_scale per rf)
        -> build_risk_factor_matrix_from_df / _xlsx   (normalise ONCE)
        -> RiskFactorMatrix (canonical by contract)
        -> scenario generators / get_snapshot / from_snapshot (never rescale)
        -> MarketDataMatrix producer (AUD-05 domain check)

Expected canonical values come from an independent oracle that reads the
`unit_scale` STRINGS of the config tables and applies its own scale table; it
never calls RiskQuant's normaliser. Tolerances are fixed before observation:
canonical levels rtol 1e-15, PVs rtol 1e-13, shocked levels atol 1e-15 after
narrowing the oracle shock to float32 (`ScenarioSetBatch.shocks` is float32 by
contract; an unnarrowed float64 oracle differs by ~1e-10 from representation
alone, not from units).
"""

from __future__ import annotations

import hostile_fixture as hf
import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_df,
    build_risk_factor_matrix_from_xlsx,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_df,
    build_risk_factor_level_snapshot_from_dict,
    build_risk_factor_level_snapshot_from_xlsx,
)
from riskquant.market_data.risk_factors.compile_snapshot import canonical_unit_scale
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

RTOL_LEVEL = 1e-15
RTOL_PV = 1e-13
ATOL_SHOCK = 1e-15

# (name, id, ccy, asset_class, subclass, quote_type, unit_scale, shift)
_EXTRA = [
    (
        "BOND_PX",
        29,
        "USD",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "PERCENT",
        "DIFFERENCE",
    ),
    ("SPRD_BP", 23, "EUR", "CREDIT", "CR_SINGLE_NAME", "SPREAD", "BP", "DIFFERENCE"),
    ("IDX_USD", 19, "USD", "EQUITY", "EQ_INDEX", "INDEX", "DECIMAL", "LOG_RETURN"),
]
_DATES = pd.to_datetime(["2027-01-01", "2027-01-02", "2027-01-03", "2027-01-04"])
# Source units exactly as declared by the config (PERCENT, BP, DECIMAL).
_SOURCE = {
    "BOND_PX": [101.10, 101.25, 100.90, 101.40],  # PRICE PERCENT
    "SPRD_BP": [-12.0, -11.5, -13.0, -12.5],  # SPREAD BP, negative legal
    "IDX_USD": [-3.0, -2.5, -2.8, -2.9],  # INDEX DECIMAL, negative legal
    "VOL_EQ_USD": [18.0, 18.6, 17.9, 18.3],  # VOL PERCENT
    "USD_OIS[3M]": [-0.10, -0.05, 0.02, -0.07],  # RATE PERCENT, negative legal
    "USDEUR": [0.920, 0.925, 0.918, 0.9259],  # FX_RATE DECIMAL
    "EQ_USD": [210.0, 211.0, 209.5, 212.37],  # PRICE DECIMAL
    "PWR_SPREAD": [-7.0, -7.4, -7.1, -7.25],  # PRICE DECIMAL, negative legal
}

# ------------------------------------------------------------- oracle
_ORACLE_SCALE = {"DECIMAL": 1.0, "PERCENT": 0.01, "BP": 1e-4}


@pytest.fixture(scope="module")
def dfs():
    d = hf.static_config_dfs()
    d["node_meta"] = d["node_meta"].vstack(
        pl.DataFrame(
            [
                dict(
                    node_id=i,
                    node_name=n,
                    quote_ccy=c,
                    asset_class=a,
                    asset_subclass=s,
                    quote_type=q,
                    unit_scale=u,
                    shift_type=sh,
                    fx_base_ccy=None,
                )
                for n, i, c, a, s, q, u, sh in _EXTRA
            ],
            schema=d["node_meta"].schema,
        )
    )
    return d


@pytest.fixture(scope="module")
def rsc(dfs):
    return compile_risk_static_config_from_dict_of_dfs(dfs)


@pytest.fixture(scope="module")
def oracle_unit(dfs):
    unit = {
        r["node_name"]: r["unit_scale"] for r in dfs["node_meta"].iter_rows(named=True)
    }
    joined = dfs["curve_point_meta"].join(dfs["curve_meta"], on="curve_id")
    for r in joined.iter_rows(named=True):
        unit[f"{r['curve_name']}{r['tenor_token']}"] = r["unit_scale"]
    return unit


def _canon(oracle_unit, name, raw):
    return np.asarray(raw, dtype=np.float64) * _ORACLE_SCALE[oracle_unit[name]]


def _history(rsc, source=_SOURCE):
    return build_risk_factor_matrix_from_df(
        df=pd.DataFrame(source, index=_DATES), static_config=rsc
    )


def _levels_by_name(rsc, snap):
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    return {
        names[int(i)]: float(v)
        for i, v in zip(snap.factor_ids, snap.factor_levels, strict=True)
    }


# ------------------------------------------------------------ unit contract
def test_every_quote_family_reaches_canonical_units_on_the_history_route(
    rsc, oracle_unit
):
    matrix = _history(rsc)
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    for col, rf in enumerate(matrix.factor_ids.tolist()):
        name = names[rf]
        np.testing.assert_allclose(
            matrix.values[:, col],
            _canon(oracle_unit, name, _SOURCE[name]),
            rtol=RTOL_LEVEL,
            atol=0,
        )
    got = _levels_by_name(rsc, matrix.get_snapshot(hf.AS_OF))
    assert got["VOL_EQ_USD"] == pytest.approx(0.183, rel=RTOL_LEVEL)
    assert got["BOND_PX"] == pytest.approx(1.014, rel=RTOL_LEVEL)
    assert got["SPRD_BP"] == pytest.approx(-0.00125, rel=RTOL_LEVEL)
    assert got["USD_OIS[3M]"] == pytest.approx(-0.0007, rel=RTOL_LEVEL)
    assert (
        got["IDX_USD"] == -2.9
        and got["PWR_SPREAD"] == -7.25
        and got["USDEUR"] == 0.9259
    )


def test_no_double_normalisation_downstream_of_the_edge(rsc):
    canonical = _history(rsc)
    snap = canonical.get_snapshot(hf.AS_OF)
    # canonical container -> snapshot -> container -> snapshot: values unchanged
    round_trip = RiskFactorMatrix.from_snapshot(snap).get_snapshot(hf.AS_OF)
    np.testing.assert_array_equal(round_trip.factor_levels, snap.factor_levels)
    # a directly constructed (canonical) matrix is never rescaled by get_snapshot
    direct = RiskFactorMatrix(
        dates=canonical.dates.copy(),
        factor_ids=canonical.factor_ids.copy(),
        values=np.asfortranarray(canonical.values.copy()),
    )
    np.testing.assert_array_equal(
        direct.get_snapshot(hf.AS_OF).factor_levels, snap.factor_levels
    )
    vol = _levels_by_name(rsc, direct.get_snapshot(hf.AS_OF))["VOL_EQ_USD"]
    assert vol == pytest.approx(0.183, rel=RTOL_LEVEL) and vol != pytest.approx(0.00183)


# ------------------------------------------------------------ route equivalence
def test_dict_df_xlsx_history_and_direct_routes_agree(rsc, oracle_unit, tmp_path):
    last = {n: v[-1] for n, v in _SOURCE.items()}
    via_dict = build_risk_factor_level_snapshot_from_dict(
        as_of_date=hf.AS_OF, levels_dict=last, static_config=rsc
    )
    vertical = pd.DataFrame({
        "risk_factor_name": list(last),
        "level": list(last.values()),
    })
    via_df = build_risk_factor_level_snapshot_from_df(
        as_of_date=hf.AS_OF, df=vertical, static_config=rsc
    )
    vertical.to_excel(tmp_path / "snap.xlsx", index=False)
    via_xlsx = build_risk_factor_level_snapshot_from_xlsx(
        as_of_date=hf.AS_OF, path=tmp_path / "snap.xlsx", static_config=rsc
    )
    via_history = _history(rsc).get_snapshot(hf.AS_OF)
    pd.DataFrame(_SOURCE, index=_DATES).to_excel(tmp_path / "hist.xlsx")
    via_history_xlsx = build_risk_factor_matrix_from_xlsx(
        path=tmp_path / "hist.xlsx", static_config=rsc
    ).get_snapshot(hf.AS_OF)
    oracle = {n: float(_canon(oracle_unit, n, last[n])) for n in last}
    via_direct = RiskFactorLevelSnapshot(
        as_of_date=hf.AS_OF,
        factor_ids=np.array(sorted(rsc.rel_name_to_rf_id[n] for n in last), np.int32),
        factor_levels=np.array(
            [oracle[n] for n in sorted(last, key=lambda k: rsc.rel_name_to_rf_id[k])],
            np.float64,
        ),
    )
    for route in (
        via_dict,
        via_df,
        via_xlsx,
        via_history,
        via_history_xlsx,
        via_direct,
    ):
        got = _levels_by_name(rsc, route)
        for n, expected in oracle.items():
            assert got[n] == pytest.approx(expected, rel=RTOL_LEVEL, abs=0), n


# ------------------------------------------------------------ economic effect
def _pv(rsc, snap):
    book = pd.DataFrame([
        dict(
            uid=1,
            instrument="SPOT",
            quantity=1.0e6,
            underlying="BOND_PX",
            settlement_ccy="EUR",
        ),
        dict(
            uid=2,
            instrument="SPOT",
            quantity=50.0,
            underlying="EQ_USD",
            settlement_ccy="EUR",
        ),
        dict(
            uid=3,
            instrument="SPOT",
            quantity=100.0,
            underlying="IDX_USD",
            settlement_ccy="USD",
        ),
    ])
    batch = compile_raw_df_into_ingest_batch(raw_df=book, static_config=rsc, copy=True)
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(hf.AS_OF)))
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    matrix = build_market_data_matrix_from_snapshot(
        as_of_date=hf.AS_OF, uix_mapper=mapper, universe_map=um, snapshot=snap
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, 1))
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    assert (
        portfolio.uix_fingerprint == state.uix_fingerprint == mapper.fingerprint
    )  # D074
    return dict(zip(portfolio.trade_uids.tolist(), out[:, 0].tolist(), strict=True))


def _oracle_pv(oracle_unit, source):
    fx = float(_canon(oracle_unit, "USDEUR", source["USDEUR"][-1]))
    return {
        1: 1.0e6 * float(_canon(oracle_unit, "BOND_PX", source["BOND_PX"][-1])) * fx,
        2: 50.0 * float(_canon(oracle_unit, "EQ_USD", source["EQ_USD"][-1])) * fx,
        3: 100.0 * float(_canon(oracle_unit, "IDX_USD", source["IDX_USD"][-1])),
    }


def test_history_route_pv_equals_canonical_route_and_oracle(rsc, oracle_unit):
    history = _pv(rsc, _history(rsc).get_snapshot(hf.AS_OF))
    canonical = _pv(
        rsc,
        build_risk_factor_level_snapshot_from_dict(
            as_of_date=hf.AS_OF,
            levels_dict={n: v[-1] for n, v in _SOURCE.items()},
            static_config=rsc,
        ),
    )
    oracle = _oracle_pv(oracle_unit, _SOURCE)
    for uid, expected in oracle.items():
        assert history[uid] == pytest.approx(expected, rel=RTOL_PV, abs=0)
        assert canonical[uid] == pytest.approx(expected, rel=RTOL_PV, abs=0)
    # the pre-remediation x100 would have been economically material
    assert abs(100.0 * oracle[1] - oracle[1]) > 1.0e7


# ------------------------------------------------------------ scenario arithmetic
def _scenario_levels(matrix, rf_id, shift_id, base):
    gen = HistoricalGenerator(matrix=matrix, horizon=1)
    chunks = []
    for batch in gen.iter_batches(
        required_shift_type_id=np.array([shift_id], np.int8),
        source_factor_ids=np.array([rf_id], np.int32),
        shock_scale=np.ones(1, np.float32),
        batch_size=8,
    ):
        out = np.empty((1, batch.shocks.shape[0]))
        apply_shocks_to_base_levels(base_levels=np.array([base]), batch=batch, out=out)
        chunks.append(out)
    return np.sort(np.concatenate(chunks, axis=1)[0])


def _oracle_difference_levels(base, canonical_history):
    """base + h-day canonical difference, narrowed to the float32 shock contract."""
    shocks = np.diff(canonical_history).astype(np.float32).astype(np.float64)
    return np.sort(base + shocks)


def test_difference_shocks_are_canonical_and_log_returns_are_unchanged(
    rsc, oracle_unit
):
    matrix = _history(rsc)
    bond = rsc.rel_name_to_rf_id["BOND_PX"]
    base = float(_canon(oracle_unit, "BOND_PX", _SOURCE["BOND_PX"][-1]))
    expected = _oracle_difference_levels(
        base, _canon(oracle_unit, "BOND_PX", _SOURCE["BOND_PX"])
    )
    np.testing.assert_allclose(
        _scenario_levels(matrix, bond, SHIFT_DIFFERENCE_ID, base),
        expected,
        rtol=0,
        atol=ATOL_SHOCK,
    )

    vol = rsc.rel_name_to_rf_id["VOL_EQ_USD"]
    base_vol = float(_canon(oracle_unit, "VOL_EQ_USD", _SOURCE["VOL_EQ_USD"][-1]))
    raw = np.asarray(_SOURCE["VOL_EQ_USD"])
    expected_log = np.sort(
        base_vol
        * np.exp(np.log(raw[1:] / raw[:-1]).astype(np.float32).astype(np.float64))
    )
    np.testing.assert_allclose(
        _scenario_levels(matrix, vol, SHIFT_LOG_RETURN_ID, base_vol),
        expected_log,
        rtol=1e-12,
    )


# ------------------------------------------------------------ D059 mutations
def test_mutations_move_consumed_outputs_and_the_oracle_together(rsc, oracle_unit):
    base_pv = _pv(rsc, _history(rsc).get_snapshot(hf.AS_OF))

    last = {k: list(v) for k, v in _SOURCE.items()}
    last["BOND_PX"][-1] = 99.75  # consumed by base valuation (uid 1)
    moved = _pv(rsc, _history(rsc, last).get_snapshot(hf.AS_OF))
    assert [u for u in base_pv if moved[u] != base_pv[u]] == [1]
    assert moved[1] == pytest.approx(
        _oracle_pv(oracle_unit, last)[1], rel=RTOL_PV, abs=0
    )
    assert _oracle_pv(oracle_unit, last)[1] != _oracle_pv(oracle_unit, _SOURCE)[1]

    early = {k: list(v) for k, v in _SOURCE.items()}
    early["BOND_PX"][1] = 102.00  # consumed by scenarios only, not base valuation
    bond = rsc.rel_name_to_rf_id["BOND_PX"]
    base = float(_canon(oracle_unit, "BOND_PX", _SOURCE["BOND_PX"][-1]))
    s0 = _scenario_levels(_history(rsc), bond, SHIFT_DIFFERENCE_ID, base)
    s1 = _scenario_levels(_history(rsc, early), bond, SHIFT_DIFFERENCE_ID, base)
    assert not np.array_equal(s0, s1)
    np.testing.assert_allclose(
        s1,
        _oracle_difference_levels(
            base, _canon(oracle_unit, "BOND_PX", early["BOND_PX"])
        ),
        rtol=0,
        atol=ATOL_SHOCK,
    )
    assert (
        _pv(rsc, _history(rsc, early).get_snapshot(hf.AS_OF)) == base_pv
    )  # vacuous for base PV


# ------------------------------------------------------------ fail closed / AUD-05
def test_unknown_names_and_unit_slots_fail_closed(rsc):
    with pytest.raises(KeyError, match="do not resolve"):
        build_risk_factor_matrix_from_df(
            df=pd.DataFrame({"NOT_A_FACTOR": [1.0]}, index=_DATES[:1]),
            static_config=rsc,
        )
    with pytest.raises(TypeError):
        build_risk_factor_matrix_from_df(
            df=pd.DataFrame({"EQ_USD": [1.0]}, index=_DATES[:1])
        )  # type: ignore[call-arg]
    with pytest.raises(KeyError, match="not risk factors"):
        canonical_unit_scale(rf_ids=np.array([10**9], np.int32), static_config=rsc)
    table = rsc.rf_id_to_unit_scale_factor
    unoccupied = int(np.flatnonzero(table <= 0.0)[0])
    with pytest.raises(KeyError, match="un-occupied"):
        canonical_unit_scale(rf_ids=np.array([unoccupied], np.int32), static_config=rsc)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("USDEUR", -0.9, "FX spot levels must be strictly positive"),
        ("USDEUR", 0.0, "FX spot levels must be strictly positive"),
        ("USDEUR", np.inf, "non-finite"),
    ],
)
def test_aud05_domain_still_applies_to_the_history_route(rsc, name, value, match):
    source = {k: list(v) for k, v in _SOURCE.items()}
    source[name][-1] = value
    matrix = _history(rsc, source)
    with pytest.raises(ValueError, match=match):
        _pv(rsc, matrix.get_snapshot(hf.AS_OF))


def test_legal_negative_history_levels_survive_to_valuation(rsc, oracle_unit):
    pv = _pv(rsc, _history(rsc).get_snapshot(hf.AS_OF))
    assert (
        pv[3] == pytest.approx(_oracle_pv(oracle_unit, _SOURCE)[3], rel=RTOL_PV)
        and pv[3] < 0.0
    )
