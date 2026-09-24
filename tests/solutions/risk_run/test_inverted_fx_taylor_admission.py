# tests/solutions/risk_run/test_inverted_fx_taylor_admission.py
"""
D084: the SPOT Taylor capability boundary for inverted-FX legs.

    inverted-FX SPOT + FULL_REVAL        admitted
    inverted-FX SPOT + TAYLOR (DERIVED)  refused: sensitizer (producer) and run
                                         admission (consumer)
    direct / no-FX SPOT + TAYLOR         admitted, also beside an inverted
                                         FULL_REVAL leg (U_D-scoped sensitisation)

Book (hostile static configuration of the market-data remediation suite; the
pair EUR/GBP exists only as EURGBP, so GBP -> EUR is inverted, D060):

    201  SPOT EQ_GBP  q=-250  settle EUR   inverted  (EURGBP, invert = 1)
    202  SPOT EQ_USD  q=+100  settle EUR   direct    (USDEUR)
    203  SPOT EQ_USD  q=+40   settle USD   no FX

The Taylor book of a run is sensitised on U_D ONLY, on its own compiled
sub-portfolio (same static configuration and base levels). Sensitivities are
rf_id-keyed (D012) and the book aligns to the run axis by identity (D082).
Comparisons are exact: `float.hex()`.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REMEDIATION = Path(__file__).resolve().parents[2] / "market_data" / "remediation"
sys.path.insert(0, str(_REMEDIATION))
import hostile_fixture as hf

import riskquant.solutions.risk_run.api as api
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
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

INVERTED, DIRECT_FX, NO_FX = 201, 202, 203
BOOK = pd.DataFrame({
    "uid": [INVERTED, DIRECT_FX, NO_FX],
    "instrument": ["SPOT"] * 3,
    "quantity": [-250.0, 100.0, 40.0],
    "underlying": ["EQ_GBP", "EQ_USD", "EQ_USD"],
    "settlement_ccy": ["EUR", "EUR", "USD"],
})
_NONE = DiscountingMode.NONE
_HISTORY_DAYS = 80


@pytest.fixture(scope="module")
def world() -> SimpleNamespace:
    rsc = compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())
    as_of = int(date_to_ord32(hf.AS_OF))
    batch = compile_raw_df_into_ingest_batch(raw_df=BOOK, static_config=rsc, copy=True)
    mapper = build_uix_mapper([batch], as_of_ord32=as_of)
    context = ValuationContext(
        static_config=rsc, ingest_batch=batch, uix_mapper=mapper, discounting_mode=_NONE
    )
    layout = build_universe_mapping(
        static_config=rsc, uix_mapper=mapper, discounting_mode=_NONE
    )
    name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    fids = np.asarray(layout.required_factor_ids, dtype=np.int32)
    base = np.array([hf.LEVELS[name[int(f)]][0] for f in fids], dtype=np.float64)
    rng = np.random.default_rng(3)
    values = np.asfortranarray(
        base
        * np.exp(np.cumsum(rng.normal(0.0, 0.01, (_HISTORY_DAYS, fids.size)), axis=0))
    )
    data = RiskFactorMatrix(
        dates=np.arange(as_of - _HISTORY_DAYS + 1, as_of + 1, dtype=np.int32),
        factor_ids=fids.copy(),
        values=values,
    )
    uids = np.array(sorted(BOOK["uid"]), dtype=np.int64)
    reporting = ReportingSpec(reporting_ccy="USD")
    return SimpleNamespace(
        rsc=rsc,
        as_of=as_of,
        context=context,
        layout=layout,
        fids=fids,
        base=base,
        data=data,
        uids=uids,
        reporting=reporting,
        rf=rsc.rel_name_to_rf_id,
    )


def _sub_portfolio(w: SimpleNamespace, uids: list[int]) -> SimpleNamespace:
    """The production objects of a U_D subset, at the run's base levels."""
    batch = compile_raw_df_into_ingest_batch(
        raw_df=BOOK[BOOK["uid"].isin(uids)].reset_index(drop=True),
        static_config=w.rsc,
        copy=True,
    )
    mapper = build_uix_mapper([batch], as_of_ord32=w.as_of)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=w.rsc, uix_mapper=mapper, discounting_mode=_NONE
    )
    row = {int(f): i for i, f in enumerate(w.fids)}
    levels = w.base[[row[int(f)] for f in layout.required_factor_ids]]
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=_NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=np.ascontiguousarray(levels[:, None]),
        num_scenarios=1,
    )
    values = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(matrix=matrix, uix_mapper=mapper),
        out=values,
    )
    v0 = dict(zip(portfolio.trade_uids.tolist(), values[:, 0].tolist(), strict=True))
    return SimpleNamespace(portfolio=portfolio, mapper=mapper, matrix=matrix, v0=v0)


def _certified_block(
    w: SimpleNamespace, sens: BlockSensitivities, v0: dict[int, float]
) -> TaylorBlock:
    """Certify `sens` as TAYLOR/DERIVED (SECOND_ORDER_FULL) at the run's base."""
    book = np.unique(np.asarray(sens.position_uids, dtype=np.int64))
    binding = TaylorBaseBinding(
        as_of_ord32=w.as_of,
        base_levels_digest=base_levels_digest(factor_ids=w.fids, base_levels=w.base),
        discounting_mode=_NONE.value,
        static_config_digest=static_config_content_digest(w.rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book, values=np.array([v0[int(u)] for u in book])
        ),
    )
    spec = build_sensitivity_book_spec(sensitivities=sens, required_factor_ids=w.fids)
    spec = dataclasses.replace(
        spec,
        taylor_declaration=derive_taylor_declaration(
            sensitivities=sens,
            order=TaylorOrder.SECOND_ORDER_FULL,
            base_binding=binding,
        ),
    )
    return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


def _production_block(w: SimpleNamespace, derived: list[int]) -> TaylorBlock:
    """U_D-scoped sensitisation through the production sensitizer."""
    sub = _sub_portfolio(w, derived)
    sens = run_sensitivities(
        portfolio=sub.portfolio, uix_mapper=sub.mapper, market_data=sub.matrix
    )
    return _certified_block(w, sens, sub.v0)


def _fabricated_inverted_block(w: SimpleNamespace) -> TaylorBlock:
    """
    Caller-built Greeks for the inverted leg: the direct bilinear formulas with
    the effective rate 1/X mislabelled on the quoted factor X. It is complete
    for SECOND_ORDER_FULL and bound to the correct base, so P1-A certifies it.
    """
    q = float(BOOK.loc[BOOK["uid"] == INVERTED, "quantity"].iloc[0])
    at = {int(f): i for i, f in enumerate(w.fids)}
    spot_id, fx_id = w.rf["EQ_GBP"], w.rf["EURGBP"]
    s0, x0 = w.base[at[spot_id]], w.base[at[fx_id]]
    sens = BlockSensitivities(
        position_uids=np.array([INVERTED, INVERTED], dtype=np.int64),
        factor_rf_ids=np.array([spot_id, fx_id], dtype=np.int32),
        first_order=np.array([q / x0, q * s0], dtype=np.float64),
        second_order=np.zeros(2, dtype=np.float64),
        cross_position_uids=np.array([INVERTED], dtype=np.int64),
        cross_factor_rf_ids_i=np.array([spot_id], dtype=np.int32),
        cross_factor_rf_ids_j=np.array([fx_id], dtype=np.int32),
        cross_second_order=np.array([q], dtype=np.float64),
    )
    return _certified_block(w, sens, _sub_portfolio(w, [INVERTED]).v0)


def _inputs(
    w: SimpleNamespace, *, derived: list[int], block: TaylorBlock | None
) -> dict[str, Any]:
    u_d = np.array(sorted(derived), dtype=np.int64)
    return {
        "scenario_config": parse_scenario_model_config({
            "model": "historical",
            "horizon": 1,
        }),
        "data": w.data,
        "required_factor_ids": w.fids,
        "required_shift_type_id": np.asarray(
            w.layout.required_shift_type_ids, dtype=np.int8
        ),
        "source_factor_ids": w.fids.copy(),
        "shock_scale": np.ones(w.fids.size, dtype=np.float32),
        "base_levels": w.base,
        "methodology": MethodologyDeclaration(
            production_position_uids=w.uids,
            full_reval_uids=np.setdiff1d(w.uids, u_d),
            taylor_supplied_uids=np.empty(0, dtype=np.int64),
            taylor_derived_uids=u_d,
        ),
        "valuation_context": w.context,
        "reporting": w.reporting,
        "confidence_level": 0.99,
        "batch_size": 16,
        "taylor_derived": block,
    }


def _hexes(values: np.ndarray) -> list[str]:
    return [float(v).hex() for v in values]


@pytest.fixture(scope="module")
def all_full(world: SimpleNamespace):
    return run_scenario_var_es(**_inputs(world, derived=[], block=None))


# ---- fixture guard ---------------------------------------------------------------


def test_fixture_leg_orientations(world: SimpleNamespace) -> None:
    """201 is inverted, 202 is direct, 203 has no FX leg: read from the graph."""
    block = world.context.portfolio.spot
    mapper = world.context.uix_mapper
    fx_eff = mapper.spot_effective_pairs[block.effective_spot_price_uix, 1]
    orientation = {}
    for uid, eff in zip(block.trade_uids.tolist(), fx_eff.tolist(), strict=True):
        orientation[uid] = (
            "none" if eff == api.INVALID_UIX else int(mapper.fx_effective_pairs[eff, 1])
        )
    assert orientation == {INVERTED: 1, DIRECT_FX: 0, NO_FX: "none"}


# ---- T1: producer boundary --------------------------------------------------------


def test_t1_sensitizer_refuses_an_inverted_fx_spot_leg(world: SimpleNamespace) -> None:
    sub = _sub_portfolio(world, BOOK["uid"].tolist())
    with pytest.raises(NotImplementedError, match=r"Inverted FX-effective.*invert=1"):
        run_sensitivities(
            portfolio=sub.portfolio, uix_mapper=sub.mapper, market_data=sub.matrix
        )


# ---- T2: run admission --------------------------------------------------------------


def test_t2_fabricated_book_moves_var_without_the_guard(
    world: SimpleNamespace, all_full, monkeypatch
) -> None:
    """D059: without the admission guard the certified fabricated book runs to
    finite, plausible, WRONG numbers. The guard is the only barrier."""
    monkeypatch.setattr(api, "_require_taylor_capable_spot", lambda **kw: None)
    run = run_scenario_var_es(
        **_inputs(world, derived=[INVERTED], block=_fabricated_inverted_block(world))
    )
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()
    assert run.portfolio_var.hex() != all_full.portfolio_var.hex()


def test_t2_inverted_leg_declared_taylor_is_refused_before_identity_and_numerics(
    world: SimpleNamespace, monkeypatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the run reached identity or numerics")

    for stage in (
        "build_run_manifest",
        "_value_positions",
        "build_scenario_generator",
        "allocate_pnl_grid",
    ):
        monkeypatch.setattr(api, stage, boom)
    with pytest.raises(CompositionContractError, match=r"inverted-FX SPOT.*\[201\]"):
        run_scenario_var_es(
            **_inputs(
                world, derived=[INVERTED], block=_fabricated_inverted_block(world)
            )
        )


# ---- T3: valid mixed portfolio -------------------------------------------------------


def test_t3_inverted_full_reval_beside_direct_derived_equals_all_full(
    world: SimpleNamespace, all_full
) -> None:
    derived = [DIRECT_FX, NO_FX]
    mixed = run_scenario_var_es(
        **_inputs(world, derived=derived, block=_production_block(world, derived))
    )
    assert mixed.pnl_provenance is not None  # the Taylor block really priced
    assert _hexes(mixed.var) == _hexes(all_full.var)
    assert _hexes(mixed.es) == _hexes(all_full.es)


# ---- T4: negative control ------------------------------------------------------------


def test_t4_direct_fx_derived_position_passes_the_admission_check(
    world: SimpleNamespace, all_full
) -> None:
    run = run_scenario_var_es(
        **_inputs(
            world, derived=[DIRECT_FX], block=_production_block(world, [DIRECT_FX])
        )
    )
    assert run.pnl_provenance is not None
    assert _hexes(run.var) == _hexes(all_full.var)
    assert _hexes(run.es) == _hexes(all_full.es)
