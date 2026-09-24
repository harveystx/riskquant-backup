# scripts/demos/e2e/03_spot_sensitivity_only.py
"""
RiskQuant SPOT -- chapter 03: sensitivity-only P&L.

A caller who has sensitivities but no pricer (a desk system, a vendor feed)
can still produce scenario P&L with `riskquant.solutions.sensitivity_pnl`:

    supplied sensitivities (position_id, factor_id, first/second order)
    + factor metadata (displacement convention, quote base)
    + a scenario set bound to a FactorAxis
    ->  run_sensitivity_pnl  ->  (M, S) float32 settlement-currency P&L

What this path does:
    dV = sum_k d1_k dx_k + 1/2 sum_k d2_k dx_k^2 + sum_{i<j} c_ij dx_i dx_j
    with dx in each factor's declared displacement convention.

What this path does NOT do:
    * no revaluation: no pricer is called and no V_0 exists;
    * no bump-and-revalue: the coefficients are taken as given;
    * no reporting-currency conversion and no VaR/ES: the output is
      settlement-currency P&L only.

Production status (current contracts, not an opinion):
    Supplied sensitivities enter a production RiskRun only as TAYLOR/SUPPLIED,
    which is REFUSED while owner policy P-1 is open. A book without a
    certified TaylorDeclaration is refused by the P1-A gate. Both refusals are
    demonstrated below. The certified sensitivity route into a RiskRun is
    TAYLOR/DERIVED (chapter 04).

Run:  python scripts/demos/e2e/03_spot_sensitivity_only.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np
import polars as pl

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    TaylorBlock,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError

ABS = int(DisplacementConvention.ABSOLUTE)


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 03 sensitivity-only P&L")
    fixture = sw.tiny_fixture()
    world = fixture.world
    fx_id, aapl_id, sap_id = world.rf("EURUSD"), world.rf("AAPL_EQ"), world.rf("SAP_EQ")

    # ---------------------------------------------------------------- 1
    sw.step("1", "What the caller supplies")
    sensitivities = pl.DataFrame({
        "position_id": pl.Series([501, 502, 503, 503], dtype=pl.Int64),
        "factor_id": pl.Series([aapl_id, sap_id, sap_id, fx_id], dtype=pl.Int32),
        "first_order": [10.0, -4.0, 10.0, 960.0],
        "second_order": [0.0, 0.5, 0.0, 0.0],
    })
    cross = pl.DataFrame({
        "position_id": pl.Series([503], dtype=pl.Int64),
        "factor_id_i": pl.Series([sap_id], dtype=pl.Int32),
        "factor_id_j": pl.Series([fx_id], dtype=pl.Int32),
        "cross_second_order": [8.0],
    })
    factor_meta = pl.DataFrame({
        "factor_id": pl.Series([fx_id, aapl_id, sap_id], dtype=pl.Int32),
        "displacement_convention_id": pl.Series([ABS, ABS, ABS], dtype=pl.Int8),
        "quote_base": [1.0, 1.0, 1.0],
    })
    sw.para(
        """
        Three externally produced sensitivity sets, in settlement currency,
        differentiated with respect to the factor LEVEL (ABSOLUTE convention,
        quote base 1). RiskQuant does not know or check what produced them.
        502 carries an illustrative supplied curvature (0.5); 503 carries the
        quanto's first orders and its cross term (8 = quantity).
        """
    )
    sw.table(
        ["position", "factor", "first_order", "second_order"],
        [(r["position_id"], world.name_of(r["factor_id"]), r["first_order"], r["second_order"])
         for r in sensitivities.iter_rows(named=True)],
        align="rlrr",
    )
    sw.table(["position", "factor i", "factor j", "cross_second_order"],
             [(503, "SAP_EQ", "EURUSD", 8.0)], align="rllr")

    # ---------------------------------------------------------------- 2
    sw.step("2", "Market contract A: displacements (ScenarioShocks)")
    axis = FactorAxis.from_keys(pl.Series("factor_id", world.required_factor_ids.tolist(), dtype=pl.Int32))
    displacement = np.array(
        [[0.0625, -0.0625], [4.0, -6.0], [2.0, 4.0]], dtype=np.float32
    )  # (K, S): rows follow the axis EURUSD, AAPL_EQ, SAP_EQ
    sw.para(
        """
        The caller already holds displacements dx in each factor's convention.
        The (K, S) matrix is BOUND to a FactorAxis: the row order is part of
        the object, so a permuted matrix cannot be passed by mistake.
        """
    )
    sw.table(["axis row", "factor", "dx s=0", "dx s=1"],
             [(i, world.name_of(r), sw.num(displacement[i, 0]), sw.num(displacement[i, 1]))
              for i, r in enumerate(world.required_factor_ids)], align="rlrr")
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sensitivities,
            factor_meta=factor_meta,
            market=axis.shocks(displacement),
            cross_sensitivities=cross,
        )
    )
    d = {world.name_of(r): displacement[i].astype(np.float64) for i, r in enumerate(world.required_factor_ids)}
    oracle = {
        501: 10.0 * d["AAPL_EQ"],
        502: -4.0 * d["SAP_EQ"] + 0.5 * 0.5 * d["SAP_EQ"] ** 2,
        503: 10.0 * d["SAP_EQ"] + 960.0 * d["EURUSD"] + 8.0 * d["SAP_EQ"] * d["EURUSD"],
    }
    sw.formula(
        "501: 10 dAAPL",
        "502: -4 dSAP + 1/2 (0.5) dSAP^2",
        "503: 10 dSAP + 960 dEURUSD + 8 dSAP dEURUSD",
    )
    sw.table(["position", "RiskQuant s=0", "hand s=0", "RiskQuant s=1", "hand s=1"],
             [(u, sw.num(result.row_for(u)[0]), sw.num(oracle[u][0]), sw.num(result.row_for(u)[1]),
               sw.num(oracle[u][1])) for u in (501, 502, 503)])
    ok.check("output is (M, S) float32, rows = position_ids ascending",
             result.pnl.dtype == np.float32 and result.pnl.shape == (3, 2)
             and result.position_uids.tolist() == [501, 502, 503])
    ok.check("P&L equals the hand Taylor polynomial exactly",
             all(np.array_equal(result.row_for(u).astype(np.float64), oracle[u]) for u in oracle))
    sw.kv([
        ("provenance.market_contract", result.provenance.market_contract),
        ("provenance.num_active_factors", result.provenance.num_active_factors),
        ("provenance.num_cross_entries", result.provenance.num_cross_entries),
    ])

    # ---------------------------------------------------------------- 3
    sw.step("3", "Market contract B: levels from the scenario seam")
    scenario_config = HistoricalGeneratorConfig(model="historical", horizon=1)
    shocked = sw.shocked_levels_for(world, scenario_config=scenario_config, data=fixture.history, batch_size=4)
    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=factor_meta,
            factor_axis=axis,
            cross_sensitivities=cross,
        )
    )
    levels_result = calculator.run(axis.levels(base_levels=world.base_levels, shocked_levels=shocked))
    sw.para(
        """
        The same book can be priced against (K,) base + (K, S) shocked levels
        -- exactly the buffer the production run fills from any scenario
        model. RiskQuant forms dx = x_s - x_0 itself (ABSOLUTE convention).
        `SensitivityPnlCalculator.compile` compiles once; `run` prices many.
        """
    )
    dxl = {n: shocked[world.row(n)] - world.base_levels[world.row(n)] for n in world.axis_names()}
    oracle_l = {
        501: 10.0 * dxl["AAPL_EQ"],
        502: -4.0 * dxl["SAP_EQ"] + 0.25 * dxl["SAP_EQ"] ** 2,
        503: 10.0 * dxl["SAP_EQ"] + 960.0 * dxl["EURUSD"] + 8.0 * dxl["SAP_EQ"] * dxl["EURUSD"],
    }
    sw.table(["position", *[f"s={s}" for s in range(shocked.shape[1])]],
             [(u, *(sw.num(x, 2) for x in levels_result.row_for(u))) for u in (501, 502, 503)])
    ok.check("levels path equals the hand polynomial on x_s - x_0, exactly",
             all(np.array_equal(levels_result.row_for(u).astype(np.float64), oracle_l[u]) for u in oracle_l))
    ok.check("provenance marks the levels contract and a reused (compiled) book",
             levels_result.provenance.market_contract == "levels" and levels_result.provenance.book_reused)
    sw.para(
        """
        Compare 503 with the quanto 103 of chapter 02 (8 SAP settled in USD):
        its FULL revaluation in scenario 2 was -22. The supplied polynomial
        above, with the cross term, reproduces it exactly, because the
        bilinear payoff q * P * X has no terms beyond second order.
        """
    )
    ok.check("503 reproduces the quanto's full-revaluation dV of -22 in scenario 2",
             float(levels_result.row_for(503)[2]) == -22.0)

    # ---------------------------------------------------------------- 4
    sw.step("4", "FAIL FAST at the sensitivity boundary")
    bad = sensitivities.vstack(pl.DataFrame({
        "position_id": pl.Series([504], dtype=pl.Int64),
        "factor_id": pl.Series([999], dtype=pl.Int32),
        "first_order": [1.0],
        "second_order": [0.0],
    }))
    ok.expect_rejection(
        "a sensitivity to a factor that has no shock row is rejected",
        SensitivityPnlInputError,
        lambda: run_sensitivity_pnl(
            SensitivityPnlRequest(sensitivities=bad, factor_meta=factor_meta, market=axis.shocks(displacement))
        ),
        reason="silently contributing zero risk in every scenario would hide exposure",
        stage="request validation, before compilation or allocation",
    )

    # ---------------------------------------------------------------- 5
    sw.step("5", "Why this is not (yet) a production RiskRun path")
    run_kwargs = dict(
        scenario_config=scenario_config,
        data=fixture.history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        valuation_context=sw.valuation_context(world),
        reporting=sw.reporting_spec(world, reporting_ccy="USD"),
        confidence_level=0.75,
        batch_size=4,
    )
    supplied = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=world.uids,
        taylor_derived_uids=np.empty(0, dtype=np.int64),
    )
    ok.expect_rejection(
        "TAYLOR/SUPPLIED positions are refused by run_scenario_var_es",
        CompositionContractError,
        lambda: run_scenario_var_es(methodology=supplied, **run_kwargs),
        reason="owner policy P-1 (supplied representations) is unresolved; a "
        "supplied book is never silently priced as DERIVED or FULL_REVAL",
        stage="composition contract, before identity and numerics",
    )
    uncertified = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=pl.DataFrame({
                "position_id": pl.Series([101, 102, 103, 103], dtype=pl.Int64),
                "factor_id": pl.Series([aapl_id, sap_id, sap_id, fx_id], dtype=pl.Int32),
                "first_order": [10.0, -4.0, 10.0, 960.0],
                "second_order": [0.0, 0.0, 0.0, 0.0],
            }),
            factor_meta=factor_meta,
            factor_axis=axis,
        )
    )
    derived = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=world.uids,
    )
    ok.expect_rejection(
        "an uncertified book (no TaylorDeclaration) is refused as TAYLOR/DERIVED",
        TaylorContractError,
        lambda: run_scenario_var_es(
            methodology=derived, taylor_derived=TaylorBlock(calculator=uncertified), **run_kwargs
        ),
        reason="P1-A: order, dependency set F(p) and base binding are unproven, "
        "so the book cannot produce production P&L",
        stage="P1-A gate, after the base valuation (S = 1), before scenarios",
    )
    sw.para(
        """
        Summary. Sensitivity-only P&L is a supported standalone calculation
        in settlement currency. It is not a production VaR: a RiskRun needs a
        certified representation bound to the run's own base (chapter 04) or
        full revaluation (chapter 05).
        """
    )
    ok.done()


if __name__ == "__main__":
    main()
