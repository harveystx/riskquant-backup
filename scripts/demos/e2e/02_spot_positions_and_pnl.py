# scripts/demos/e2e/02_spot_positions_and_pnl.py
"""
RiskQuant SPOT -- chapter 02: position-level P&L and D081, by hand.

One scenario, every position, every intermediate:

    base level x0 -> shocked level x_s -> V_0 -> V_s -> dV  (settlement ccy)
    -> q_0, q_s (settlement -> reporting rate) -> dW        (reporting ccy)

D081 (LOCKED): production reporting P&L is ordinary mark-to-market,

    dW = V_s * q_s - V_0 * q_0            (definition)
       = dV * q_s + V_0 * (q_s - q_0)     (the form RiskQuant evaluates)

The chapter computes BOTH forms by hand, shows they agree, and shows the
production RiskRun agrees with them exactly. It also evaluates the retired
D014 form `dV * q_s` to show what it gets wrong.

Oracle independence: the hand values use only `q * P * X` arithmetic on the
printed numbers. RiskQuant's scenario P&L is taken from the anatomy
recomposition, which is first proven bitwise-equal to the RiskRun VaR/ES.

Run:  python scripts/demos/e2e/02_spot_positions_and_pnl.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np
import pandas as pd

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import MethodologyDeclaration
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig

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
SCENARIO: int = 2
CONFIDENCE: float = 0.75
BATCH_SIZE: int = 4


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 02 positions and P&L (D081)")

    sw.step("1", "The book and the production run")
    fixture = sw.tiny_fixture(POSITIONS)
    world, history = fixture.world, fixture.history
    sw.table(
        ["uid", "quantity", "underlying", "settle", "value formula"],
        [
            (101, "10", "AAPL_EQ", "USD", "V = 10 x AAPL"),
            (102, "-4", "SAP_EQ", "EUR", "V = -4 x SAP            (EUR)"),
            (103, "8", "SAP_EQ", "USD", "V = 8 x SAP x EURUSD    (quanto)"),
            (104, "1000", "EURUSD", "USD", "V = 1000 x EURUSD       (EUR cash)"),
        ],
        align="rrlll",
    )
    scenario_config = HistoricalGeneratorConfig(model="historical", horizon=1)
    method = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=world.uids,
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.empty(0, dtype=np.int64),
    )
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    run = run_scenario_var_es(
        scenario_config=scenario_config,
        data=history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=sw.valuation_context(world),
        reporting=reporting,
        confidence_level=CONFIDENCE,
        batch_size=BATCH_SIZE,
    )
    anatomy = sw.scenario_pnl_anatomy(
        world,
        scenario_config=scenario_config,
        data=history,
        method=method,
        taylor_derived=None,
        reporting=reporting,
        confidence_level=CONFIDENCE,
        batch_size=BATCH_SIZE,
    )
    sw.para(
        """
        The RiskRun carries VaR/ES, not scenario P&L. The anatomy helper
        recomposes the same building blocks the run uses, so the per-scenario
        P&L becomes visible. It is trusted only after this check:
        """
    )
    ok.check("anatomy VaR/ES equal the RiskRun VaR/ES bitwise", sw.anatomy_matches_run(anatomy, run))

    sw.step("2", f"Scenario {SCENARIO}: base and shocked levels")
    x0 = {n: float(world.base_levels[world.row(n)]) for n in world.axis_names()}
    xs = {n: float(anatomy.shocked_levels[world.row(n), SCENARIO]) for n in world.axis_names()}
    hist = np.asarray(sw.TINY_HISTORY_VALUES)
    sw.para(
        f"""
        Historical scenario {SCENARIO} is the 1-day move from
        {sw.TINY_HISTORY_DATES[SCENARIO]} to {sw.TINY_HISTORY_DATES[SCENARIO + 1]},
        applied to TODAY's base levels (DIFFERENCE shift: x_s = x0 + move).
        """
    )
    sw.table(
        ["factor", "x0", "historical move", "x_s"],
        [(n, sw.num(x0[n]), sw.num(hist[SCENARIO + 1, j] - hist[SCENARIO, j]), sw.num(xs[n]))
         for j, n in enumerate(sw.TINY_HISTORY_NAMES)],
    )
    ok.check("x_s = x0 + historical move, exactly, for every factor",
             all(xs[n] == x0[n] + (hist[SCENARIO + 1, j] - hist[SCENARIO, j])
                 for j, n in enumerate(sw.TINY_HISTORY_NAMES)))

    sw.step("3", "Settlement-currency value and dV")
    a, s, f = xs["AAPL_EQ"], xs["SAP_EQ"], xs["EURUSD"]
    a0, s0, f0 = x0["AAPL_EQ"], x0["SAP_EQ"], x0["EURUSD"]
    hand_v0 = {101: 10 * a0, 102: -4 * s0, 103: 8 * s0 * f0, 104: 1000 * f0}
    hand_vs = {101: 10 * a, 102: -4 * s, 103: 8 * s * f, 104: 1000 * f}
    settle = POSITIONS.set_index("uid")["settlement_ccy"].to_dict()
    sw.table(
        ["uid", "ccy", "V_0", "V_s", "dV = V_s - V_0", "RiskQuant dV"],
        [
            (u, settle[u], sw.num(hand_v0[u], 2), sw.num(hand_vs[u], 2), sw.num(hand_vs[u] - hand_v0[u], 2),
             sw.num(anatomy.settlement_pnl[anatomy.row(u), SCENARIO], 2))
            for u in (101, 102, 103, 104)
        ],
        align="rlrrrr",
    )
    ok.check("RiskQuant dV equals the hand dV exactly for every position",
             all(anatomy.settlement_pnl[anatomy.row(u), SCENARIO] == hand_vs[u] - hand_v0[u] for u in hand_v0))
    sw.para(
        """
        Note 103: the quanto's dV mixes both moves. SAP rose (+4) and EURUSD
        fell (-0.0625): 8 x 124 x 1.1875 - 8 x 120 x 1.25 = -22.
        """
    )

    sw.step("4", "Reporting conversion: both D081 forms, by hand")
    q0 = {"USD": 1.0, "EUR": f0}
    qs = {"USD": 1.0, "EUR": f}
    rows = []
    dw_def: dict[int, float] = {}
    for u in (101, 102, 103, 104):
        c = settle[u]
        dv = hand_vs[u] - hand_v0[u]
        definition = hand_vs[u] * qs[c] - hand_v0[u] * q0[c]
        term1, term2 = dv * qs[c], hand_v0[u] * (qs[c] - q0[c])
        dw_def[u] = definition
        rows.append((u, c, sw.num(q0[c]), sw.num(qs[c]), sw.num(term1, 2), sw.num(term2, 2),
                     sw.num(definition, 2), sw.num(anatomy.reporting.converted_pnl[anatomy.row(u), SCENARIO], 2)))
        ok.check(f"uid {u}: V_s q_s - V_0 q_0 == dV q_s + V_0 (q_s - q_0)", definition == term1 + term2)
    sw.table(["uid", "ccy", "q_0", "q_s", "dV*q_s", "V0*(qs-q0)", "dW", "RiskQuant"], rows, align="rlrrrrrr")
    ok.check("RiskQuant dW equals the hand D081 value exactly for every position",
             all(anatomy.reporting.converted_pnl[anatomy.row(u), SCENARIO] == dw_def[u] for u in dw_def))
    sw.para(
        """
        Position 102 is the lesson. In EUR it LOSES 16 (short SAP, SAP up
        4). In USD it GAINS 11. The short EUR position was worth -600 USD at
        the base FX rate; EURUSD fell, so the same liability is now worth
        less in USD: +30. The retired D014 convention dV * q_s keeps only the
        first term (-19) and reports the wrong sign.
        """
    )
    d014 = (hand_vs[102] - hand_v0[102]) * f
    ok.check("the retired D014 form dV*q_s differs from D081 for 102 (sign flips)",
             d014 == -19.0 and dw_def[102] == 11.0 and np.sign(d014) != np.sign(dw_def[102]))
    sw.para(
        """
        Positions settled in the reporting currency have q = 1 exactly, so
        D081 reduces to dW = dV: the translation term is identically zero.
        """
    )
    ok.check("USD-settled rows: translation term is exactly zero, dW == dV",
             all(anatomy.reporting.converted_pnl[anatomy.row(u)].tolist()
                 == anatomy.settlement_pnl[anatomy.row(u)].tolist() for u in (101, 103, 104)))

    sw.step("5", "From scenario P&L to VaR")
    total = anatomy.reporting.portfolio_pnl
    sw.table(["scenario", *[str(s_) for s_ in range(anatomy.num_scenarios)]],
             [("portfolio dW (USD)", *(sw.num(x, 2) for x in total))])
    worst = int(np.argmin(total))
    sw.kv([
        ("confidence", CONFIDENCE),
        ("tail mass m = (1-C) * S", (1 - CONFIDENCE) * anatomy.num_scenarios),
        ("portfolio VaR (RiskRun)", f"{sw.num(run.portfolio_var, 2)} USD"),
        ("worst scenario (hand)", f"s={worst}: {sw.num(total[worst], 2)} USD"),
    ])
    ok.check("portfolio VaR is the worst scenario's portfolio dW (m = 1)",
             run.portfolio_var == float(total.min()))
    ok.check("portfolio dW is the sum of the hand D081 rows in every scenario",
             np.array_equal(total, anatomy.reporting.converted_pnl.sum(axis=0)))
    ok.done()


if __name__ == "__main__":
    main()
