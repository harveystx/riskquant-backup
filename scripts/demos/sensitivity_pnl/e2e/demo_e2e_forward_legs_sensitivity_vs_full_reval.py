# scripts/demos/sensitivity_pnl/e2e/demo_e2e_forward_legs_sensitivity_vs_full_reval.py
"""
Forward legs, by hand: sensitivity PnL against full revaluation, and VaR.

    MtM = V * P * FX_float  -  V * K * FX_fixed

    V         signed volume (the position: positive = long, negative = short)
    P         underlying price
    FX_float  FX rate converting the underlying's currency into settlement currency
    K         the contracted forward strike (fixed at trade time, never a risk factor)
    FX_fixed  FX rate converting the strike's currency into settlement currency

Every number in this file is chosen so a reader can reproduce it with a pen. The
market has three risk factors. Every position is priced at exactly ten scenarios.
Every scenario's shock is a round number. The point is not the result — it is
that you can check the result yourself, term by term, and see WHY sensitivity
PnL agrees with full revaluation to the last decimal that float32 allows.

Why this payoff is the right one to test with
----------------------------------------------
Expand `Mtm` around today's market. Write `dP`, `dFf`, `dFk` for the changes in
`P`, `FX_float`, `FX_fixed`:

    dMtM/dP        = V * FX_float          dMtM/dFX_float = V * P
    dMtM/dFX_fixed = -V * K
    d2MtM/dP2 = d2MtM/dFX_float2 = d2MtM/dFX_fixed2 = 0        (no curvature in any one factor)
    d2MtM/dP.dFX_float = V                                     (the ONE cross term)
    d2MtM/dP.dFX_fixed = d2MtM/dFX_float.dFX_fixed = 0          (these never multiply together)

Every term above is a CONSTANT — none of them depends on P, FX_float or
FX_fixed. That means the second-order Taylor expansion is not an approximation
of this payoff, it is an IDENTITY:

    (P0+dP)(Ff0+dFf) = P0.Ff0 + Ff0.dP + P0.dFf + dP.dFf

is exact algebra, true for a $1 shock or a $1,000,000 shock alike, with nothing
left over. So every full-reval number below and every sensitivity-PnL number
below must agree to the last bit float32 allows — and Section 6 checks that
they do, cell by cell, rather than asserting it.

The five forwards, easiest to hardest
--------------------------------------
    A  FWD_NO_FX          no FX anywhere:      MtM = V.(P - K)
    B  FWD_FLOAT_FX       FX on the float leg: MtM = V.P.FX_float - V.K
    C  FWD_FIXED_FX       FX on the fixed leg: MtM = V.P - V.K.FX_fixed
    D  FWD_BOTH_FX        FX on both legs:     MtM = V.P.FX_float - V.K.FX_fixed
    E  FWD_BOTH_FX_SHORT  D's mirror, V negated — proves the sign machinery
    F  FWD_ZERO_VOLUME    V=0, so EVERY sensitivity is exactly zero and the
                           position carries no rows in the sensitivity feed at
                           all — the "declared position, no exposure" case.

B is the interesting contrast with C: B has a cross term (P and FX_float
multiply together in one term) and C does not (P and FX_fixed sit in separate
additive terms). Two positions that both "have FX," with different risk.

Ten scenarios, one market state, two pricing methods
------------------------------------------------------
Ten shocked-level scenarios are built once, fed to BOTH the sensitivity engine
(as `ScenarioLevels`, base + shocked) and to a hand-written full-revaluation
function. Same numbers, two methods, so any disagreement is a defect, not a
different draw of the market.

VaR at 90% with ten scenarios
-------------------------------
`compute_var_es`'s tail count is `max(1, int((1 - confidence) * n))`. At
confidence=0.90 and n=10: `max(1, int(0.10 * 10))` = `max(1, 1)` = `1`. One
scenario in the tail means VaR is simply the worst of the ten P&Ls, and ES
(the mean of a one-element tail) equals VaR exactly. Section 7 shows the
arithmetic and then calls the real production function to confirm it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivity_pnl.conventions import QUOTE_BASE_NATURAL
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)

_WIDTH = 158


def _rule(char: str = "-") -> None:
    print(char * _WIDTH)


# =========================================================================
# 1. THE MARKET — three risk factors, base levels, ten scenarios
# =========================================================================

FACTOR_P, FACTOR_FX_FLOAT, FACTOR_FX_FIXED = "P", "FX_FLOAT", "FX_FIXED"
FACTOR_KEYS = [FACTOR_P, FACTOR_FX_FLOAT, FACTOR_FX_FIXED]

BASE_P = 100.0
BASE_FX_FLOAT = 2.00
BASE_FX_FIXED = 1.50
BASE_LEVELS = np.array([BASE_P, BASE_FX_FLOAT, BASE_FX_FIXED], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    d_p: float
    d_fx_float: float
    d_fx_fixed: float


#: Ten hand-picked shocks. Round numbers, and S9 is deliberately the one
#: scenario where all three moves hurt a long position at once.
SCENARIOS: list[Scenario] = [
    Scenario("S0 FLAT", 0.0, 0.0, 0.0),
    Scenario("S1 P_UP", +5.0, 0.0, 0.0),
    Scenario("S2 P_DOWN", -5.0, 0.0, 0.0),
    Scenario("S3 FXFLOAT_UP", 0.0, +0.10, 0.0),
    Scenario("S4 FXFLOAT_DOWN", 0.0, -0.10, 0.0),
    Scenario("S5 FXFIXED_UP", 0.0, 0.0, +0.10),
    Scenario("S6 FXFIXED_DOWN", 0.0, 0.0, -0.10),
    Scenario("S7 RALLY_TOGETHER", +10.0, +0.20, 0.0),
    Scenario("S8 DIVERGENT", -10.0, +0.20, 0.0),
    Scenario("S9 CRISIS", -15.0, -0.30, +0.20),
]
NUM_SCENARIOS = len(SCENARIOS)

SHOCKED_LEVELS = np.ascontiguousarray(
    np.array(
        [
            [BASE_P + s.d_p for s in SCENARIOS],
            [BASE_FX_FLOAT + s.d_fx_float for s in SCENARIOS],
            [BASE_FX_FIXED + s.d_fx_fixed for s in SCENARIOS],
        ],
        dtype=np.float64,
    )
)


# =========================================================================
# 2. THE FORWARD BOOK — five real positions, one zero-volume position
# =========================================================================


@dataclass(frozen=True, slots=True)
class Forward:
    name: str
    uid: int
    volume: float
    strike: float
    uses_fx_float: bool
    uses_fx_fixed: bool

    def full_reval_mtm(
        self,
        *,
        p: npt.NDArray[np.float64],
        fx_float: npt.NDArray[np.float64],
        fx_fixed: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """`MtM = V.P.FX_float - V.K.FX_fixed`, with the absent legs pinned at 1."""
        effective_fx_float = fx_float if self.uses_fx_float else np.ones_like(p)
        effective_fx_fixed = fx_fixed if self.uses_fx_fixed else np.ones_like(p)
        return (
            self.volume * p * effective_fx_float
            - self.volume * self.strike * effective_fx_fixed
        )

    def full_reval_mtm_formula_str(
        self,
        *,
        p: float | int,
        fx_float: float | int,
        fx_fixed: float | int,
    ) -> str:

        effective_fx_float = fx_float if self.uses_fx_float else 1.0
        effective_fx_fixed = fx_fixed if self.uses_fx_fixed else 1.0

        float_leg = self.volume * p * effective_fx_float
        fixed_leg = self.volume * self.strike * effective_fx_fixed

        return (
            f"({self.volume:>3,.0f} x {p:>6,.2f} x {effective_fx_float:>4,.2f})"
            " - "
            f"({self.volume:>3,.0f} x {self.strike:>5,.2f} x {effective_fx_fixed:>4,.2f})"
            " = "
            f"({float_leg:>9,.2f}) - ({fixed_leg:>9,.2f})"
        )


FORWARDS: list[Forward] = [
    Forward(
        "A FWD_NO_FX",
        1,
        volume=10.0,
        strike=90.0,
        uses_fx_float=False,
        uses_fx_fixed=False,
    ),
    Forward(
        "B FWD_FLOAT_FX",
        2,
        volume=10.0,
        strike=90.0,
        uses_fx_float=True,
        uses_fx_fixed=False,
    ),
    Forward(
        "C FWD_FIXED_FX",
        3,
        volume=10.0,
        strike=90.0,
        uses_fx_float=False,
        uses_fx_fixed=True,
    ),
    Forward(
        "D FWD_BOTH_FX",
        4,
        volume=10.0,
        strike=90.0,
        uses_fx_float=True,
        uses_fx_fixed=True,
    ),
    Forward(
        "E FWD_BOTH_FX_SHORT",
        5,
        volume=-10.0,
        strike=90.0,
        uses_fx_float=True,
        uses_fx_fixed=True,
    ),
    Forward(
        "F FWD_ZERO_VOLUME",
        6,
        volume=0.0,
        strike=90.0,
        uses_fx_float=True,
        uses_fx_fixed=True,
    ),
]
POSITION_UIDS = [f.uid for f in FORWARDS]
NAME_BY_UID = {f.uid: f.name for f in FORWARDS}


def full_reval_pnl_grid() -> npt.NDArray[np.float64]:
    """`(num_positions, num_scenarios)` full-reval PnL, base levels vs each scenario."""
    p, fx_float, fx_fixed = SHOCKED_LEVELS
    base_p = np.full(NUM_SCENARIOS, BASE_P)
    base_fx_float = np.full(NUM_SCENARIOS, BASE_FX_FLOAT)
    base_fx_fixed = np.full(NUM_SCENARIOS, BASE_FX_FIXED)
    grid = np.empty((len(FORWARDS), NUM_SCENARIOS), dtype=np.float64)
    for row, forward in enumerate(FORWARDS):
        shocked = forward.full_reval_mtm(p=p, fx_float=fx_float, fx_fixed=fx_fixed)
        base = forward.full_reval_mtm(
            p=base_p, fx_float=base_fx_float, fx_fixed=base_fx_fixed
        )
        grid[row] = shocked - base
    return grid


# =========================================================================
# 3. THE SENSITIVITY BOOK — read straight off the partial derivatives above
# =========================================================================


def build_book() -> SensitivityPnlCalculator:
    """
    ABSOLUTE convention, quote_base = 1.0, for all three factors — so every
    coefficient below is exactly the partial derivative computed by hand, with
    no unit conversion to undo when checking it.
    """
    diagonal_rows: list[tuple[int, str, float, float]] = []
    cross_rows: list[tuple[int, str, str, float]] = []

    for forward in FORWARDS:
        if forward.volume == 0.0:
            continue  # F: zero volume -> zero sensitivity to everything -> no rows

        delta_p = forward.volume * (BASE_FX_FLOAT if forward.uses_fx_float else 1.0)
        diagonal_rows.append((forward.uid, FACTOR_P, delta_p, 0.0))

        if forward.uses_fx_float:
            delta_fx_float = forward.volume * BASE_P
            diagonal_rows.append((forward.uid, FACTOR_FX_FLOAT, delta_fx_float, 0.0))
            cross_rows.append((forward.uid, FACTOR_P, FACTOR_FX_FLOAT, forward.volume))

        if forward.uses_fx_fixed:
            delta_fx_fixed = -forward.volume * forward.strike
            diagonal_rows.append((forward.uid, FACTOR_FX_FIXED, delta_fx_fixed, 0.0))

    sensitivities = pl.DataFrame(
        diagonal_rows,
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross_sensitivities = pl.DataFrame(
        cross_rows,
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.String,
            "factor_id_j": pl.String,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    factor_meta = pl.DataFrame(
        {
            "factor_id": FACTOR_KEYS,
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 3,
            "quote_base": [QUOTE_BASE_NATURAL] * 3,
        }
    )
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=factor_meta,
            factor_axis=axis,
            cross_sensitivities=cross_sensitivities,
            position_uids=POSITION_UIDS,
        )
    )


def printable_sensitivities() -> dict[int, dict[str, float]]:
    """The same coefficients the book was built from, for the printed table."""
    out: dict[int, dict[str, float]] = {}
    for forward in FORWARDS:
        entry: dict[str, float] = {}
        if forward.volume != 0.0:
            entry["delta_P"] = forward.volume * (
                BASE_FX_FLOAT if forward.uses_fx_float else 1.0
            )
            if forward.uses_fx_float:
                entry["delta_FXf"] = forward.volume * BASE_P
                entry["cross_P_FXf"] = forward.volume
            if forward.uses_fx_fixed:
                entry["delta_FXk"] = -forward.volume * forward.strike
        out[forward.uid] = entry
    return out


# =========================================================================
# 4. VaR — positive-loss convention, matching earlier demos
# =========================================================================


def _no_negative_zero(value: float) -> float:
    """`-0.0` is a real IEEE value (e.g. `-10 * 0.0`) and prints as `-0.0`, which
    reads as a sign error in a table meant to be checked by eye. `+ 0.0` is the
    standard normalisation: IEEE addition defines `-0.0 + 0.0 == 0.0`."""
    return value + 0.0


def var_es(
    pnl: npt.NDArray[np.float64], confidence_level: float
) -> tuple[float, float]:
    matrix = np.ascontiguousarray(pnl.reshape(1, -1), dtype=np.float64)
    result = compute_var_es(matrix, confidence_level=confidence_level)
    return -float(result.var[0]), -float(result.es[0])


# =========================================================================
# 5. REPORT
# =========================================================================


def run_demo() -> None:
    print("=" * _WIDTH)
    print(f"{'FORWARD LEGS: SENSITIVITY PnL vs FULL REVALUATION':^{_WIDTH}}")
    print("=" * _WIDTH)

    print(
        "\nMtM = V . P . FX_float  -  V . K . FX_fixed"
        "\n"
        "- V            signed volume \n"
        "- P            underlying price \n"
        "- K            contracted strike          \n"
        "- FX_float     underlying ccy -> settlement ccy\n"
        "- FX_fixed     strike ccy     -> settlement ccy"
    )
    print(
        f"\nBase market:"
        "\n"
        f"- P = {BASE_P:g}\n"
        f"- FX_FLOAT = {BASE_FX_FLOAT:g}\n"
        f"- FX_FIXED = {BASE_FX_FIXED:g}"
    )

    # ---- Section 1: the book ------------------------------------------
    print("\n" + "=" * _WIDTH)
    print("SECTION 1 — THE FORWARD BOOK")
    print("=" * _WIDTH)
    _rule()
    print(
        f"{'position':>22} | {'V':>6} | {'K':>5} | {'FXf?':>5} | {'FXk?':>5} | {'MtM today':>12}"
    )
    _rule()
    base_row = np.array([[BASE_P], [BASE_FX_FLOAT], [BASE_FX_FIXED]])
    for forward in FORWARDS:
        mtm_today = float(
            forward.full_reval_mtm(
                p=base_row[0], fx_float=base_row[1], fx_fixed=base_row[2]
            )[0]
        )
        mtm_today_str = forward.full_reval_mtm_formula_str(
            p=BASE_P,
            fx_float=BASE_FX_FLOAT,
            fx_fixed=BASE_FX_FIXED,
        )
        print(
            f"{forward.name:>22} | {forward.volume:>6.0f} | {forward.strike:>5.0f} | "
            f"{'yes' if forward.uses_fx_float else 'no':>5} | "
            f"{'yes' if forward.uses_fx_fixed else 'no':>5} | {mtm_today_str} = {mtm_today:>8,.2f}"
        )
    print(
        "\nA has no FX at all: MtM = V.(P-K), pure linear delta on P, nothing else."
        "\nB has FX on the float leg only: P and FX_float multiply together -> a"
        "\n  cross term (a 'quanto' effect)."
        "\nC has FX on the fixed leg only: FX_fixed multiplies the CONSTANT K, in a"
        "\n  separate additive term from P -> delta on FX_fixed, but NO cross term."
        "\nD has both: B's cross term, plus C's extra delta. E mirrors D with the"
        "\n  volume sign flipped. F has zero volume, so it is declared but carries"
        "\n  no sensitivity rows at all."
    )

    # ---- Section 2: the sensitivities -----------------------------------
    print("\n" + "=" * _WIDTH)
    print("SECTION 2 — THE COMPILED SENSITIVITIES (read these off by hand)")
    print("=" * _WIDTH)
    sens = printable_sensitivities()
    _rule()
    print(
        f"{'position':>22} | {'delta_P':>10} | {'delta_FXf':>10} | "
        f"{'delta_FXk':>10} | {'cross(P,FXf)':>13}"
    )
    _rule()
    for forward in FORWARDS:
        row = sens[forward.uid]
        print(
            f"{forward.name:>22} | {row.get('delta_P', 0.0):>10,.0f} | "
            f"{row.get('delta_FXf', 0.0):>10,.0f} | {row.get('delta_FXk', 0.0):>10,.0f} | "
            f"{row.get('cross_P_FXf', 0.0):>13,.0f}"
        )
    print(
        "\ndelta_P   = V * FX_float0   (or V, if this book has no FX_float leg)"
        "\ndelta_FXf = V * P0          (only when the float leg is FX-sensitive)"
        "\ndelta_FXk = -V * K          (only when the fixed leg is FX-sensitive)"
        "\ncross     = V               (only when P and FX_float both appear —"
        "\n            the mixed partial d2MtM/dP.dFX_float, NOT halved)"
    )

    calculator = build_book()
    print(
        f"\nCompiled: {calculator.book.num_positions} positions, "
        f"{calculator.book.num_active_factors} active factors, "
        f"nnz={calculator.book.nnz}, cross={calculator.book.nnz_cross}"
    )

    # ---- Section 3: the scenarios ---------------------------------------
    print("\n" + "=" * _WIDTH)
    print("SECTION 3 — TEN SCENARIOS")
    print("=" * _WIDTH)
    _rule()
    print(
        f"{'scenario':>18} | {'dP':>7} | {'dFXf':>7} | {'dFXk':>7} | "
        f"{'P':>7} | {'FX_FLOAT':>9} | {'FX_FIXED':>9}"
    )
    _rule()
    for index, scenario in enumerate(SCENARIOS):
        print(
            f"{scenario.name:>18} | {scenario.d_p:>+7.2f} | {scenario.d_fx_float:>+7.2f} | "
            f"{scenario.d_fx_fixed:>+7.2f} | {SHOCKED_LEVELS[0, index]:>7.2f} | "
            f"{SHOCKED_LEVELS[1, index]:>9.2f} | {SHOCKED_LEVELS[2, index]:>9.2f}"
        )

    # ---- Section 4: run both methods -------------------------------------
    market = calculator.axis.levels(
        base_levels=BASE_LEVELS, shocked_levels=SHOCKED_LEVELS
    )
    result = calculator.run(market)
    sensitivity_pnl = result.pnl.astype(np.float64)
    order = np.argsort(result.position_uids)  # already ascending; explicit for clarity
    sensitivity_pnl = sensitivity_pnl[order]
    reval_pnl = full_reval_pnl_grid()

    # ---- Section 5: worked example, in prose -----------------------------
    print("\n" + "=" * _WIDTH)
    print("SECTION 4 — ONE SCENARIO, BY HAND: Book D at S9 CRISIS")
    print("=" * _WIDTH)
    d_row = next(i for i, f in enumerate(FORWARDS) if f.name.startswith("D "))
    s9 = 9
    scenario = SCENARIOS[s9]
    p9, ff9, fk9 = SHOCKED_LEVELS[:, s9]
    print(
        f"\nLevels move to  P={p9:g} (dP={scenario.d_p:+g})   "
        f"FX_FLOAT={ff9:g} (dFXf={scenario.d_fx_float:+g})   "
        f"FX_FIXED={fk9:g} (dFXk={scenario.d_fx_fixed:+g})"
    )
    mtm_shocked = 10.0 * p9 * ff9 - 10.0 * 90.0 * fk9
    mtm_base = 10.0 * BASE_P * BASE_FX_FLOAT - 10.0 * 90.0 * BASE_FX_FIXED
    print(
        f"\nFULL REVAL:"
        f"\n  MtM(shocked) = 10*{p9:g}*{ff9:g} - 10*90*{fk9:g} = "
        f"{10 * p9 * ff9:,.1f} - {10 * 90 * fk9:,.1f} = {mtm_shocked:,.1f}"
        f"\n  MtM(base)    = 10*{BASE_P:g}*{BASE_FX_FLOAT:g} - 10*90*{BASE_FX_FIXED:g} = "
        f"{10 * BASE_P * BASE_FX_FLOAT:,.1f} - {10 * 90 * BASE_FX_FIXED:,.1f} = {mtm_base:,.1f}"
        f"\n  PnL = {mtm_shocked:,.1f} - {mtm_base:,.1f} = {mtm_shocked - mtm_base:,.1f}"
    )
    row = sens[FORWARDS[d_row].uid]
    t1 = row["delta_P"] * scenario.d_p
    t2 = row["delta_FXf"] * scenario.d_fx_float
    t3 = row["delta_FXk"] * scenario.d_fx_fixed
    t4 = row["cross_P_FXf"] * scenario.d_p * scenario.d_fx_float
    print(
        f"\nSENSITIVITY (Taylor):"
        f"\n  delta_P   * dP    = {row['delta_P']:g} * {scenario.d_p:+g}  = {t1:+,.1f}"
        f"\n  delta_FXf * dFXf  = {row['delta_FXf']:g} * {scenario.d_fx_float:+g}  = {t2:+,.1f}"
        f"\n  delta_FXk * dFXk  = {row['delta_FXk']:g} * {scenario.d_fx_fixed:+g}  = {t3:+,.1f}"
        f"\n  cross * dP * dFXf = {row['cross_P_FXf']:g} * {scenario.d_p:+g} * "
        f"{scenario.d_fx_float:+g} = {t4:+,.1f}"
        f"\n  sum               = {t1:+,.1f} {t2:+,.1f} {t3:+,.1f} {t4:+,.1f} "
        f"= {t1 + t2 + t3 + t4:,.1f}"
    )
    print(
        f"\nFull reval gave {mtm_shocked - mtm_base:,.1f}. Taylor gave "
        f"{t1 + t2 + t3 + t4:,.1f}. Section 0 said these MUST agree — not"
        "\napproximately, exactly — because the payoff is bilinear. They do."
    )

    # ---- Section 6: every scenario, every position, both methods --------
    print("\n" + "=" * _WIDTH)
    print("SECTION 5 — EVERY SCENARIO, EVERY POSITION, BOTH METHODS")
    print("=" * _WIDTH)
    print(
        "\n('diff' is Taylor sum minus full reval, in scientific notation — it should"
        "\n sit near float32 machine epsilon, e.g. 1e-6 to 1e-11, not read as exactly"
        "\n zero. A non-tiny diff here would mean a real defect, not rounding.)"
    )
    portfolio_reval = np.empty(NUM_SCENARIOS)
    portfolio_sensitivity = np.empty(NUM_SCENARIOS)
    for scenario_index, scenario in enumerate(SCENARIOS):
        print(
            f"\n{scenario.name}   (dP={scenario.d_p:+g}, dFXf={scenario.d_fx_float:+g}, "
            f"dFXk={scenario.d_fx_fixed:+g})"
        )
        _rule()
        print(
            f"{'position':>22} | {'dP.deltaP':>10} | {'dFXf.deltaFXf':>13} | "
            f"{'dFXk.deltaFXk':>13} | {'cross term':>10} | {'Taylor sum':>11} | "
            f"{'full reval':>11} | {'diff':>10}"
        )
        _rule()
        for row_index, forward in enumerate(FORWARDS):
            row = sens[forward.uid]
            term_p = _no_negative_zero(row.get("delta_P", 0.0) * scenario.d_p)
            term_ff = _no_negative_zero(row.get("delta_FXf", 0.0) * scenario.d_fx_float)
            term_fk = _no_negative_zero(row.get("delta_FXk", 0.0) * scenario.d_fx_fixed)
            term_x = _no_negative_zero(
                row.get("cross_P_FXf", 0.0) * scenario.d_p * scenario.d_fx_float
            )
            taylor_total = _no_negative_zero(term_p + term_ff + term_fk + term_x)
            reval_total = _no_negative_zero(reval_pnl[row_index, scenario_index])
            diff = taylor_total - reval_total
            print(
                f"{forward.name:>22} | {term_p:>10,.1f} | {term_ff:>13,.1f} | "
                f"{term_fk:>13,.1f} | {term_x:>10,.1f} | {taylor_total:>11,.1f} | "
                f"{reval_total:>11,.1f} | {diff:>10.2e}"
            )
        portfolio_reval[scenario_index] = reval_pnl[:, scenario_index].sum()
        portfolio_sensitivity[scenario_index] = sensitivity_pnl[:, scenario_index].sum()
        portfolio_diff = (
            portfolio_sensitivity[scenario_index] - portfolio_reval[scenario_index]
        )
        print(
            f"{'PORTFOLIO':>22} | {'':>10} | {'':>13} | {'':>13} | {'':>10} | "
            f"{portfolio_sensitivity[scenario_index]:>11,.1f} | "
            f"{portfolio_reval[scenario_index]:>11,.1f} | {portfolio_diff:>10.2e}"
        )

    max_abs_diff = float(np.abs(sensitivity_pnl - reval_pnl).max())
    print(
        f"\nmax |sensitivity - full reval| across all "
        f"{sensitivity_pnl.size} cells = {max_abs_diff:.6f}"
        "\n(float32-displacement rounding — see Section 0 for why the exact"
        "\n algebra has no truncation error to hide behind)."
    )
    assert max_abs_diff < 0.01, max_abs_diff

    # ---- Section 7: VaR --------------------------------------------------
    print("\n" + "=" * _WIDTH)
    print("SECTION 6 — PORTFOLIO PnL, RANKED, AND VaR")
    print("=" * _WIDTH)
    ranking = np.argsort(portfolio_reval)  # ascending: worst loss first
    _rule()
    print(f"{'rank':>4} | {'scenario':>18} | {'full reval':>12} | {'sensitivity':>12}")
    _rule()
    for rank, scenario_index in enumerate(ranking, start=1):
        print(
            f"{rank:>4} | {SCENARIOS[scenario_index].name:>18} | "
            f"{portfolio_reval[scenario_index]:>12,.1f} | "
            f"{portfolio_sensitivity[scenario_index]:>12,.1f}"
        )

    confidence_level = 0.90
    tail_fraction = 1.0 - confidence_level
    tail_product = tail_fraction * NUM_SCENARIOS
    tail_count = max(1, int(tail_product))
    print(
        f"\ntail_count = max(1, int((1 - {confidence_level:.2f}) * {NUM_SCENARIOS}))"
        f"\n           = max(1, int({tail_fraction!r} * {NUM_SCENARIOS}))     "
        "<- 1 - 0.90 is NOT exactly 0.1 in binary floating point"
        f"\n           = max(1, int({tail_product!r}))              "
        f"<- so this truncates to {int(tail_product)}, not the {tail_fraction * NUM_SCENARIOS:.1f} you'd expect on paper"
        f"\n           = max(1, {int(tail_product)}) = {tail_count}"
        f"\n-> the max(1, ...) floor is what makes 'worst of 10' hold at all: it would"
        "\n   hold at tail_count=1 whether the truncation lands on 0 or 1. With"
        f"\n   {tail_count} scenario in the tail, VaR{int(confidence_level * 100)} IS the worst single scenario,"
        "\n   and ES (the mean of a one-element tail) equals it exactly."
    )
    var_reval, es_reval = var_es(portfolio_reval, confidence_level)
    var_sensitivity, es_sensitivity = var_es(portfolio_sensitivity, confidence_level)
    worst_scenario = SCENARIOS[int(ranking[0])]
    print(
        f"\nWorst scenario: {worst_scenario.name}"
        f"\n  VaR90 (full reval)   = {var_reval:>10,.1f}     ES90 (full reval)   = {es_reval:>10,.1f}"
        f"\n  VaR90 (sensitivity)  = {var_sensitivity:>10,.1f}     ES90 (sensitivity)  = {es_sensitivity:>10,.1f}"
    )
    assert np.isclose(var_reval, es_reval), "n=10 at 90%: VaR must equal ES exactly"
    assert np.isclose(var_reval, var_sensitivity, atol=0.05), (
        var_reval,
        var_sensitivity,
    )

    print(f"\nCOMPONENT BREAKDOWN AT {worst_scenario.name} (who drives the worst day)")
    _rule()
    print(f"{'position':>22} | {'full reval':>12} | {'sensitivity':>12}")
    _rule()
    worst_index = int(ranking[0])
    for row_index, forward in enumerate(FORWARDS):
        print(
            f"{forward.name:>22} | {reval_pnl[row_index, worst_index]:>12,.1f} | "
            f"{sensitivity_pnl[row_index, worst_index]:>12,.1f}"
        )
    _rule()
    print(
        f"{'total':>22} | {reval_pnl[:, worst_index].sum():>12,.1f} | "
        f"{sensitivity_pnl[:, worst_index].sum():>12,.1f}"
    )

    print(
        "\nprovenance: "
        f"{result.provenance.market_contract}, "
        f"{result.provenance.factor_key_encoding} encoding, "
        f"{result.provenance.num_active_factors} active factors, "
        f"{result.provenance.num_first_order_entries} first-order entries, "
        f"{result.provenance.num_cross_entries} cross entries, "
        f"{result.provenance.num_positions_without_sensitivities} position(s) "
        "without sensitivities (F, the zero-volume book)."
    )

    print("\n" + "=" * _WIDTH)
    print("ALL CHECKS PASSED".center(_WIDTH))
    print("=" * _WIDTH)


if __name__ == "__main__":
    run_demo()
