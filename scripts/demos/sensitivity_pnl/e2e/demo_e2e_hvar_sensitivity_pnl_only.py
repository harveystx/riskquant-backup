# scripts/demos/sensitivity_pnl/e2e/demo_e2e_hvar_sensitivity_pnl_only.py
"""
Historical VaR from sensitivities alone. No revaluation anywhere in this file.

Historical simulation asks one question: what would today's book have lost on
each of the last N trading days? A full-reval HVaR answers it by repricing the
book N times. This one never prices anything — it takes today's greeks, applies
each historical day's factor move as a Taylor displacement, and reads the loss
distribution straight off the resulting `(M, S)` grid.

    daily factor level history  (K, T)
        -> historical displacements      per factor, in its DECLARED space
        -> absorb_displacement           (K, S) f32 -> (M, S) f32, one call
        -> compute_var_es                unweighted order statistic
        -> age-weighted VaR              weighted order statistic (see below)
        -> rolling backtest              Kupiec POF + Basel traffic light

Why this uses the standalone path
---------------------------------
`absorb_displacement` takes displacements directly rather than shocked levels.
For HVaR that is not a shortcut, it is the natural contract: a historical
scenario IS a displacement — yesterday's move applied to today's book — and
reconstructing shocked levels only to have the engine difference them back out
would add a round trip and a base-level dependency for nothing.

Nothing below imports `riskquant.simlab`, and the engine is constructed with no
`base_levels` at all. This file is the proof that `sensitivity_pnl` stands
alone.

The one thing HVaR gives you that Monte Carlo cannot
----------------------------------------------------
Every scenario has a DATE. When the tail is driven by 2 March rather than by an
anonymous draw, the risk conversation changes: you can ask what happened that
day and whether you believe it can happen again. The report names the dates.

The three failure modes this demo makes visible
-----------------------------------------------
1.  **Window effects (ghosting).** A violent day ages out of the window and VaR
    falls overnight with no change in the book. The report finds the largest
    such drop and names the scenario that left.
2.  **Clustered exceptions.** HVaR reacts to volatility only as fast as the
    window fills. Exception counts alone pass; exceptions arriving in a clump
    are the real signal, so the backtest reports the longest run.
3.  **Taylor reach.** Historical windows contain real crises, and a second-order
    expansion evaluated at a 20% move is extrapolating. Reported per factor.

Everything here is synthetic and seeded. The numbers are illustrative; the model
structure is the point.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import (
    QUOTE_BASE_NATURAL,
    QUOTE_BASE_PER_BASIS_POINT,
    QUOTE_BASE_PER_PERCENT,
    QUOTE_BASE_PER_VOL_POINT,
    DisplacementConvention,
)
from riskquant.sensitivity_pnl.engine import SensitivityPnlEngine
from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid

#: Trading days of factor history to synthesise (~5 years).
HISTORY_DAYS = 1_300

#: Observation window. Basel's floor is one year; 500 days is the common desk
#: choice — long enough to contain a crisis, short enough to respond to one.
HVAR_WINDOW = 500

#: Age-weighting decay (Boudoukh-Richardson-Whitelaw). 1.0 disables weighting.
#: 0.99 gives a ~100-day half life, so a 500-day window still carries a crisis
#: but does not weight it as heavily as last week.
DECAY_LAMBDA = 0.99

CONFIDENCE_LEVELS = (0.95, 0.975, 0.99)
BACKTEST_LEVEL = 0.99

#: Displacement magnitude beyond which a second-order expansion is extrapolating
#: rather than interpolating. Horizon-dependent, so it belongs to the run
#: configuration rather than to the engine. Purely diagnostic.
TAYLOR_STRAIN_THRESHOLD = 0.10

#: Day 0 of the synthetic history, as a day ordinal (days since 1970-01-01).
#: Dates stay integers everywhere and are formatted only at the report edge.
EPOCH_ORDINAL = 19_723  # 2024-01-02

SEED = 20260727


# =========================================================================
# 1. RISK-FACTOR UNIVERSE
# =========================================================================

FACTOR_LOOKUP: dict[str, int] = {
    "EQ.ACME.SPOT": 100_100,
    "EQ.BOREAS.SPOT": 100_200,
    "IR.USD.2Y": 200_100,
    "IR.USD.5Y": 200_200,
    "IR.USD.10Y": 200_300,
    "VOL.ACME.1Y": 300_100,
    "VOL.BOREAS.1Y": 300_200,
    "FX.EURUSD": 400_100,
}

#: The space each factor's greeks were differentiated in — and therefore the
#: space its historical shift must be measured in. In HVaR terms this IS the
#: shift type: relative for equity, absolute for rates and vol, log for FX.
#: Measuring a historical move in the wrong space is not a rounding error, it is
#: a wrong number of the same order as the right one.
FACTOR_CONVENTION: dict[str, DisplacementConvention] = {
    "EQ.ACME.SPOT": DisplacementConvention.RELATIVE,
    "EQ.BOREAS.SPOT": DisplacementConvention.RELATIVE,
    "IR.USD.2Y": DisplacementConvention.ABSOLUTE,
    "IR.USD.5Y": DisplacementConvention.ABSOLUTE,
    "IR.USD.10Y": DisplacementConvention.ABSOLUTE,
    "VOL.ACME.1Y": DisplacementConvention.ABSOLUTE,
    "VOL.BOREAS.1Y": DisplacementConvention.ABSOLUTE,
    "FX.EURUSD": DisplacementConvention.LOG,
}

FACTOR_QUOTE_BASE: dict[str, float] = {
    "EQ.ACME.SPOT": QUOTE_BASE_PER_PERCENT,  # cash delta per 1% move
    "EQ.BOREAS.SPOT": QUOTE_BASE_PER_PERCENT,
    "IR.USD.2Y": QUOTE_BASE_PER_BASIS_POINT,  # DV01
    "IR.USD.5Y": QUOTE_BASE_PER_BASIS_POINT,
    "IR.USD.10Y": QUOTE_BASE_PER_BASIS_POINT,
    "VOL.ACME.1Y": QUOTE_BASE_PER_VOL_POINT,  # vega per vol point
    "VOL.BOREAS.1Y": QUOTE_BASE_PER_VOL_POINT,
    "FX.EURUSD": QUOTE_BASE_NATURAL,
}

#: Starting level for the synthetic history, and the annualised volatility of
#: each factor's displacement in the units its convention implies.
FACTOR_START_LEVEL: dict[str, float] = {
    "EQ.ACME.SPOT": 172.40,
    "EQ.BOREAS.SPOT": 58.10,
    "IR.USD.2Y": 0.0431,
    "IR.USD.5Y": 0.0446,
    "IR.USD.10Y": 0.0468,
    "VOL.ACME.1Y": 0.2280,
    "VOL.BOREAS.1Y": 0.2740,
    "FX.EURUSD": 1.0975,
}

FACTOR_SIGMA: dict[str, float] = {
    "EQ.ACME.SPOT": 0.26,
    "EQ.BOREAS.SPOT": 0.33,
    "IR.USD.2Y": 0.0080,
    "IR.USD.5Y": 0.0086,
    "IR.USD.10Y": 0.0091,
    "VOL.ACME.1Y": 0.0850,
    "VOL.BOREAS.1Y": 0.1050,
    "FX.EURUSD": 0.085,
}


def factor_names_in_row_order() -> list[str]:
    return sorted(FACTOR_LOOKUP, key=lambda name: FACTOR_LOOKUP[name])


def level_universe_factor_ids() -> npt.NDArray[np.int32]:
    return np.array(sorted(FACTOR_LOOKUP.values()), dtype=np.int32)


def factor_meta_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "factor_id": list(FACTOR_LOOKUP),
            "displacement_convention_id": [
                int(FACTOR_CONVENTION[name]) for name in FACTOR_LOOKUP
            ],
            "quote_base": [FACTOR_QUOTE_BASE[name] for name in FACTOR_LOOKUP],
        }
    )


# =========================================================================
# 2. THE BOOK — sensitivities only, no pricing model anywhere
# =========================================================================

SENSITIVITY_ROWS: list[tuple[str, str, float, float]] = [
    # position_id,       factor_id,        first_order,  second_order
    ("XOTC_ACME_AUTOCALL", "EQ.ACME.SPOT", 412_000.0, -18_400.0),
    ("XOTC_ACME_AUTOCALL", "VOL.ACME.1Y", -96_000.0, 1_200.0),
    ("XOTC_ACME_AUTOCALL", "IR.USD.2Y", -18_400.0, 42.0),
    ("XOTC_BOREAS_CLIQUET", "EQ.BOREAS.SPOT", -140_000.0, -46_000.0),
    ("XOTC_BOREAS_CLIQUET", "VOL.BOREAS.1Y", 154_000.0, -2_900.0),
    ("STRN_USD_CALLABLE", "IR.USD.2Y", -41_200.0, 118.0),
    ("STRN_USD_CALLABLE", "IR.USD.5Y", -88_600.0, 305.0),
    ("STRN_USD_CALLABLE", "IR.USD.10Y", -134_900.0, 690.0),
    ("STRN_USD_CALLABLE", "VOL.ACME.1Y", -22_500.0, 180.0),
    ("XOTC_QUANTO_BASKET", "EQ.ACME.SPOT", 96_500.0, -4_200.0),
    ("XOTC_QUANTO_BASKET", "EQ.BOREAS.SPOT", 88_100.0, -3_800.0),
    ("XOTC_QUANTO_BASKET", "FX.EURUSD", 720_000.0, -140_000.0),
    ("XOTC_QUANTO_BASKET", "IR.USD.5Y", -9_800.0, 25.0),
    ("RV_VOL_SPREAD", "VOL.ACME.1Y", 61_000.0, 0.0),
    ("RV_VOL_SPREAD", "VOL.BOREAS.1Y", -58_400.0, 0.0),
    # Linear cash equity — in a sensitivity-only model even the trivial
    # positions come in as greeks. Delta is exact for these, so no Taylor error
    # is introduced; they are carried here rather than revalued because this
    # model has no pricer to revalue them with.
    ("CASH_EQ_ACME", "EQ.ACME.SPOT", 379_280.0, 0.0),
    ("CASH_EQ_BOREAS", "EQ.BOREAS.SPOT", -278_880.0, 0.0),
]

CROSS_SENSITIVITY_ROWS: list[tuple[str, str, str, float]] = [
    ("XOTC_ACME_AUTOCALL", "EQ.ACME.SPOT", "VOL.ACME.1Y", -9_400.0),
    ("XOTC_BOREAS_CLIQUET", "EQ.BOREAS.SPOT", "VOL.BOREAS.1Y", -12_800.0),
    ("XOTC_QUANTO_BASKET", "EQ.ACME.SPOT", "FX.EURUSD", 21_500.0),
    ("XOTC_QUANTO_BASKET", "EQ.BOREAS.SPOT", "FX.EURUSD", 18_900.0),
]

POSITION_LOOKUP: dict[str, int] = {
    "XOTC_ACME_AUTOCALL": 8_100_001,
    "XOTC_BOREAS_CLIQUET": 8_100_002,
    "STRN_USD_CALLABLE": 8_100_003,
    "XOTC_QUANTO_BASKET": 8_100_004,
    "RV_VOL_SPREAD": 8_100_005,
    "CASH_EQ_ACME": 8_100_006,
    "CASH_EQ_BOREAS": 8_100_007,
}


def build_book() -> CompiledSensitivityBook:
    """Strings die here; units fold here; the CSR SoA book comes out."""
    diagonal = pl.DataFrame(
        SENSITIVITY_ROWS,
        schema={
            "position_id": pl.String,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross = pl.DataFrame(
        CROSS_SENSITIVITY_ROWS,
        schema={
            "position_id": pl.String,
            "factor_id_i": pl.String,
            "factor_id_j": pl.String,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    return compile_sensitivity_book(
        sensitivities=diagonal,
        factor_meta=factor_meta_frame(),
        level_universe_factor_ids=level_universe_factor_ids(),
        cross_sensitivities=cross,
        position_id_lookup=POSITION_LOOKUP,
        factor_id_lookup=FACTOR_LOOKUP,
    )


# =========================================================================
# 3. HISTORY -> DISPLACEMENTS
# =========================================================================


def synthesise_history(
    *, num_days: int, seed: int = SEED
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    A `(K, T)` daily factor level series and its `(T,)` date ordinals.

    Stand-in for the market-data warehouse. Built with GARCH-style volatility
    clustering and a shared market factor, because a history of independent
    Gaussian days would make every HVaR diagnostic below look perfect — the
    clustering IS the phenomenon the backtest is meant to detect.
    """
    rng = np.random.default_rng(seed)
    names = factor_names_in_row_order()
    num_factors = len(names)

    loading = np.array([0.85, 0.90, 0.35, 0.40, 0.42, -0.70, -0.66, 0.25])
    idiosyncratic = np.sqrt(np.clip(1.0 - loading**2, 0.05, None))

    # GARCH(1,1) on the common market factor: persistent, occasionally violent.
    omega, alpha, beta = 0.02, 0.13, 0.85
    variance = omega / (1.0 - alpha - beta)
    market = np.empty(num_days)
    for t in range(num_days):
        shock = rng.standard_t(df=6) / np.sqrt(6.0 / 4.0)
        market[t] = np.sqrt(variance) * shock
        variance = omega + alpha * market[t] ** 2 + beta * variance

    unit_moves = loading[:, None] * market[None, :] + idiosyncratic[
        :, None
    ] * rng.standard_normal((num_factors, num_days))
    daily_sigma = np.array([FACTOR_SIGMA[name] for name in names]) / np.sqrt(252.0)
    displacement = unit_moves * daily_sigma[:, None]

    start = np.array([FACTOR_START_LEVEL[name] for name in names])
    is_absolute = np.array(
        [FACTOR_CONVENTION[name] is DisplacementConvention.ABSOLUTE for name in names],
        dtype=np.bool_,
    )

    levels = np.empty((num_factors, num_days), dtype=np.float64)
    levels[:, 0] = start
    for t in range(1, num_days):
        step = displacement[:, t]
        levels[is_absolute, t] = levels[is_absolute, t - 1] + step[is_absolute]
        levels[~is_absolute, t] = levels[~is_absolute, t - 1] * np.exp(
            step[~is_absolute]
        )

    # Rates are absolute-shifted and can wander through zero on a long synthetic
    # path; floor them so the LOG/RELATIVE factors' base-level guard and the
    # rate levels themselves stay economically sensible.
    rate_rows = np.array([name.startswith("IR.") for name in names], dtype=np.bool_)
    np.clip(levels[rate_rows], 0.0005, None, out=levels[rate_rows])

    # Weekends are not trading days; the ordinal series only needs to be
    # monotonic and business-day-like for the report to read correctly.
    ordinals = EPOCH_ORDINAL + np.arange(num_days, dtype=np.int32) * 7 // 5
    return np.ascontiguousarray(levels), np.ascontiguousarray(ordinals)


def historical_displacements(
    levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """
    `(K, T)` levels -> `(K, T-1)` float32 displacements, per declared convention.

    This is the entire "shift type" question of a historical VaR model, and it
    is resolved by the same `DisplacementConvention` the greeks were quoted
    against. Column j is the move from day j to day j+1.

    float32 because that is the engine's standalone input contract and the
    `riskquant.constants` rule for scenario shocks: representation error sits
    three to five orders below the Taylor truncation error it feeds.
    """
    names = factor_names_in_row_order()
    previous, current = levels[:, :-1], levels[:, 1:]
    out = np.empty_like(current)
    for row, name in enumerate(names):
        convention = FACTOR_CONVENTION[name]
        if convention is DisplacementConvention.ABSOLUTE:
            out[row] = current[row] - previous[row]
        elif convention is DisplacementConvention.RELATIVE:
            out[row] = current[row] / previous[row] - 1.0
        else:
            out[row] = np.log(current[row] / previous[row])
    return np.ascontiguousarray(out, dtype=np.float32)


# =========================================================================
# 4. THE HVaR ENGINE — one grid, one engine, reused across every window
# =========================================================================


class HistoricalVarModel:
    """
    Today's book against the whole factor history, in ONE engine pass.

    The realisation that shapes this class: a historical VaR window is a
    contiguous SLICE of the scenario axis. Every window the model will ever
    report on — the live one, and every window a backtest walks through — is a
    view into one `(M, T-1)` grid. Re-running the engine per window would repeat
    the same arithmetic once per backtest day; here the engine runs once, over
    every historical move, and windowing is pure indexing afterwards.

    `base_levels` is never supplied. The engine has no notion of a market level
    in this model, and `absorb_levels` would refuse to run — the intended
    guarantee, not a limitation.
    """

    def __init__(
        self,
        *,
        book: CompiledSensitivityBook,
        displacements: npt.NDArray[np.float32],
        window: int,
        decay: float,
        batch_size: int = 500,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must satisfy 0 < lambda <= 1")
        num_scenarios = displacements.shape[1]
        if window > num_scenarios:
            raise ValueError(
                f"window of {window} exceeds the {num_scenarios} historical moves"
            )
        self.book = book
        self.window = window
        self.grid: PnlGrid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=num_scenarios
        )
        engine = SensitivityPnlEngine(
            book=book,
            out=self.grid.block_view(0),
            batch_size=min(batch_size, num_scenarios),
        )

        offset = 0
        while offset < num_scenarios:
            batch_len = min(engine.batch_size, num_scenarios - offset)
            engine.absorb_displacement(
                displacement=displacements,
                batch_len=batch_len,
                scenario_offset=offset,
                source_offset=offset,
            )
            offset += batch_len
        self.batches = -(-num_scenarios // engine.batch_size)

        # Summed in f64 explicitly: accumulating positions in f32 loses the
        # small legs, and the tail is exactly where offsetting positions matter.
        self.scenario_pnl = self.grid.pnl.sum(axis=0, dtype=np.float64)

        # Age weights, newest scenario last, normalised to a probability mass.
        age = np.arange(window - 1, -1, -1, dtype=np.float64)
        weights = decay**age
        self.weights = weights / weights.sum()

    def window_pnl(self, *, end_exclusive: int) -> npt.NDArray[np.float64]:
        """`(window,)` portfolio PnL view for the window ending at `end_exclusive`."""
        start = end_exclusive - self.window
        if start < 0:
            raise ValueError(
                f"window of {self.window} needs {self.window} scenarios before "
                f"index {end_exclusive}"
            )
        return self.scenario_pnl[start:end_exclusive]

    def window_grid(self, *, end_exclusive: int) -> npt.NDArray[np.float32]:
        """`(M, window)` per-position PnL view for the same window."""
        return self.grid.pnl[:, end_exclusive - self.window : end_exclusive]


# =========================================================================
# 5. RISK METRICS
# =========================================================================


def var_es(
    portfolio: npt.NDArray[np.float64], confidence_level: float
) -> tuple[float, float]:
    """
    Unweighted VaR and ES as POSITIVE LOSSES.

    `compute_var_es` is a lower-tail metric on the values it is handed. PnL is
    signed with losses negative, so the raw result is the worst PnL — a negative
    number — and the sign flip is what turns it into a loss. Not cosmetic:
    reversed, it reports the best scenario as the VaR.
    """
    matrix = np.ascontiguousarray(portfolio.reshape(1, -1), dtype=np.float64)
    result = compute_var_es(matrix, confidence_level=confidence_level)
    return -float(result.var[0]), -float(result.es[0])


def weighted_var_es(
    *,
    portfolio: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    confidence_level: float,
) -> tuple[float, float]:
    """
    Age-weighted (Boudoukh-Richardson-Whitelaw) VaR and ES, as positive losses.

    Deliberately NOT routed through `compute_var_es`. That function computes an
    unweighted order statistic — the k-th worst of S equally likely scenarios —
    and there is no argument you can pass it that makes a scenario count for
    more than another. Weighted HVaR needs the weighted quantile: sort by PnL
    ascending, walk the cumulative weight, and stop at 1 - alpha.

    Passing weighted scenarios to an unweighted estimator is the kind of error
    that produces a plausible number, so the separation is structural rather
    than a matter of care. `riskquant.scenario_metrics` has no weighted
    estimator today; see the note at the end of this module.
    """
    order = np.argsort(portfolio)
    sorted_pnl = portfolio[order]
    cumulative = np.cumsum(weights[order])
    tail_mass = 1.0 - confidence_level

    cut = int(np.searchsorted(cumulative, tail_mass, side="left"))
    cut = min(cut, sorted_pnl.shape[0] - 1)

    tail_weights = weights[order][: cut + 1]
    tail_pnl = sorted_pnl[: cut + 1]
    expected_shortfall = float(np.dot(tail_pnl, tail_weights) / tail_weights.sum())
    return -float(sorted_pnl[cut]), -expected_shortfall


def component_es(
    *,
    position_pnl: npt.NDArray[np.float32],
    portfolio: npt.NDArray[np.float64],
    confidence_level: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Per-position mean PnL over the scenarios that make up the PORTFOLIO's tail.

    The decomposition that adds up. A position's own worst days are generally
    not the portfolio's worst days, so summing standalone per-position ES
    over-states the total by exactly the diversification benefit. `tail_count`
    mirrors `_compute_lower_tail_parameters`; the run asserts the identity
    rather than trusting it.
    """
    scenario_count = portfolio.shape[0]
    tail_count = max(1, int((1.0 - confidence_level) * scenario_count))
    tail_scenarios = np.argpartition(portfolio, tail_count - 1)[:tail_count]
    contribution = position_pnl[:, tail_scenarios].mean(axis=1, dtype=np.float64)
    # `+ 0.0` normalises IEEE negative zero, which a flat position produces
    # under negation and which prints as "-0".
    return -contribution + 0.0, tail_scenarios


def format_ordinal(ordinal: int) -> str:
    """Day ordinal -> ISO date, at the report edge only."""
    return str(np.datetime64(int(ordinal), "D"))


# =========================================================================
# 6. BACKTEST
# =========================================================================


def kupiec_pof(*, exceptions: int, observations: int, level: float) -> float:
    """
    Kupiec proportion-of-failures likelihood ratio, chi-square with 1 dof.

    Tests only the exception COUNT. It is blind to when they arrive, which is
    why the run also reports the longest exception run — HVaR's characteristic
    failure is clustering, not miscounting.
    """
    expected_rate = 1.0 - level
    if exceptions == 0:
        return -2.0 * observations * np.log(1.0 - expected_rate)
    observed_rate = exceptions / observations
    return -2.0 * (
        (observations - exceptions) * np.log(1.0 - expected_rate)
        + exceptions * np.log(expected_rate)
        - (observations - exceptions) * np.log(1.0 - observed_rate)
        - exceptions * np.log(observed_rate)
    )


def basel_traffic_light(*, exceptions: int, observations: int) -> str:
    """
    Basel zones, scaled from the 250-day table to the observed sample length.

    The published boundaries (green <5, yellow 5-9, red >=10 at 250 days and
    99%) are counts, not rates, so comparing a longer backtest against them
    directly would be wrong.
    """
    scale = observations / 250.0
    if exceptions < 5 * scale:
        return "GREEN"
    if exceptions < 10 * scale:
        return "YELLOW"
    return "RED"


def longest_exception_run(breaches: npt.NDArray[np.bool_]) -> int:
    """Longest consecutive stretch of breaches — the clustering signal."""
    longest = current = 0
    for breached in breaches:
        current = current + 1 if breached else 0
        longest = max(longest, current)
    return longest


# =========================================================================
# 7. REPORT
# =========================================================================

_WIDTH = 88


def _rule() -> None:
    print("-" * _WIDTH)


def run_demo() -> None:
    print("=" * _WIDTH)
    print(f"{'SENSITIVITY-ONLY HISTORICAL VaR':^{_WIDTH}}")
    print("=" * _WIDTH)

    book = build_book()
    levels, ordinals = synthesise_history(num_days=HISTORY_DAYS)
    displacements = historical_displacements(levels)
    # Displacement column j is the move INTO day j+1, so it carries that date.
    scenario_ordinals = ordinals[1:]

    print(
        f"\nBook       : {book.num_positions} positions, "
        f"{book.num_active_factors} active factors, nnz={book.nnz}, "
        f"cross={book.nnz_cross}"
    )
    print(
        f"History    : {displacements.shape[1]:,} daily moves, "
        f"{format_ordinal(int(scenario_ordinals[0]))} to "
        f"{format_ordinal(int(scenario_ordinals[-1]))}, "
        f"{displacements.dtype}, {displacements.nbytes / 1024:.0f} KB"
    )

    model = HistoricalVarModel(
        book=book,
        displacements=displacements,
        window=HVAR_WINDOW,
        decay=DECAY_LAMBDA,
    )
    num_scenarios = displacements.shape[1]
    portfolio = model.window_pnl(end_exclusive=num_scenarios)
    window_ordinals = scenario_ordinals[-HVAR_WINDOW:]
    print(
        f"Live window: {HVAR_WINDOW} scenarios, "
        f"{format_ordinal(int(window_ordinals[0]))} to "
        f"{format_ordinal(int(window_ordinals[-1]))}"
    )
    print(
        f"Grid       : {model.grid.pnl.shape[0]} rows x "
        f"{model.grid.num_scenarios:,} scenarios, {model.grid.pnl.dtype}, "
        f"{model.grid.pnl.nbytes / 1024:.0f} KB, "
        f"one pass in {model.batches} batches"
    )

    # ---- Headline ----------------------------------------------------
    print("\nHVaR  (positive = loss)")
    _rule()
    print(
        f"{'level':>8} | {'VaR':>14} | {'ES':>14} | {'weighted VaR':>14} | "
        f"{'weighted ES':>14}"
    )
    _rule()
    for level in CONFIDENCE_LEVELS:
        plain_var, plain_es = var_es(portfolio, level)
        aged_var, aged_es = weighted_var_es(
            portfolio=portfolio, weights=model.weights, confidence_level=level
        )
        print(
            f"{level:>8.1%} | {plain_var:>14,.0f} | {plain_es:>14,.0f} | "
            f"{aged_var:>14,.0f} | {aged_es:>14,.0f}"
        )
    _rule()
    print(
        f"Age weighting: lambda={DECAY_LAMBDA}, "
        f"half life {np.log(0.5) / np.log(DECAY_LAMBDA):.0f} days, "
        f"newest scenario carries {model.weights[-1] / model.weights[0]:.0f}x "
        f"the oldest"
    )

    # ---- Which days drive the tail -----------------------------------
    # The HVaR-only question. Every scenario is a date, so the tail has names.
    live_grid = model.window_grid(end_exclusive=num_scenarios)
    contribution, tail_scenarios = component_es(
        position_pnl=live_grid,
        portfolio=portfolio,
        confidence_level=BACKTEST_LEVEL,
    )
    worst = tail_scenarios[np.argsort(portfolio[tail_scenarios])]
    print(f"\nTAIL SCENARIOS AT {BACKTEST_LEVEL:.1%}  ({worst.shape[0]} days)")
    _rule()
    print(
        f"{'rank':>5} | {'date':>12} | {'portfolio PnL':>16} | "
        f"{'largest factor move':>26}"
    )
    _rule()
    names = factor_names_in_row_order()
    for rank, scenario in enumerate(worst[:6], start=1):
        column = displacements[:, num_scenarios - HVAR_WINDOW + scenario]
        driver = int(np.argmax(np.abs(column)))
        print(
            f"{rank:>5} | "
            f"{format_ordinal(int(window_ordinals[scenario])):>12} | "
            f"{portfolio[scenario]:>16,.0f} | "
            f"{names[driver]:>18} {column[driver]:>+7.2%}"
        )

    # ---- Component attribution ---------------------------------------
    _, portfolio_es = var_es(portfolio, BACKTEST_LEVEL)
    uid_to_name = {uid: name for name, uid in POSITION_LOOKUP.items()}
    print(f"\nCOMPONENT ES AT {BACKTEST_LEVEL:.1%}")
    _rule()
    print(f"{'position':>22} | {'component ES':>16} | {'share':>8}")
    _rule()
    for row in np.argsort(-contribution):
        uid = int(model.grid.trade_uids[row])
        print(
            f"{uid_to_name[uid]:>22} | {contribution[row]:>16,.0f} | "
            f"{contribution[row] / portfolio_es:>8.1%}"
        )
    _rule()
    print(f"{'total':>22} | {contribution.sum():>16,.0f} | {1.0:>8.1%}")
    assert np.isclose(contribution.sum(), portfolio_es, rtol=1e-6), (
        contribution.sum(),
        portfolio_es,
    )

    # ---- Rolling backtest --------------------------------------------
    # Clean-P&L proxy: the SAME sensitivity model evaluated on the realised next
    # day. In FRTB language this is risk-theoretical P&L, and it tests exactly
    # one thing — whether the historical window is a good predictor of tomorrow.
    # It does NOT test the Taylor approximation, because the approximation sits
    # on both sides of the comparison. Backtesting the pricing would need actual
    # trading P&L, which no sensitivity-only model can produce.
    backtest_days = range(HVAR_WINDOW, num_scenarios)
    var_series = np.empty(len(backtest_days))
    for index, day in enumerate(backtest_days):
        var_series[index] = var_es(model.window_pnl(end_exclusive=day), BACKTEST_LEVEL)[
            0
        ]
    # The realised move on day `d` is scenario `d` — the same number the model
    # already produced in its single pass. No re-running anything.
    realised = model.scenario_pnl[HVAR_WINDOW:num_scenarios]

    breaches = realised < -var_series
    exceptions = int(breaches.sum())
    observations = breaches.shape[0]
    expected = observations * (1.0 - BACKTEST_LEVEL)
    likelihood_ratio = kupiec_pof(
        exceptions=exceptions, observations=observations, level=BACKTEST_LEVEL
    )

    print(f"\nROLLING BACKTEST AT {BACKTEST_LEVEL:.1%}  (risk-theoretical P&L)")
    _rule()
    print(f"  observations            {observations:>10,}")
    print(f"  exceptions              {exceptions:>10,}   expected {expected:>8.1f}")
    print(f"  exception rate          {exceptions / observations:>10.2%}")
    print(
        f"  Kupiec POF LR           {likelihood_ratio:>10.2f}   (chi2 1dof: 3.84 at 5%)"
    )
    print(
        f"  verdict                 "
        f"{'REJECT' if likelihood_ratio > 3.84 else 'accept':>10}   "
        f"count consistent with the level"
    )
    print(
        f"  Basel zone              "
        f"{basel_traffic_light(exceptions=exceptions, observations=observations):>10}"
    )
    print(
        f"  longest breach run      {longest_exception_run(breaches):>10}   "
        f"clustering, which the count test cannot see"
    )

    # ---- Ghost effect ------------------------------------------------
    # HVaR's signature artefact: a violent day ages out of the window and VaR
    # falls overnight with no change in the book and no change in the market.
    #
    # var_series[i] covers scenarios [i, i+W). Stepping to i+1 drops scenario i
    # and admits scenario i+W, so a fall can come from EITHER — losing a tail
    # scenario, or admitting a benign one that shifts the order statistic. The
    # attribution below is checked against the pre-drop tail rather than
    # assumed, because "a big day rolled off" is the explanation everyone
    # reaches for and it is not always the right one.
    daily_change = np.diff(var_series)
    ghost = int(np.argmin(daily_change))
    departing = ghost  # scenario index that leaves the window on this step
    pre_drop_var = var_series[ghost]
    departing_pnl = model.scenario_pnl[departing]
    was_in_tail = departing_pnl < -pre_drop_var

    print("\nWINDOW EFFECT (GHOSTING)")
    _rule()
    print(
        f"  Largest one-day VaR fall with no change in the book: "
        f"{daily_change[ghost]:,.0f} "
        f"({-100 * daily_change[ghost] / pre_drop_var:.1f}% of VaR)"
    )
    print(
        f"  observed {format_ordinal(int(scenario_ordinals[HVAR_WINDOW + ghost]))}"
        f"   window dropped "
        f"{format_ordinal(int(scenario_ordinals[departing]))} "
        f"(PnL {departing_pnl:,.0f})"
    )
    if was_in_tail:
        print(
            "  The departing scenario was inside the pre-drop tail, so it is the\n"
            "  cause: a loss the model had been carrying simply became too old."
        )
    else:
        print(
            "  The departing scenario was NOT in the pre-drop tail, so the fall is\n"
            "  the order statistic moving as the window re-sorts, not one day\n"
            "  ageing out. Worth knowing before explaining it to anyone."
        )
    print(
        "  Either way nothing happened in the market. This is why HVaR is\n"
        "  reported alongside its window, and why a VaR fall is not on its own\n"
        "  evidence of de-risking."
    )

    # ---- Taylor reach ------------------------------------------------
    # Live window against the whole history. The contrast is the point: whether
    # a second-order expansion is interpolating or extrapolating depends on
    # which 500 days you happen to be standing in, and that changes with no
    # action from anyone.
    magnitude = np.abs(displacements.astype(np.float64))
    live = magnitude[:, -HVAR_WINDOW:]
    print(f"\nTAYLOR REACH  (|displacement|, threshold {TAYLOR_STRAIN_THRESHOLD:.0%})")
    _rule()
    print(
        f"{'factor':>22} | {'live 99th':>11} | {'live worst':>11} | "
        f"{'history worst':>14} | {'':>9}"
    )
    _rule()
    for slot in range(book.num_active_factors):
        row = int(book.active_level_uix[slot])
        live_worst = float(live[row].max())
        history_worst = float(magnitude[row].max())
        flag = (
            "STRAINED"
            if live_worst > TAYLOR_STRAIN_THRESHOLD
            else ("(in history)" if history_worst > TAYLOR_STRAIN_THRESHOLD else "")
        )
        print(
            f"{names[row]:>22} | {float(np.quantile(live[row], 0.99)):>11.4f} | "
            f"{live_worst:>11.4f} | {history_worst:>14.4f} | {flag:>9}"
        )
    _rule()
    print(
        "A second-order expansion evaluated at a 20% move is extrapolating, and\n"
        "the error is BIASED — short gamma over-states losses — so it washes out\n"
        "neither across positions nor in the tail. A factor STRAINED in the live\n"
        "window is a candidate for routing to full revaluation; one strained only\n"
        "in history is a warning about the window you will be in next year."
    )

    print(
        "\nNOTE ON THE BACKTEST. History and realised P&L come from the same\n"
        "generator here, so a well-calibrated result is expected and proves\n"
        "only that the windowing, the order statistic and the alignment are\n"
        "right. Against real data the same code tests the market model too.\n"
        "It never tests the Taylor approximation: that sits on both sides of\n"
        "the comparison and cancels. Only revaluation can test that."
    )
    print("\n" + "=" * _WIDTH)


if __name__ == "__main__":
    run_demo()
