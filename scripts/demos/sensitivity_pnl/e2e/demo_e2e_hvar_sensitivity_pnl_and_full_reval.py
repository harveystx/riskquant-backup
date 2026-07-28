# scripts/demos/sensitivity_pnl/e2e/demo_e2e_hvar_sensitivity_pnl_and_full_reval.py
"""
End-to-end sensitivity-based VaR run.

Walks the whole pipeline for a hybrid book — some positions priced by full
revaluation, the rest carried as sensitivities — and produces the numbers a
market-risk desk actually reports: portfolio VaR and ES at several confidence
levels, per-block tail risk, and per-position component attribution.

    edge (Polars, strings)
        -> compile_sensitivity_book          strings die, units fold, CSR SoA
        -> allocate_pnl_grid                 ONE (M_total, S) f32 grid, two blocks
        -> stream scenario batches           both blocks off ONE market state
        -> compute_var_es                    lower-tail VaR / ES per group
        -> component attribution             exact decomposition of ES

The five things this demo exists to show
----------------------------------------
1.  Displacement conventions are per factor, declared, and load-bearing. The
    book below quotes equity risk per 1% relative move, rates per basis point,
    vega per vol point, and FX in log space. Feeding a log return into an
    absolute delta is the dominant silent-wrongness failure of every
    sensitivity PnL engine; here the convention is data, resolved at compile.

2.  One market state drives both PnL blocks. The full-reval stand-in and the
    sensitivity engine read the SAME shocked-levels buffer each batch. If they
    saw different draws the joint tail would be a fiction, and the
    diversification benefit reported below would be an artefact of the random
    number generator.

3.  One allocation. `allocate_pnl_grid` hands each engine a row-slice view of a
    single `(M_total, S)` f32 grid. No concatenate, no second copy, and
    overlapping position UIDs are rejected at allocation rather than
    double-counted in every tail statistic.

4.  Streaming. Scenario levels are generated into a `(N_req, B)` buffer that is
    refilled per batch, never materialised at `(N_req, S)`. The engine's scratch
    is bounded by `batch_size`, so scenario count costs only the output grid.

5.  The model's own limits are reported, not hidden. Second-order Taylor is an
    approximation that degrades with displacement magnitude, and it degrades
    asymmetrically. The run ends with a diagnostic on how far into the expansion
    the scenarios actually reach.

Everything here is synthetic and seeded. The numbers are illustrative; the
plumbing is the point.
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

NUM_SCENARIOS = 100_000
BATCH_SIZE = 500
CONFIDENCE_LEVELS = (0.95, 0.975, 0.99, 0.995)
SEED = 20260727

#: Displacement magnitude beyond which a second-order expansion is doing more
#: extrapolating than anyone should be comfortable with. Horizon dependent — a
#: 15% move is routine over ten days and a crisis over one — so it belongs to
#: the run configuration, not to the engine. Purely diagnostic either way.
TAYLOR_STRAIN_THRESHOLD = 0.15


# =========================================================================
# 1. THE RISK-FACTOR UNIVERSE
# =========================================================================
#
# `factor_id` is the caller's integer factor axis: the row index of the
# base-levels vector and of every shocked-levels batch. It is deliberately NOT
# dense from zero — real universes live in high, sparse namespaces, and the
# compiler compacts whatever it is handed rather than allocating by max id.

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

#: Today's market.
BASE_LEVELS_BY_NAME: dict[str, float] = {
    "EQ.ACME.SPOT": 184.20,
    "EQ.BOREAS.SPOT": 61.75,
    "IR.USD.2Y": 0.0398,
    "IR.USD.5Y": 0.0412,
    "IR.USD.10Y": 0.0435,
    "VOL.ACME.1Y": 0.2650,
    "VOL.BOREAS.1Y": 0.3120,
    "FX.EURUSD": 1.0840,
}

#: Annualised volatility of each factor's DISPLACEMENT, in the units that
#: factor's convention implies — relative for equity, absolute rate for the
#: curve, absolute vol for vega, log for FX.
FACTOR_SIGMA: dict[str, float] = {
    "EQ.ACME.SPOT": 0.28,
    "EQ.BOREAS.SPOT": 0.35,
    "IR.USD.2Y": 0.0085,
    "IR.USD.5Y": 0.0090,
    "IR.USD.10Y": 0.0095,
    "VOL.ACME.1Y": 0.0900,
    "VOL.BOREAS.1Y": 0.1100,
    "FX.EURUSD": 0.09,
}

#: The space each factor's greeks were differentiated in. A property of how the
#: pricing desk produced them, not of the asset class — declared, never inferred.
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

#: How many quoted units make one natural unit of displacement. Order-n
#: coefficients are folded by `quote_base ** n` at compile, so the kernels never
#: see a unit.
FACTOR_QUOTE_BASE: dict[str, float] = {
    "EQ.ACME.SPOT": QUOTE_BASE_PER_PERCENT,  # cash delta per 1% move
    "EQ.BOREAS.SPOT": QUOTE_BASE_PER_PERCENT,
    "IR.USD.2Y": QUOTE_BASE_PER_BASIS_POINT,  # DV01
    "IR.USD.5Y": QUOTE_BASE_PER_BASIS_POINT,
    "IR.USD.10Y": QUOTE_BASE_PER_BASIS_POINT,
    "VOL.ACME.1Y": QUOTE_BASE_PER_VOL_POINT,  # vega per vol point
    "VOL.BOREAS.1Y": QUOTE_BASE_PER_VOL_POINT,
    "FX.EURUSD": QUOTE_BASE_NATURAL,  # log-space sensitivity
}


def _factor_names_in_row_order() -> list[str]:
    return sorted(FACTOR_LOOKUP, key=lambda name: FACTOR_LOOKUP[name])


def level_universe_factor_ids() -> npt.NDArray[np.int32]:
    """Sorted, unique int32 factor axis. The compiler requires both."""
    return np.array(sorted(FACTOR_LOOKUP.values()), dtype=np.int32)


def base_levels() -> npt.NDArray[np.float64]:
    """Today's levels, row-aligned to `level_universe_factor_ids()`."""
    return np.ascontiguousarray(
        [BASE_LEVELS_BY_NAME[name] for name in _factor_names_in_row_order()],
        dtype=np.float64,
    )


def factor_sigma_by_row() -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        [FACTOR_SIGMA[name] for name in _factor_names_in_row_order()],
        dtype=np.float64,
    )


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
# 2. THE EDGE BOOK — human-readable, string-keyed, long format
# =========================================================================
#
# The shape a risk feed actually arrives in: one row per (position, factor),
# greeks in whatever units the pricing desk quotes them. In production this
# frame is the output of an assembler; here it is literal so the arithmetic
# stays checkable by hand.

SENSITIVITY_ROWS: list[tuple[str, str, float, float]] = [
    # position_id,       factor_id,        first_order,  second_order
    # --- ACME autocallable: long delta, short gamma, short vega ---
    ("XOTC_ACME_AUTOCALL", "EQ.ACME.SPOT", 412_000.0, -18_400.0),
    ("XOTC_ACME_AUTOCALL", "VOL.ACME.1Y", -96_000.0, 1_200.0),
    ("XOTC_ACME_AUTOCALL", "IR.USD.2Y", -18_400.0, 42.0),
    # --- BOREAS cliquet: short gamma is the whole risk ---
    ("XOTC_BOREAS_CLIQUET", "EQ.BOREAS.SPOT", -140_000.0, -46_000.0),
    ("XOTC_BOREAS_CLIQUET", "VOL.BOREAS.1Y", 154_000.0, -2_900.0),
    # --- Callable structured note: curve risk across three pillars ---
    ("STRN_USD_CALLABLE", "IR.USD.2Y", -41_200.0, 118.0),
    ("STRN_USD_CALLABLE", "IR.USD.5Y", -88_600.0, 305.0),
    ("STRN_USD_CALLABLE", "IR.USD.10Y", -134_900.0, 690.0),
    ("STRN_USD_CALLABLE", "VOL.ACME.1Y", -22_500.0, 180.0),
    # --- Quanto basket: equity risk in a foreign numeraire ---
    ("XOTC_QUANTO_BASKET", "EQ.ACME.SPOT", 96_500.0, -4_200.0),
    ("XOTC_QUANTO_BASKET", "EQ.BOREAS.SPOT", 88_100.0, -3_800.0),
    ("XOTC_QUANTO_BASKET", "FX.EURUSD", 720_000.0, -140_000.0),
    ("XOTC_QUANTO_BASKET", "IR.USD.5Y", -9_800.0, 25.0),
    # --- Vol relative value: two legs, no gamma ---
    ("RV_VOL_SPREAD", "VOL.ACME.1Y", 61_000.0, 0.0),
    ("RV_VOL_SPREAD", "VOL.BOREAS.1Y", -58_400.0, 0.0),
]

#: Mixed partials. Each unordered pair appears EXACTLY ONCE and contributes the
#: FULL mixed partial `c_ij * x_i * x_j` — no factor of one half. Vanna on the
#: equity exotics, quanto correlation risk on the basket.
CROSS_SENSITIVITY_ROWS: list[tuple[str, str, str, float]] = [
    # position_id,        factor_id_i,      factor_id_j,     cross_second_order
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
    # Carried for reporting even though it is flat today. Without it in
    # `extra_position_uids` the block's row set would change shape the day a
    # position hedges out, breaking every downstream row join.
    "XOTC_HEDGED_STUB": 8_100_006,
}

#: Positions priced by full revaluation. Disjoint from the sensitivity book —
#: `allocate_pnl_grid` enforces that rather than trusting it.
FULL_REVAL_LOOKUP: dict[str, int] = {
    "CASH_EQ_ACME": 9_200_001,
    "FWD_EQ_BOREAS": 9_200_002,
}


def sensitivity_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
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
    return diagonal, cross


def build_book() -> CompiledSensitivityBook:
    """
    Strings die here.

    Every label resolves through an explicit lookup, and every factor must exist
    in the level universe: a sensitivity to a factor outside the scenario
    universe would contribute exactly 0.0 to every scenario and pass every smoke
    test, which is the worst failure mode available to this module.
    """
    diagonal, cross = sensitivity_frames()
    return compile_sensitivity_book(
        sensitivities=diagonal,
        factor_meta=factor_meta_frame(),
        level_universe_factor_ids=level_universe_factor_ids(),
        cross_sensitivities=cross,
        position_id_lookup=POSITION_LOOKUP,
        factor_id_lookup=FACTOR_LOOKUP,
        extra_position_uids=np.array(
            [POSITION_LOOKUP["XOTC_HEDGED_STUB"]], dtype=np.int64
        ),
    )


# =========================================================================
# 3. SCENARIO GENERATION — streamed, never materialised
# =========================================================================


class ScenarioSource:
    """
    Fills a `(N_req, B)` C-contiguous f64 shocked-levels buffer, one batch at a
    time. The buffer is allocated once and refilled in place; a full
    `(N_req, S)` matrix is never built.

    Returns are a one-factor Student-t mixture: a correlated Gaussian core
    scaled by a shared inverse-chi draw. That gives fat tails AND tail
    dependence, which matters here because cross-gamma and a joint grid only
    earn their keep when several factors move together.

    Shocks are applied in each factor's own displacement space, so the levels
    handed to the engine are consistent with the conventions the book declared.
    """

    def __init__(
        self,
        *,
        num_factors: int,
        batch_size: int,
        horizon_days: float = 1.0,
        trading_days: float = 252.0,
        tail_dof: float = 6.0,
        seed: int = SEED,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty((num_factors, batch_size), dtype=np.float64, order="C")
        self.base = base_levels()
        self.scale = factor_sigma_by_row() * np.sqrt(horizon_days / trading_days)
        self.tail_dof = tail_dof
        self.cholesky = np.linalg.cholesky(self._correlation_matrix(num_factors))
        self.is_absolute = np.array(
            [
                FACTOR_CONVENTION[name] is DisplacementConvention.ABSOLUTE
                for name in _factor_names_in_row_order()
            ],
            dtype=np.bool_,
        )

    @staticmethod
    def _correlation_matrix(n: int) -> npt.NDArray[np.float64]:
        """Blocked correlation: equities together, curve tightly, vol against spot."""
        correlation = np.eye(n)
        pairs = {
            (0, 1): 0.62,  # ACME vs BOREAS spot
            (2, 3): 0.94,  # curve pillars
            (2, 4): 0.86,
            (3, 4): 0.96,
            (0, 5): -0.68,  # spot vs its own vol — the leverage effect
            (1, 6): -0.64,
            (5, 6): 0.55,  # vols together
            (0, 7): 0.18,  # equity vs FX
            (1, 7): 0.15,
            (2, 7): -0.22,  # rates vs FX
        }
        for (i, j), rho in pairs.items():
            correlation[i, j] = correlation[j, i] = rho
        # Project onto the nearest correlation matrix: a hand-written block is
        # not positive definite in general and Cholesky would fail on it.
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        correlation = (
            eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
        )
        scale = np.sqrt(np.diag(correlation))
        return correlation / np.outer(scale, scale)

    def fill(self, batch_len: int) -> npt.NDArray[np.float64]:
        """Refill and return the buffer. Only `[:, :batch_len]` is meaningful."""
        num_factors = self.base.shape[0]
        gaussian = self.cholesky @ self.rng.standard_normal((num_factors, batch_len))
        tail_mixer = np.sqrt(
            self.tail_dof / self.rng.chisquare(self.tail_dof, size=batch_len)
        )
        displacement = gaussian * tail_mixer * self.scale[:, None]

        view = self.buffer[:, :batch_len]
        absolute = self.is_absolute
        view[absolute] = self.base[absolute, None] + displacement[absolute]
        view[~absolute] = self.base[~absolute, None] * np.exp(displacement[~absolute])
        return self.buffer


# =========================================================================
# 4. FULL-REVALUATION STAND-IN
# =========================================================================


def revalue_full_block(
    *,
    shocked_levels: npt.NDArray[np.float64],
    batch_len: int,
    scenario_offset: int,
    out: npt.NDArray[np.float32],
) -> None:
    """
    Stand-in for the real valuation engine, priced off the SAME buffer.

    A cash equity position and a forward, revalued exactly rather than expanded.
    Trivial by design — what matters is that it reads the identical market state
    the sensitivity block reads and writes into its own row-slice of the shared
    grid. Swap this for `run_valuation` and nothing else in the run changes.
    """
    eq_acme = shocked_levels[0, :batch_len]
    eq_boreas = shocked_levels[1, :batch_len]
    discount_rate = shocked_levels[3, :batch_len]  # IR.USD.5Y

    acme_shares = 220_000.0  # ~USD 40 mm notional
    boreas_shares = -480_000.0  # ~USD 30 mm notional, short
    forward_strike = 63.10
    year_fraction = 0.75

    cash_pnl = acme_shares * (eq_acme - BASE_LEVELS_BY_NAME["EQ.ACME.SPOT"])
    forward_pnl = boreas_shares * (
        (eq_boreas - forward_strike) * np.exp(-discount_rate * year_fraction)
        - (BASE_LEVELS_BY_NAME["EQ.BOREAS.SPOT"] - forward_strike)
        * np.exp(-BASE_LEVELS_BY_NAME["IR.USD.5Y"] * year_fraction)
    )

    window = slice(scenario_offset, scenario_offset + batch_len)
    np.copyto(out[0, window], cash_pnl, casting="same_kind")
    np.copyto(out[1, window], forward_pnl, casting="same_kind")


# =========================================================================
# 5. THE RUN
# =========================================================================


def run_scenarios(
    *, book: CompiledSensitivityBook, num_scenarios: int, batch_size: int
) -> tuple[PnlGrid, TaylorStrainMonitor]:
    """
    Stream every scenario batch through both PnL engines.

    Returns the populated grid and a Taylor-strain monitor fed from every batch.
    """
    full_reval_uids = np.array(
        [FULL_REVAL_LOOKUP[name] for name in sorted(FULL_REVAL_LOOKUP)],
        dtype=np.int64,
    )
    # Block 0 = full reval, block 1 = sensitivities. Overlapping UIDs raise HERE,
    # before a single scenario is drawn, rather than silently double-counting in
    # every tail statistic.
    grid = allocate_pnl_grid(
        block_position_uids=[full_reval_uids, book.position_uids],
        num_scenarios=num_scenarios,
    )
    full_reval_view = grid.block_view(0)

    engine = SensitivityPnlEngine(
        book=book,
        out=grid.block_view(1),
        base_levels=base_levels(),
        batch_size=batch_size,
    )
    source = ScenarioSource(num_factors=base_levels().shape[0], batch_size=batch_size)
    strain = TaylorStrainMonitor(
        num_active_factors=book.num_active_factors,
        threshold=TAYLOR_STRAIN_THRESHOLD,
    )

    offset = 0
    while offset < num_scenarios:
        batch_len = min(batch_size, num_scenarios - offset)
        shocked_levels = source.fill(batch_len)

        revalue_full_block(
            shocked_levels=shocked_levels,
            batch_len=batch_len,
            scenario_offset=offset,
            out=full_reval_view,
        )
        engine.absorb_levels(
            shocked_levels=shocked_levels,
            batch_len=batch_len,
            scenario_offset=offset,
        )
        strain.absorb(displacement=engine.displacement_scratch, batch_len=batch_len)
        offset += batch_len

    return grid, strain


# =========================================================================
# 6. RISK METRICS
# =========================================================================


def portfolio_pnl(grid: PnlGrid) -> npt.NDArray[np.float64]:
    """
    Collapse `(M_total, S)` f32 to a `(S,)` f64 portfolio row.

    Summed in f64 explicitly: accumulating positions in f32 loses the small legs,
    and the tail is exactly where offsetting positions matter.
    """
    return grid.pnl.sum(axis=0, dtype=np.float64)


def var_es_table(
    rows: npt.NDArray[np.float64], confidence_levels=CONFIDENCE_LEVELS
) -> dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    VaR and ES per row, reported as POSITIVE LOSSES.

    `compute_var_es` is a lower-tail metric on the values it is handed. PnL is
    signed with losses negative, so the raw result is the worst PnL — a negative
    number — and the sign flip below is what turns it into a loss. Not cosmetic:
    getting it backwards reports the best scenario as the VaR.
    """
    matrix = np.ascontiguousarray(np.atleast_2d(rows), dtype=np.float64)
    table = {}
    for level in confidence_levels:
        result = compute_var_es(matrix, confidence_level=level)
        table[level] = (-result.var, -result.es)
    return table


def component_attribution(
    *, grid: PnlGrid, portfolio: npt.NDArray[np.float64], confidence_level: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Per-position component ES: each position's mean PnL over the scenarios that
    make up the PORTFOLIO's tail — not over its own tail.

    This is the decomposition that adds up. A position's own worst scenarios are
    generally not the portfolio's, so summing standalone per-position ES
    over-states the total by exactly the diversification benefit. Component ES
    sums to portfolio ES identically, which the run asserts.

    `tail_count` matches `_compute_lower_tail_parameters` exactly. If the two
    ever drift the attribution silently averages over the wrong scenario set,
    which is why the identity is asserted rather than assumed.
    """
    scenario_count = portfolio.shape[0]
    tail_count = max(1, int((1.0 - confidence_level) * scenario_count))
    tail_scenarios = np.argpartition(portfolio, tail_count - 1)[:tail_count]
    contribution = grid.pnl[:, tail_scenarios].mean(axis=1, dtype=np.float64)
    # `+ 0.0` normalises IEEE negative zero, which a flat position produces
    # under negation and which prints as '-0'.
    return -contribution + 0.0, tail_scenarios


class TaylorStrainMonitor:
    """
    Tracks how far into the expansion the run actually reaches, per active factor.

    A second-order Taylor series truncates at O(x^3). At a 1% displacement the
    truncation error is ~1e-5 relative and irrelevant; at 15% it is
    percent-level and it is BIASED — short gamma over-states losses, so the
    error neither washes out across a portfolio nor washes out in the tail.
    Reporting the reach is the difference between a VaR number and a VaR number
    that can be defended.

    Accumulated across EVERY batch rather than sampled from the last one: the
    scenarios that matter are the extreme ones, and they are exactly the ones a
    single batch is least likely to contain. Two `(Ka,)` running arrays, updated
    per batch — nothing is retained at scenario resolution.
    """

    def __init__(self, *, num_active_factors: int, threshold: float) -> None:
        self.threshold = threshold
        self.worst = np.zeros(num_active_factors, dtype=np.float64)
        self.exceedances = np.zeros(num_active_factors, dtype=np.int64)
        self.observations = 0

    def absorb(self, *, displacement: npt.NDArray[np.float32], batch_len: int) -> None:
        magnitude = np.abs(displacement[:, :batch_len].astype(np.float64))
        np.maximum(self.worst, magnitude.max(axis=1), out=self.worst)
        self.exceedances += (magnitude > self.threshold).sum(axis=1)
        self.observations += batch_len

    def rows(self, book: CompiledSensitivityBook) -> list[tuple[str, float, float]]:
        name_by_id = {factor_id: name for name, factor_id in FACTOR_LOOKUP.items()}
        return [
            (
                name_by_id[int(book.active_factor_ids[slot])],
                float(self.worst[slot]),
                float(self.exceedances[slot]) / self.observations,
            )
            for slot in range(book.num_active_factors)
        ]


# =========================================================================
# 7. REPORT
# =========================================================================

_WIDTH = 84


def _rule() -> None:
    print("-" * _WIDTH)


def run_demo() -> None:
    print("=" * _WIDTH)
    print(f"{'SENSITIVITY-BASED VaR — END TO END':^{_WIDTH}}")
    print("=" * _WIDTH)

    book = build_book()
    print(
        f"\nCompiled book: {book.num_positions} positions, "
        f"{book.num_active_factors} active factors of "
        f"{level_universe_factor_ids().shape[0]} in the universe, "
        f"nnz={book.nnz}, cross={book.nnz_cross}"
    )

    grid, strain = run_scenarios(
        book=book, num_scenarios=NUM_SCENARIOS, batch_size=BATCH_SIZE
    )
    portfolio = portfolio_pnl(grid)
    print(
        f"PnL grid: {grid.pnl.shape[0]} rows x {grid.num_scenarios:,} scenarios, "
        f"{grid.pnl.dtype}, {grid.pnl.nbytes / 1024**2:.1f} MB, "
        f"{grid.num_blocks} blocks, one allocation"
    )

    # ---- Portfolio tail risk -----------------------------------------
    print("\nPORTFOLIO TAIL RISK  (positive = loss)")
    _rule()
    print(f"{'level':>8} | {'VaR':>16} | {'ES':>16} | {'ES / VaR':>10}")
    _rule()
    for level, (var, es) in var_es_table(portfolio).items():
        print(
            f"{level:>8.1%} | {var[0]:>16,.0f} | {es[0]:>16,.0f} | "
            f"{es[0] / var[0]:>10.3f}"
        )
    _rule()
    print(
        f"{'mean PnL':>8} | {portfolio.mean():>16,.0f} | "
        f"{'std dev':>16} | {portfolio.std():>10,.0f}"
    )

    # ---- Block decomposition -----------------------------------------
    # Standalone VaR per block, then the diversification benefit. Both blocks
    # saw the same draws, so this number is real rather than an RNG artefact.
    block_rows = np.ascontiguousarray(
        np.vstack(
            [
                grid.block_view(index).sum(axis=0, dtype=np.float64)
                for index in range(grid.num_blocks)
            ]
        )
    )
    level = 0.99
    block_var, block_es = var_es_table(block_rows, (level,))[level]
    total_var, total_es = (
        column[0] for column in var_es_table(portfolio, (level,))[level]
    )
    print(f"\nBLOCK DECOMPOSITION AT {level:.1%}")
    _rule()
    print(f"{'block':>22} | {'rows':>5} | {'standalone VaR':>16} | {'ES':>16}")
    _rule()
    for index, label in enumerate(("full revaluation", "sensitivity Taylor")):
        print(
            f"{label:>22} | {grid.block_view(index).shape[0]:>5} | "
            f"{block_var[index]:>16,.0f} | {block_es[index]:>16,.0f}"
        )
    _rule()
    print(
        f"{'portfolio':>22} | {grid.pnl.shape[0]:>5} | "
        f"{total_var:>16,.0f} | {total_es:>16,.0f}"
    )
    print(
        f"{'diversification':>22} | {'':>5} | "
        f"{block_var.sum() - total_var:>16,.0f} | "
        f"{1 - total_var / block_var.sum():>15.1%} of the sum"
    )

    # ---- Component attribution ---------------------------------------
    contribution, tail_scenarios = component_attribution(
        grid=grid, portfolio=portfolio, confidence_level=level
    )
    uid_to_name = {
        uid: name for name, uid in (POSITION_LOOKUP | FULL_REVAL_LOOKUP).items()
    }
    print(
        f"\nCOMPONENT ES AT {level:.1%}  ({tail_scenarios.shape[0]:,} tail scenarios)"
    )
    _rule()
    print(f"{'position':>22} | {'component ES':>16} | {'share':>8} | {'block':>6}")
    _rule()
    for row in np.argsort(-contribution):
        uid = int(grid.trade_uids[row])
        block = "reval" if uid in FULL_REVAL_LOOKUP.values() else "sensi"
        print(
            f"{uid_to_name[uid]:>22} | {contribution[row]:>16,.0f} | "
            f"{contribution[row] / total_es:>8.1%} | {block:>6}"
        )
    _rule()
    print(f"{'total':>22} | {contribution.sum():>16,.0f} | {1.0:>8.1%} |")

    # Component ES is an exact decomposition, not an approximation. If this ever
    # drifts, the attribution is averaging over the wrong scenario set.
    assert np.isclose(contribution.sum(), total_es, rtol=1e-6), (
        contribution.sum(),
        total_es,
    )

    # ---- Model-validity diagnostic -----------------------------------
    print(
        f"\nTAYLOR STRAIN  (|displacement| over all {NUM_SCENARIOS:,} scenarios; "
        f"threshold {TAYLOR_STRAIN_THRESHOLD:.0%})"
    )
    _rule()
    print(f"{'factor':>22} | {'worst |x|':>12} | {'exceedance':>12} | {'':>8}")
    _rule()
    for name, worst, exceedance in strain.rows(book):
        flag = "STRAINED" if worst > TAYLOR_STRAIN_THRESHOLD else ""
        print(f"{name:>22} | {worst:>12.4f} | {exceedance:>12.2e} | {flag:>8}")
    _rule()
    print(
        "Second-order Taylor truncates at O(x^3) and the error is BIASED: short\n"
        "gamma over-states losses, so it washes out neither across positions nor\n"
        "in the tail. A STRAINED factor is a candidate for routing to full\n"
        "revaluation, not a reason to distrust the whole number."
    )
    print("\n" + "=" * _WIDTH)


if __name__ == "__main__":
    run_demo()
