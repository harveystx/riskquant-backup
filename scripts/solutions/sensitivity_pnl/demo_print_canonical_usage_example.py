# scripts/solutions/sensitivity_pnl/demo_print_canonical_usage_example.py
"""
Prints a complete, self-contained, copy-paste-ready example of
`riskquant.solutions.sensitivity_pnl`: build a book, compile it, price a
market, and compute VaR/ES with `riskquant.scenario_metrics`.

This script's own job is to PRINT. What it prints is verified, separately, to
actually run as a standalone file -- see
`tests/solutions/sensitivity_pnl/test_canonical_usage_example_is_runnable.py`,
which writes the exact string below to a temp file and executes it as a real
subprocess, the same way a person copying it into `my_script.py` would. There
is exactly one copy of the example text -- this module\'s
`CANONICAL_EXAMPLE_SOURCE` constant -- so what gets printed and what gets
tested can never drift apart.

Why the canonical inputs look the way they do
-----------------------------------------------
Three positions, deliberately not all alike:

    1001  delta + gamma on EQ_ACME, delta + cross-gamma on FX_EURUSD
          (a quanto-style equity position settled in a foreign currency)
    1002  delta + gamma on EQ_GLOBEX only (a plain single-factor position)
    1003  delta ONLY on EQ_ACME, no gamma -- and a partial hedge against 1001

That last detail is not decorative: 1003's negative EQ_ACME delta nets against
part of 1001's positive one, so the PRINTED portfolio VaR comes out LOWER than
1001's own standalone VaR -- a real, visible diversification effect from an
honest hedge, not a synthetic number with nothing to explain.

Why this lives under scripts/solutions, not scripts/demos
-------------------------------------------------------------
This script's entire purpose is "the canonical inputs expected by the
solutions API" -- an onboarding artifact for the facade, not a proof of the
underlying math (that is what scripts/demos/sensitivity_pnl/e2e/'s other
scripts are for). It belongs alongside `demo_e2e.py`, not next to a kernel
throughput benchmark. The embedded example imports ONLY from
`riskquant.solutions.sensitivity_pnl.api` for anything sensitivity-PnL-
specific, including `QUOTE_BASE_NATURAL` / `QUOTE_BASE_PER_PERCENT` — these
were originally pulled from `riskquant.sensitivity_pnl.conventions` directly,
which defeated the entire point of a canonical-usage example: it modelled
reaching past the facade as normal. The facade did not expose them; that was
the gap to close, not a reason to route around it. `api.py` now re-exports
them.

The one remaining import from outside `riskquant.solutions` is
`riskquant.scenario_metrics.api.var_es.compute_var_es`. `scenario_metrics` is
not nested under `riskquant.solutions`, but its own module path already has
an `.api.` segment — the same pattern `solutions.sensitivity_pnl.api` uses —
which reads as that package's own designated public surface rather than an
internal module being reached past. That is a judgement call, not a certainty;
if VaR/ES computation is meant to go through a `solutions`-tier wrapper too,
that wrapper does not exist yet and is a separate piece of work from this
file.
"""

from __future__ import annotations

__all__ = ["CANONICAL_EXAMPLE_SOURCE"]

CANONICAL_EXAMPLE_SOURCE = r'''
"""
Sensitivity PnL, VaR, and Expected Shortfall with riskquant.solutions.sensitivity_pnl
=====================================================================================

Run with:   python canonical_example.py
Requires:   riskquant (which brings numpy and polars with it) -- nothing else.

This script is self-contained: paste it into a .py file or a notebook cell and
it runs as-is, no external data, no other project files.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.solutions.sensitivity_pnl.api import (
    QUOTE_BASE_NATURAL,
    QUOTE_BASE_PER_PERCENT,
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)

# ---------------------------------------------------------------------------
# 1. Risk factor universe
# ---------------------------------------------------------------------------
# One row per risk factor: how a shock to it is expressed (its
# DisplacementConvention) and how many quoted units make one natural unit of
# that displacement (its quote_base). RELATIVE + QUOTE_BASE_PER_PERCENT means
# "sensitivities below are dollars of PnL per 1% move" -- the common desk
# convention for cash delta; LOG + QUOTE_BASE_NATURAL means "dollars per unit
# of log return," a common FX/vol convention.
FACTOR_META = pl.DataFrame(
    {
        "factor_id": ["EQ_ACME", "EQ_GLOBEX", "FX_EURUSD"],
        "displacement_convention_id": [
            int(DisplacementConvention.RELATIVE),
            int(DisplacementConvention.RELATIVE),
            int(DisplacementConvention.LOG),
        ],
        "quote_base": [
            QUOTE_BASE_PER_PERCENT,
            QUOTE_BASE_PER_PERCENT,
            QUOTE_BASE_NATURAL,
        ],
    }
)

# ---------------------------------------------------------------------------
# 2. The book: one row per (position, factor) the position is sensitive to
# ---------------------------------------------------------------------------
# first_order  = delta  (PnL per 1 quote_base unit of move)
# second_order = gamma  (PnL per (1 quote_base unit)^2 of move; 0.0 if none)
# A position absent from a factor here has NO exposure to it -- there is no
# "zero row" to write; the absence itself is the zero.
SENSITIVITIES = pl.DataFrame(
    {
        "position_id": [1001, 1001, 1002, 1003],
        "factor_id": ["EQ_ACME", "FX_EURUSD", "EQ_GLOBEX", "EQ_ACME"],
        "first_order": [12_000.0, 30_000.0, -9_000.0, -3_000.0],
        "second_order": [-150.0, 0.0, 60.0, 0.0],
    }
)

# Cross (mixed second-order) terms are optional and go in their own frame.
# Each unordered factor pair appears once; the coefficient is the FULL mixed
# partial d2V/dx_i.dx_j, not halved.
CROSS_SENSITIVITIES = pl.DataFrame(
    {
        "position_id": [1001],
        "factor_id_i": ["EQ_ACME"],
        "factor_id_j": ["FX_EURUSD"],
        "cross_second_order": [-80.0],
    }
)

# The exact row set of every result this book ever produces, ascending.
# Declaring a position here with no rows above (there is none in this
# example) is how a fully-hedged position still gets an exact-zero row
# instead of silently vanishing from the output.
POSITION_UIDS = [1001, 1002, 1003]

# ---------------------------------------------------------------------------
# 3. Compile the book against a factor axis
# ---------------------------------------------------------------------------
# The axis is the seam: every market priced through this calculator must be
# built from THIS axis (axis.shocks(...) / axis.levels(...)) so its row order
# can only ever match what the book was compiled against.
axis = FactorAxis.from_keys(FACTOR_META["factor_id"])

calculator = SensitivityPnlCalculator.compile(
    SensitivityBookSpec(
        sensitivities=SENSITIVITIES,
        factor_meta=FACTOR_META,
        factor_axis=axis,
        cross_sensitivities=CROSS_SENSITIVITIES,
        position_uids=POSITION_UIDS,
    )
)

# ---------------------------------------------------------------------------
# 4. Build a market: scenarios of correlated factor returns
# ---------------------------------------------------------------------------
# In production these scenarios might be historical (a rolling window of past
# returns) or Monte Carlo (as here). Either way the engine only ever needs a
# (num_factors, num_scenarios) array of displacements in each factor's
# declared convention -- a RELATIVE return for the two equities, a LOG return
# for the FX pair -- built in the SAME axis row order via axis.shocks(...).
NUM_SCENARIOS = 50_000
CONFIDENCE_LEVEL = 0.99

rng = np.random.default_rng(seed=2026)

# A plausible correlation structure: the two equities move together somewhat,
# and EUR/USD has a mild negative correlation with EQ_ACME (a "leverage
# effect" style relationship) and is nearly independent of EQ_GLOBEX.
correlation = np.array(
    [
        [1.00, 0.35, -0.20],
        [0.35, 1.00, 0.05],
        [-0.20, 0.05, 1.00],
    ]
)
volatility = np.array([0.05, 0.06, 0.02])  # per-scenario return volatility
cholesky = np.linalg.cholesky(correlation)

independent_draws = rng.standard_normal((axis.num_factors, NUM_SCENARIOS))
returns = (cholesky @ independent_draws) * volatility[:, None]

market = axis.shocks(np.ascontiguousarray(returns.astype(np.float32)))

# ---------------------------------------------------------------------------
# 5. Price it
# ---------------------------------------------------------------------------
# book_reused=False: this calculator is compiled and run exactly once here.
# In a ticking job -- the same book priced against a new scenario set every
# few seconds -- later calls would pass book_reused=True, and compilation
# would happen only on the very first one.
result = calculator.run(market, book_reused=False)

# ---------------------------------------------------------------------------
# 6. VaR and Expected Shortfall, per position and for the portfolio
# ---------------------------------------------------------------------------
portfolio_pnl = result.pnl.sum(axis=0, dtype=np.float64)
groups = np.ascontiguousarray(np.vstack([result.pnl.astype(np.float64), portfolio_pnl]))
risk = compute_var_es(groups, confidence_level=CONFIDENCE_LEVEL)

# compute_var_es reports the worst PnL in the tail, a NEGATIVE number for a
# loss; VaR and ES are conventionally reported as POSITIVE loss figures.
var_by_group = -risk.var
es_by_group = -risk.es

print(
    f"Compiled book: {calculator.num_positions} positions, "
    f"{calculator.num_active_factors} active factors, "
    f"{calculator.num_first_order_entries} first-order entries, "
    f"{calculator.num_cross_entries} cross entries"
)
print(f"Priced {NUM_SCENARIOS:,} scenarios\n")

level_pct = int(CONFIDENCE_LEVEL * 100)
print(f"{'position':>12} | {'VaR' + str(level_pct):>14} | {'ES' + str(level_pct):>14}")
print("-" * 46)
for uid, var, es in zip(result.position_uids, var_by_group[:-1], es_by_group[:-1]):
    print(f"{uid:>12} | {var:>14,.0f} | {es:>14,.0f}")
print("-" * 46)
print(f"{'portfolio':>12} | {var_by_group[-1]:>14,.0f} | {es_by_group[-1]:>14,.0f}")
'''.strip("\n")


def run_demo() -> None:
    width = 88
    print("=" * width)
    print("COPY EVERYTHING BETWEEN THE TWO MARKER LINES BELOW.".center(width))
    print("Paste it into a .py file or a notebook cell and run it as-is.".center(width))
    print("=" * width)
    print()
    print("# ----- COPY FROM HERE " + "-" * (width - 22))
    print(CANONICAL_EXAMPLE_SOURCE)
    print("# ----- COPY TO HERE " + "-" * (width - 20))
    print()
    print("=" * width)
    print(
        "Save it as, e.g., sensitivity_pnl_var_example.py and run:\n"
        "  python sensitivity_pnl_var_example.py\n"
        "\n"
        "Needs only riskquant installed -- it brings numpy and polars with it."
    )
    print("=" * width)


if __name__ == "__main__":
    run_demo()
