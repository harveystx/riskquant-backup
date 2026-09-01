# scripts/demos/observations/demo_hand_verifiable_observations.py

"""
Hand-verifiable demonstration of ``riskquant.observations``.

The package owns the empirical-observation contract: canonicalisation,
validation, ownership, and the rank artifacts that every rank estimator and the
stress engine share. This demo proves that contract on arrays small enough to
check by eye, and verifies each numerical claim against an independent code
path (SciPy or longhand arithmetic) rather than against the package itself.

The edge-tier ``preprocess`` module is demonstrated last, and deliberately: its
operations change the empirical sample, and the demo shows the change rather
than presenting them as cleanup.

Run:
    python scripts/demos/observations/demo_hand_verifiable_observations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riskquant.observations.compile import compile_ranks, compile_sample  # noqa: E402
from riskquant.observations.orientation import ObservationOrientation  # noqa: E402
from riskquant.observations.preprocess import (  # noqa: E402
    ImputationStrategy,
    ZeroFilterMode,
    filter_incomplete_observations,
    filter_zero_observations,
    impute_observations,
)

LINE = "=" * 78
VARIABLES_ON_ROWS = ObservationOrientation.VARIABLES_ON_ROWS
OBSERVATIONS_ON_ROWS = ObservationOrientation.OBSERVATIONS_ON_ROWS


def _banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def demo_orientation_is_declared_never_inferred() -> None:
    _banner("ORIENTATION  --  declared, never inferred")

    panel = np.array([[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]])
    print("A (2, 4) array is unambiguous only because the caller says which")
    print("axis carries the variables. The package never guesses.\n")
    print(f"raw = \n{panel}")

    by_rows = compile_sample(panel, orientation=VARIABLES_ON_ROWS)
    by_columns = compile_sample(
        np.ascontiguousarray(panel.T), orientation=OBSERVATIONS_ON_ROWS
    )

    print(f"\nVARIABLES_ON_ROWS    -> shape {by_rows.observations.shape}")
    print(f"OBSERVATIONS_ON_ROWS -> shape {by_columns.observations.shape}   (transposed once)")
    print(
        "both give the same canonical matrix = "
        f"{np.array_equal(by_rows.observations, by_columns.observations)}"
    )


def demo_canonical_layout_and_ownership() -> None:
    _banner("CANONICAL LAYOUT AND OWNERSHIP")

    panel = np.asfortranarray([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
    print(f"input: dtype={panel.dtype}, C-contiguous={panel.flags.c_contiguous}")

    sample = compile_sample(panel, orientation=VARIABLES_ON_ROWS)
    print(
        f"compiled: dtype={sample.observations.dtype}, "
        f"C-contiguous={sample.observations.flags.c_contiguous}, "
        f"writeable={sample.observations.flags.writeable}"
    )

    print("\nSnapshot semantics: the compiled sample owns a copy.")
    print(f"  compiled array is the caller's array = "
          f"{sample.observations is panel}")
    original = sample.observations.copy()
    panel[0, 0] = 1e9
    print("  caller mutated its array afterwards")
    print(f"  compiled sample unchanged = {np.array_equal(sample.observations, original)}")

    print("\nThe compiled arrays cannot be written through:")
    try:
        sample.observations[0, 0] = 0.0
    except ValueError as error:
        print(f"  write rejected: {error}")


def demo_sum_of_squares_is_the_degeneracy_gate() -> None:
    _banner("SUM OF SQUARES  --  the degeneracy gate, computed without a (K,S) scratch")

    panel = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    print(f"x = {panel[0]}   (mean 2, deviations -1, 0, +1)")
    print(f"y = {panel[1]}   (constant)")

    longhand_x = float(((panel[0] - panel[0].mean()) ** 2).sum())
    longhand_y = float(((panel[1] - panel[1].mean()) ** 2).sum())
    sample = compile_sample(panel, orientation=VARIABLES_ON_ROWS)

    print(f"\nlonghand   = [{longhand_x}, {longhand_y}]")
    print(f"package    = {sample.sum_of_squares.tolist()}")
    print(f"shape      = {sample.sum_of_squares.shape}   (K,), never (K, S)")
    print("\nThe second variable is EXACTLY zero, not merely small. The gate is")
    print("an exact-zero test, not a tolerance: a variance of 1e-30 is")
    print("ill-conditioned, and deciding where that starts is caller policy.")


def demo_rank_artifacts() -> None:
    _banner("RANK ARTIFACTS  --  one sort, three consumers")

    panel = np.array([[10.0, 30.0, 20.0, 20.0, 40.0]])
    panel = np.vstack([panel, np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])])
    print(f"x = {panel[0]}   (the two 20s tie)")
    print(f"y = {panel[1]}")

    sample = compile_sample(panel, orientation=VARIABLES_ON_ROWS)
    ranks = compile_ranks(sample)

    print(f"\naverage ranks of x = {ranks.ranks[0]}")
    print(f"scipy.stats.rankdata = {rankdata(panel[0], method='average')}")
    print(
        "  identical = "
        f"{np.array_equal(ranks.ranks[0], rankdata(panel[0], method='average'))}"
    )
    print("  the tied pair spans ranks 2 and 3, so both receive 2.5")

    print(f"\nsort_order of x = {ranks.sort_order[0]}  dtype={ranks.sort_order.dtype}")
    print("  a stable ascending argsort. This is the artifact the stress engine")
    print("  consumes; it carries no dependence semantics.")

    tie_pairs_longhand = 1.0  # one tied pair among the two 20s
    print(f"\ntie_pairs of x = {ranks.tie_pairs[0]}   longhand = {tie_pairs_longhand}")
    print("  sum over tie blocks of t(t-1)/2; one block of size 2 gives 1.")

    print("\nOne sort serves ranks, sort_order and tie_pairs together.")


def demo_validation_fails_loud() -> None:
    _banner("VALIDATION  --  fails loud, with a cell address")

    cases = (
        ("1-D input", np.arange(6.0)),
        ("too few variables", np.zeros((1, 5))),
        ("too few observations", np.zeros((3, 1))),
        ("NaN at (1, 2)", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]])),
        ("+Inf", np.array([[1.0, 2.0, 3.0], [4.0, np.inf, 6.0]])),
    )
    for label, bad in cases:
        try:
            compile_sample(bad, orientation=VARIABLES_ON_ROWS)
            print(f"  {label:<22} ACCEPTED  <- unexpected")
        except (TypeError, ValueError) as error:
            print(f"  {label:<22} rejected: {str(error)[:78]}")

    print("\nNothing is repaired, imputed or coerced to make execution succeed.")


def demo_preprocess_changes_the_sample() -> None:
    _banner("PREPROCESS (EDGE)  --  every operation changes the empirical sample")

    panel = np.array(
        [
            [1.0, 0.0, 2.0, np.nan, 3.0],
            [4.0, 0.0, 0.0, 5.0, 6.0],
        ]
    )
    print(f"raw =\n{panel}")

    retained, index = filter_incomplete_observations(
        panel, orientation=VARIABLES_ON_ROWS
    )
    print(f"\ncomplete-case selection keeps observations {index.tolist()}")
    print(f"  -> {retained.shape[1]} of {panel.shape[1]} observations survive")
    print("  the retained index is returned so dates can be realigned")

    filtered, kept = filter_zero_observations(
        retained,
        orientation=VARIABLES_ON_ROWS,
        mode=ZeroFilterMode.DROP_IF_ANY_ZERO,
        zero_tolerance=0.0,
    )
    print(f"\nDROP_IF_ANY_ZERO keeps {kept.tolist()} -> {filtered.shape[1]} observations")
    print("  aggressive by construction: with K variables each independently")
    print("  zero with probability p, expected retention is (1-p)^K")

    imputed = impute_observations(
        panel,
        orientation=VARIABLES_ON_ROWS,
        strategy=ImputationStrategy.VARIABLE_MEAN,
    )
    observed_mean = float(np.nanmean(panel[0]))
    print(f"\nmean imputation fills the NaN with {imputed[0, 3]:.6f}")
    print(f"  observed mean of x = {observed_mean:.6f}   (longhand check)")
    print("  this inserts a synthetic observation at the marginal centre, which")
    print("  mechanically shrinks the estimated covariance of that variable")

    print("\nNone of these runs automatically anywhere in the library.")


def main() -> None:
    demo_orientation_is_declared_never_inferred()
    demo_canonical_layout_and_ownership()
    demo_sum_of_squares_is_the_degeneracy_gate()
    demo_rank_artifacts()
    demo_validation_fails_loud()
    demo_preprocess_changes_the_sample()
    print(f"\n{LINE}\nobservations contract demonstrated.\n{LINE}")


if __name__ == "__main__":
    main()
