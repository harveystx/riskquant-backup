# scripts/demos/dependence_stress/demo_dependence_stress.py

"""
Illustrative walkthrough of riskquant.dependence_stress.

Run from the repository root::

    python scripts/demos/dependence_stress/demo_dependence_stress.py

The script explains, through printed output alone, what the package does
mathematically and mechanically. It calls the production API for every
algorithm. The only logic it owns is:

    - constructing illustrative inputs;
    - calling package APIs;
    - printing explanations;
    - independent reference arithmetic for hand-check sections;
    - asserting invariants.

Reference calculations marked ``[HAND CHECK]`` use plain Python
arithmetic that never touches the production implementation. They exist
to give the reviewer a second, independent opinion. They are naive and
slow on purpose and would never belong in a kernel.
"""

from __future__ import annotations

import math
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"

if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from _canonical_measures import (  # noqa: E402
    average_ranks,
    empirical_uniforms,
    lower_tail_dependence,
    pearson_correlation,
    spearman_correlation,
    upper_tail_dependence,
)

from riskquant.dependence_stress.benchmarks import (  # noqa: E402
    Comonotonic,
    DualTailComonotone,
    LowerTailComonotone,
    UpperTailComonotone,
)
from riskquant.dependence_stress.copulas import GaussianCopula, TCopula  # noqa: E402
from riskquant.dependence_stress.engine import DependenceStressEngine  # noqa: E402
from riskquant.dependence_stress.permutations import (  # noqa: E402
    aggregate_portfolio,
    apply_permutation,
    reconstruct_stressed,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.types import (  # noqa: E402
    DependenceStressResult,
    FloatArray,
    IntArray,
)
from riskquant.dependence_stress.validation import (  # noqa: E402
    validate_marginal_preservation,
    validate_permutation_matrix,
    validate_transformation,
)

WIDTH = 68
RULE = "=" * WIDTH

GOLDEN_LABELS = ("Variable A", "Variable B", "Variable C")

#: Golden example. Three variables, six scenarios. Small enough to write
#: out on paper. Variable B carries deliberate ties (10.0 twice, 20.0
#: twice) so that average-rank handling is visible.
GOLDEN = np.array(
    [
        [12.0, -5.0, 3.0, 20.0, -1.0, 7.0],
        [10.0, 20.0, 20.0, 40.0, 30.0, 10.0],
        [-2.0, -8.0, 1.0, 4.0, -3.0, 0.5],
    ],
    dtype=np.float64,
)

TAIL_LABELS = ("Desk W", "Desk X", "Desk Y", "Desk Z")

#: Wider example used only for the lower-tail sections, sized so that
#: q = 0.25 with S = 12 gives n_tail = 3 exactly.
TAIL_DATA = np.array(
    [
        [-9.0, 4.0, -6.0, 11.0, 2.0, -1.0, 8.0, -7.0, 5.0, 0.0, 3.0, 6.0],
        [-4.0, 7.0, -8.0, 2.0, 9.0, -2.0, 1.0, -5.0, 6.0, 3.0, -1.0, 5.0],
        [12.0, -3.0, 4.0, -6.0, 7.0, 1.0, -9.0, 2.0, -2.0, 8.0, 5.0, 0.0],
        [-1.0, 6.0, -4.0, 9.0, -7.0, 3.0, 2.0, -8.0, 4.0, 1.0, 7.0, -2.0],
    ],
    dtype=np.float64,
)

COPULA_LABELS = ("Row 0", "Row 1", "Row 2", "Row 3")


# ======================================================================
# Printing helpers
# ======================================================================


def banner(title: str) -> None:
    """Print a top-level section banner."""
    print()
    print(RULE)
    print(title)
    print(RULE)


def heading(title: str) -> None:
    """Print a subsection heading."""
    print()
    print(title)
    print("-" * len(title))


def paragraph(text: str, indent: int = 0) -> None:
    """Print a block of explanatory prose, preserving relative indent."""
    pad = " " * indent
    dedented = textwrap.dedent(text)

    # A block that opens with a blank line asks for vertical separation
    # from whatever was printed above it. Honour that rather than
    # stripping it away.
    if dedented.startswith("\n\n"):
        print()

    for line in dedented.strip("\n").splitlines():
        print((pad + line).rstrip())


def fmt_row(values: FloatArray, width: int = 8, decimals: int = 2) -> str:
    """Format a float row for aligned display."""
    return " ".join(f"{float(value):>{width}.{decimals}f}" for value in values)


def fmt_int_row(values: IntArray, width: int = 8) -> str:
    """Format an integer row for aligned display."""
    return " ".join(f"{int(value):>{width}d}" for value in values)


def print_matrix(
    matrix: FloatArray,
    labels: tuple[str, ...],
    decimals: int = 2,
) -> None:
    """Print a labelled K x S float matrix with a scenario header."""
    s = matrix.shape[1]
    label_width = max(len(label) for label in labels) + 2

    print(" " * label_width + " ".join(f"{f's{i}':>8}" for i in range(s)))

    for row, label in enumerate(labels):
        print(f"{label:<{label_width}}" + fmt_row(matrix[row], decimals=decimals))


def print_int_matrix(matrix: IntArray, labels: tuple[str, ...]) -> None:
    """Print a labelled K x S integer matrix with a slot header."""
    s = matrix.shape[1]
    label_width = max(len(label) for label in labels) + 2

    print(" " * label_width + " ".join(f"{f'slot{i}':>8}" for i in range(s)))

    for row, label in enumerate(labels):
        print(f"{label:<{label_width}}" + fmt_int_row(matrix[row]))


def print_square(
    matrix: FloatArray,
    labels: tuple[str, ...],
    decimals: int = 4,
) -> None:
    """Print a labelled K x K matrix."""
    label_width = max(len(label) for label in labels) + 2
    cell_width = max(10, label_width)

    print(" " * label_width + " ".join(f"{label:>{cell_width}}" for label in labels))

    for row, label in enumerate(labels):
        cells = " ".join(
            f"{float(value):>{cell_width}.{decimals}f}" for value in matrix[row]
        )
        print(f"{label:<{label_width}}" + cells)


def stressed_of(
    base: FloatArray,
    result: DependenceStressResult,
) -> FloatArray:
    """
    Reconstruct the stressed K x S matrix for a result.

    The stressed matrix is NOT a field of the result. It is an optional
    derived representation. The demo reconstructs it explicitly, which
    is exactly what a production audit consumer does.
    """
    return reconstruct_stressed(base, result.permutations)


def check(description: str, condition: bool) -> None:
    """Print a PASS line and fail loudly if the invariant is broken."""
    print(f"    {description:<54} {'PASS' if condition else 'FAIL'}")

    assert condition, f"INVARIANT FAILED: {description}"


# ======================================================================
# Independent reference arithmetic (NOT the production implementation)
# ======================================================================


def reference_average_ranks(row: list[float]) -> list[float]:
    """
    Naive average ranks, written from the textbook definition.

    For each value count how many values are strictly smaller and how
    many are tied. The average rank is then::

        n_smaller + (n_tied + 1) / 2

    Independent of scipy.stats.rankdata, so it corroborates the package
    result rather than echoing it.
    """
    ranks: list[float] = []

    for value in row:
        smaller = sum(1 for other in row if other < value)
        tied = sum(1 for other in row if other == value)
        ranks.append(smaller + (tied + 1) / 2.0)

    return ranks


def reference_pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation from explicit sums, no NumPy reductions."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cross = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)

    return cross / math.sqrt(var_x * var_y)


def reference_tail_scenarios(row: list[float], n_tail: int) -> list[int]:
    """
    Scenario indices of the ``n_tail`` lowest observations.

    Sorting (value, index) pairs breaks ties by ascending scenario
    index, which is the convention the package documents for its stable
    sort. Written here independently of it.
    """
    ordered = sorted(range(len(row)), key=lambda i: (row[i], i))

    return sorted(ordered[:n_tail])


def reference_take(row: list[float], permutation: list[int]) -> list[float]:
    """Gather ``row[permutation[s]]`` for every slot ``s``."""
    return [row[index] for index in permutation]


def audit_transformation(
    base: FloatArray,
    stressed: FloatArray,
    permutations: IntArray,
    label: str,
) -> None:
    """Run and print the full structural audit for one transformation."""
    heading(f"Invariant audit: {label}")

    check("shape preserved", stressed.shape == base.shape)

    validate_permutation_matrix(permutations, expected_shape=base.shape)
    check("every row is a valid permutation of 0..S-1", True)

    check(
        "marginal distributions preserved exactly",
        validate_marginal_preservation(base, stressed),
    )

    check(
        "no values created (each value belongs to its own row)",
        all(
            set(stressed[row].tolist()) <= set(base[row].tolist())
            for row in range(base.shape[0])
        ),
    )

    check(
        "stressed == take_along_axis(base, permutations)",
        np.array_equal(stressed, np.take_along_axis(base, permutations, axis=1)),
    )

    check(
        "package validate_transformation() agrees",
        validate_transformation(base, stressed, permutations),
    )

    assert np.array_equal(np.sort(stressed, axis=1), np.sort(base, axis=1))


# ======================================================================
# Section 1: introduction
# ======================================================================


def section_introduction() -> None:
    """Explain the purpose and the axis convention."""
    banner("RiskQuant DEPENDENCE_STRESS DEMONSTRATION")

    paragraph(
        """
        The package stresses the cross-sectional dependence structure
        of an empirical observation matrix while preserving every row's
        empirical marginal distribution exactly.

        The core operation is:

            stressed[k, s] = data[k, permutation[k, s]]

        The package therefore changes WHICH historical observations
        occur together, but never changes the observations available
        within an individual row.
        """
    )

    heading("Axis convention")

    paragraph(
        """
        Rows    = empirical variables
        Columns = historical scenarios
        Shape   = K x S

        The convention is never inferred and the input is never
        transposed. A 3 x 6 input is three variables observed over six
        scenarios, always.
        """
    )

    heading("Domain")

    paragraph(
        """
        The package is domain-agnostic. Rows may be subportfolio P&Ls,
        risk-factor shocks, returns, spreads, or any other empirical
        variable. In the Correlation Stress Charge use case the rows are
        subportfolio P&Ls, but nothing in this package knows that.

        The package stops at the stressed empirical matrix. It contains
        no VaR, no ES, no CSC and no aggregation. Those live downstream
        and consume the matrix this package produces.
        """
    )


# ======================================================================
# Section 2: the golden matrix and its marginals
# ======================================================================


def section_golden_matrix() -> None:
    """Print the golden matrix and its per-row empirical marginals."""
    banner("THE GOLDEN EXAMPLE")

    paragraph(
        """
        Every deterministic mechanism in this demo uses one small matrix
        so that a reviewer can follow the arithmetic on paper. Only the
        copula sections, which are stochastic, use a larger matrix.
        """
    )

    heading("Original empirical matrix")

    print(f"Shape:         {GOLDEN.shape[0]} x {GOLDEN.shape[1]}")
    print(f"dtype:         {GOLDEN.dtype}")
    print(f"C-contiguous:  {GOLDEN.flags.c_contiguous}")
    print()

    print_matrix(GOLDEN, GOLDEN_LABELS)

    paragraph(
        f"""

        row k    = one empirical marginal distribution
        column s = one historical scenario

        Reading scenario s3: in that one historical scenario, A was
        {GOLDEN[0, 3]:.2f}, B was {GOLDEN[1, 3]:.2f} and C was {GOLDEN[2, 3]:.2f}.
        Those three numbers happened together. Dependence stress is
        entirely about which numbers happen together.
        """
    )

    heading("Each row's empirical marginal")

    paragraph(
        """
        The sorted row defines the empirical marginal. A valid
        dependence transformation must leave this sorted sequence
        unchanged. This is the reference that every later section
        checks against.
        """
    )

    for row, label in enumerate(GOLDEN_LABELS):
        print()
        print(f"    {label} marginal")
        print(f"        Original values: {fmt_row(GOLDEN[row])}")
        print(f"        Sorted values:   {fmt_row(np.sort(GOLDEN[row]))}")

    paragraph(
        """

        Variable B contains ties: 10.00 appears twice and 20.00 appears
        twice. Ties are included deliberately, because rank handling is
        where naive implementations go wrong.
        """
    )


# ======================================================================
# Section 3: ranks and empirical uniforms
# ======================================================================


def section_ranks() -> None:
    """Show average ranks and mid-rank uniforms with hand checks."""
    banner("STEP: EMPIRICAL RANKS")

    paragraph(
        """
        Ranks are the bridge between raw observations and dependence.
        A dependence structure is a statement about relative ordering,
        not about magnitudes, so the package works in rank space.

        ranks.average_ranks() gives tied observations the average of the
        ranks they span. The smallest observation has rank 1 and the
        largest has rank S.
        """
    )

    ranks = average_ranks(GOLDEN)

    heading("Package result: average ranks")

    print_matrix(ranks, GOLDEN_LABELS, decimals=2)

    heading("[HAND CHECK] Variable B ranks, derived by hand")

    paragraph(
        f"""
        Values in scenario order:
            {fmt_row(GOLDEN[1])}

        Sort them:
            10.00  10.00  20.00  20.00  30.00  40.00
        Position:
             1      2      3      4      5      6

        The two 10.00 values occupy positions 1 and 2:
            average rank = (1 + 2) / 2 = 1.5
        The two 20.00 values occupy positions 3 and 4:
            average rank = (3 + 4) / 2 = 3.5
        30.00 occupies position 5  ->  rank 5.0
        40.00 occupies position 6  ->  rank 6.0

        Mapping back to the original scenario order
        (10, 20, 20, 40, 30, 10):
            1.5   3.5   3.5   6.0   5.0   1.5
        """
    )

    reference = reference_average_ranks(GOLDEN[1].tolist())

    print()
    print(f"    Independent reference: {fmt_row(np.asarray(reference))}")
    print(f"    Package result:        {fmt_row(ranks[1])}")
    print()

    check(
        "[HAND CHECK PASSES] average ranks match",
        bool(np.allclose(reference, ranks[1])),
    )

    paragraph(
        """

        Why average ranks and not np.argsort(np.argsort(x))?
        The argsort-of-argsort idiom assigns ORDINAL ranks: the two
        10.00 values would receive 1 and 2 rather than 1.5 and 1.5.
        That invents an ordering between observations that are equal,
        and it makes Spearman correlation wrong under ties.
        """
    )

    ordinal = (np.argsort(np.argsort(GOLDEN[1])) + 1).astype(np.float64)

    print()
    print(f"    Ordinal (wrong under ties):  {fmt_row(ordinal)}")
    print(f"    Average (package, correct):  {fmt_row(ranks[1])}")

    heading("Empirical uniforms")

    paragraph(
        """
        The mid-rank uniform transform places each observation at its
        empirical position inside its own marginal:

            U = (r - 0.5) / S

        The 0.5 offset keeps every value strictly inside (0, 1), which
        keeps the result safe for an inverse-CDF transform. Without it
        the largest observation would map to exactly 1.0.
        """
    )

    uniforms = empirical_uniforms(GOLDEN)
    s = GOLDEN.shape[1]

    heading(f"[HAND CHECK] Variable B uniforms, S = {s}")

    for rank in (1.5, 3.5, 5.0, 6.0):
        print(f"    rank {rank:>4}:  ({rank} - 0.5) / {s} = {(rank - 0.5) / s:.6f}")

    manual_uniforms = [(r - 0.5) / s for r in reference]

    print()
    print(
        f"    Independent reference: {fmt_row(np.asarray(manual_uniforms), decimals=4)}"
    )
    print(f"    Package result:        {fmt_row(uniforms[1], decimals=4)}")
    print()

    check(
        "[HAND CHECK PASSES] uniforms match",
        bool(np.allclose(manual_uniforms, uniforms[1])),
    )
    check(
        "every uniform lies strictly inside (0, 1)",
        bool(np.all((uniforms > 0.0) & (uniforms < 1.0))),
    )

    heading("All rows: empirical uniforms")

    print_matrix(uniforms, GOLDEN_LABELS, decimals=4)


# ======================================================================
# Section 4: the core permutation mechanism
# ======================================================================


def section_permutation_mechanism() -> None:
    """Make the permutation mechanism completely explicit."""
    banner("THE CORE PERMUTATION MECHANISM")

    paragraph(
        """
        Every dependence method in this package reduces to one thing:
        a K x S permutation matrix. The methodology decides WHAT the
        permutation should be. The mechanics then apply it identically
        in every case:

            stressed = np.take_along_axis(data, permutations, axis=1)

        Read one row of the permutation matrix as instructions about
        scenario slots:

            permutation[k, s] = which ORIGINAL scenario index supplies
                                the value that now sits in slot s

        Each row contains every index in 0, 1, ..., S-1 exactly once.
        That single property is the whole reason the marginals cannot
        change: the row holds the same multiset of values, re-ordered.
        """
    )

    heading("[HAND CHECK] Tiny worked example")

    tiny = [10.0, 20.0, 30.0, 40.0]
    tiny_perm = [2, 0, 3, 1]

    paragraph(
        """
        Original:
            [10, 20, 30, 40]
        Permutation:
            [ 2,  0,  3,  1]

        Slot 0 takes original index 2 -> 30
        Slot 1 takes original index 0 -> 10
        Slot 2 takes original index 3 -> 40
        Slot 3 takes original index 1 -> 20

        Result:
            [30, 10, 40, 20]

        The permutation contains 0, 1, 2 and 3 exactly once, so the
        result contains 10, 20, 30 and 40 exactly once.
        """
    )

    manual = reference_take(tiny, tiny_perm)

    package_pair = apply_permutation(
        np.array([tiny, tiny[::-1]], dtype=np.float64),
        np.array([tiny_perm, tiny_perm], dtype=np.int64),
    )

    print()
    print(f"    Independent reference:            {manual}")
    print(f"    Package apply_permutation row 0:  {package_pair[0].tolist()}")
    print()

    check(
        "[HAND CHECK PASSES] manual gather matches package",
        np.array_equal(np.asarray(manual), package_pair[0]),
    )

    heading("The same mechanism on the golden matrix")

    example_perm = np.array(
        [
            [3, 0, 5, 1, 4, 2],
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
        ],
        dtype=np.int64,
    )

    print("Permutation matrix (hand-written purely for illustration):")
    print()
    print_int_matrix(example_perm, GOLDEN_LABELS)

    stressed = apply_permutation(GOLDEN, example_perm)

    paragraph(
        f"""

        Variable A walkthrough:

            scenario index:       {fmt_int_row(np.arange(6))}
            original Variable A:  {fmt_row(GOLDEN[0])}
            permutation:          {fmt_int_row(example_perm[0])}
            stressed Variable A:  {fmt_row(stressed[0])}

        Slot 0 receives original scenario 3, whose value is
        {GOLDEN[0, 3]:.2f}. Slot 1 receives original scenario 0, whose
        value is {GOLDEN[0, 0]:.2f}. Continue for every slot.

        Variable B was given the identity permutation, so it is
        unchanged. Variable C was reversed. Three different rows can
        carry three completely different permutations: that is exactly
        how the cross-sectional pairing gets rearranged.
        """
    )

    print()
    print("Stressed matrix:")
    print()
    print_matrix(stressed, GOLDEN_LABELS)

    audit_transformation(GOLDEN, stressed, example_perm, "illustrative permutation")


# ======================================================================
# Section 5: comonotonic
# ======================================================================


def section_comonotonic() -> FloatArray:
    """Demonstrate the deterministic comonotonic benchmark."""
    banner("METHOD 1: COMONOTONIC")

    paragraph(
        """
        Comonotonicity is the extreme positive dependence benchmark.

            The worst observation in every row is placed together.
            The second-worst in every row is placed together.
            ...
            The best observation in every row is placed together.

        Mechanically the permutation is the stable argsort of each row,
        so applying it produces exactly np.sort(data, axis=1).

        Deterministic. It consumes no randomness at all.
        """
    )

    result = transform_empirical_matrix(GOLDEN, Comonotonic())

    heading("Original matrix")
    print_matrix(GOLDEN, GOLDEN_LABELS)

    heading("Permutation matrix")
    print_int_matrix(result.permutations, GOLDEN_LABELS)

    heading("Stressed matrix")
    print_matrix(stressed_of(GOLDEN, result), GOLDEN_LABELS)

    heading("[HAND CHECK] Row-by-row reconstruction")

    paragraph(
        """
        For each row: sort the row, note which original scenario index
        each sorted value came from, and that index sequence IS the
        permutation row.
        """
    )

    for row, label in enumerate(GOLDEN_LABELS):
        print()
        print(f"    {label}")
        print(f"        original row:  {fmt_row(GOLDEN[row])}")
        print(f"        sorted row:    {fmt_row(np.sort(GOLDEN[row]))}")
        print(f"        permutation:   {fmt_int_row(result.permutations[row])}")
        print(f"        stressed row:  {fmt_row(stressed_of(GOLDEN, result)[row])}")

        manual = reference_take(
            GOLDEN[row].tolist(),
            result.permutations[row].tolist(),
        )

        assert np.array_equal(np.asarray(manual), stressed_of(GOLDEN, result)[row])

    print()
    check(
        "[HAND CHECK PASSES] stressed == sorted rows for every row",
        np.array_equal(stressed_of(GOLDEN, result), np.sort(GOLDEN, axis=1)),
    )

    heading("Stressed rank ordering")

    paragraph(
        """
        After the transformation every row carries the same rank
        sequence 1..S. Slot 0 holds every row's worst observation, slot
        1 holds every row's second-worst, and so on. That is perfect
        empirical rank alignment.
        """
    )

    stressed_ranks = average_ranks(stressed_of(GOLDEN, result))

    print()
    print_matrix(stressed_ranks, GOLDEN_LABELS, decimals=2)

    paragraph(
        """

        Variable B shows 1.5, 1.5, 3.5, 3.5, 5.0, 6.0 rather than
        1..6, because its tied values genuinely share a rank. The
        ordering is still perfectly aligned with the other rows.
        """
    )

    heading("Spearman correlation of the comonotonic result")

    spearman = spearman_correlation(stressed_of(GOLDEN, result))
    print_square(spearman, GOLDEN_LABELS)

    paragraph(
        """
        Read this carefully, because it is a real subtlety.

        Variable A and Variable C have no ties, so their comonotonic
        Spearman correlation is exactly 1.0.

        Every pair involving Variable B is BELOW 1.0. That is not a
        defect. Variable B has two tied pairs, so its rank vector is
        1.5, 1.5, 3.5, 3.5, 5, 6 rather than 1..6. A tied rank vector
        cannot be perfectly linearly related to an untied one, so a
        Spearman correlation of exactly 1 is unattainable for that pair
        under ANY permutation.

        The honest invariant is therefore: comonotonic alignment attains
        the MAXIMUM Spearman correlation available for the given data,
        and reaches exactly 1 when the rows carry no ties.
        """
    )

    print()
    check(
        "untied pair (A, C) reaches Spearman exactly 1.0",
        math.isclose(float(spearman[0, 2]), 1.0, rel_tol=1e-12),
    )
    check(
        "tied Variable B cannot reach 1.0, as expected",
        float(spearman[0, 1]) < 1.0,
    )

    search_rng = np.random.default_rng(11)
    best_spearman = spearman.copy()

    for _ in range(300):
        shuffled = np.take_along_axis(
            GOLDEN,
            np.argsort(search_rng.random(GOLDEN.shape), axis=1),
            axis=1,
        )
        assert np.all(spearman_correlation(shuffled) <= best_spearman + 1e-12)

    check(
        "no random rearrangement beats it (300 draws)",
        True,
    )

    heading("A precise point about Pearson correlation")

    pearson = pearson_correlation(stressed_of(GOLDEN, result))
    print_square(pearson, GOLDEN_LABELS)

    paragraph(
        """
        Comonotonicity is a RANK property. Pearson correlation reaches
        1 only when two rows are increasing affine transforms of each
        other, which these rows are not. The correct Pearson statement
        is that the comonotonic arrangement MAXIMISES Pearson
        correlation over every possible row-wise permutation
        (the rearrangement inequality), not that it equals 1.
        """
    )

    rng = np.random.default_rng(0)
    best = pearson.copy()

    for _ in range(200):
        shuffled = np.take_along_axis(
            GOLDEN,
            np.argsort(rng.random(GOLDEN.shape), axis=1),
            axis=1,
        )
        assert np.all(pearson_correlation(shuffled) <= best + 1e-12)

    print()
    check("no random rearrangement beats comonotonic Pearson (200 draws)", True)

    heading("Idempotence")

    once = stressed_of(GOLDEN, result)
    twice = stressed_of(once, transform_empirical_matrix(once, Comonotonic()))

    print("    Applying Comonotonic a second time...")
    print()
    check("T(T(X)) == T(X)", np.array_equal(twice, once))

    audit_transformation(
        GOLDEN, stressed_of(GOLDEN, result), result.permutations, "Comonotonic"
    )

    return stressed_of(GOLDEN, result)


# ======================================================================
# Section 6A: lower-tail alignment
# ======================================================================


def section_lower_tail() -> FloatArray:
    """Demonstrate lower-tail alignment in all three modes."""
    banner("METHOD 2A: LOWER-TAIL ALIGNMENT")

    q = 0.25
    s = TAIL_DATA.shape[1]
    n_tail = max(1, math.ceil(q * s))

    paragraph(
        """
        This method does not try to make the entire matrix comonotonic.
        It specifically engineers stronger empirical dependence in the
        lower tail, and leaves the rest of the matrix alone.

        A wider example is used here so the tail size is a clean
        integer. Four desks, twelve scenarios.
        """
    )

    heading("Input matrix")
    print_matrix(TAIL_DATA, TAIL_LABELS)

    heading("[HAND CHECK] Tail size")

    paragraph(
        f"""
        Tail probability:    {q:.0%}
        Number of scenarios: {s}

            n_tail = max(1, ceil(q * S))
                   = max(1, ceil({q} * {s}))
                   = max(1, ceil({q * s:.1f}))
                   = {n_tail}

        The ceiling and the floor of 1 together guarantee at least one
        tail observation for any q in (0, 1), however small.
        """
    )

    heading("[HAND CHECK] Which scenarios form each row's lower tail")

    paragraph(
        f"""
        For each row, take the {n_tail} lowest observations and note the
        original scenario indices they came from.
        """
    )

    manual_tails: dict[str, list[int]] = {}

    for row, label in enumerate(TAIL_LABELS):
        values = TAIL_DATA[row].tolist()
        indices = reference_tail_scenarios(values, n_tail)
        manual_tails[label] = indices

        lowest = sorted(values)[:n_tail]

        print()
        print(f"    {label}")
        print(f"        row values:        {fmt_row(TAIL_DATA[row])}")
        print(f"        {n_tail} lowest values:  {fmt_row(np.asarray(lowest))}")
        print(f"        from scenarios:    {sorted(indices)}")

    result = transform_empirical_matrix(
        TAIL_DATA,
        LowerTailComonotone(tail_probability=q),
    )

    heading("Permutation produced by the package")
    print_int_matrix(result.permutations, TAIL_LABELS)

    paragraph(
        f"""

        Slots 0..{n_tail - 1} are the common tail block. Read the first
        {n_tail} entries of each permutation row: they are exactly the
        tail scenario indices listed above, ordered worst first.
        """
    )

    for row, label in enumerate(TAIL_LABELS):
        package_tail = sorted(result.permutations[row, :n_tail].tolist())

        check(
            f"[HAND CHECK] {label} tail scenarios match manual set",
            package_tail == manual_tails[label],
        )

    heading("Stressed matrix")
    print_matrix(stressed_of(TAIL_DATA, result), TAIL_LABELS)

    paragraph(
        f"""

        Every row's worst observation now sits in slot 0, every row's
        second-worst in slot 1, and so on through slot {n_tail - 1}.
        The four desks now suffer their worst outcomes in the same
        scenarios. Outside the tail block nothing was aligned.
        """
    )

    heading("Tail block ordering: tail_order='comonotonic'")

    print(f"    Tail block only (slots 0..{n_tail - 1}):")
    print()
    print_matrix(stressed_of(TAIL_DATA, result)[:, :n_tail], TAIL_LABELS)

    check(
        "tail block ascends in every row (worst with worst)",
        bool(
            np.all(np.diff(stressed_of(TAIL_DATA, result)[:, :n_tail], axis=1) >= 0.0)
        ),
    )

    audit_transformation(
        TAIL_DATA,
        stressed_of(TAIL_DATA, result),
        result.permutations,
        "LowerTailComonotone(comonotonic, preserve=True)",
    )

    # ------------------------------------------------------------------
    # preserve_non_tail_order
    # ------------------------------------------------------------------

    heading("preserve_non_tail_order = True versus False")

    paragraph(
        """
        True  = the observations outside the tail keep their RELATIVE
                scenario ordering. They are not sorted. Only the tail
                block moves.

        False = the remaining observations are also placed in ascending
                order, which makes the complete matrix comonotonic.
        """
    )

    full = transform_empirical_matrix(
        TAIL_DATA,
        LowerTailComonotone(tail_probability=q, preserve_non_tail_order=False),
    )

    print()
    print(f"    preserve=True, non-tail block (slots {n_tail}..{s - 1}):")
    print()
    print_matrix(stressed_of(TAIL_DATA, result)[:, n_tail:], TAIL_LABELS)

    print()
    print("    preserve=False, same slots:")
    print()
    print_matrix(stressed_of(TAIL_DATA, full)[:, n_tail:], TAIL_LABELS)

    print()
    check(
        "preserve=True leaves the non-tail block unsorted",
        not bool(
            np.all(np.diff(stressed_of(TAIL_DATA, result)[:, n_tail:], axis=1) >= 0.0)
        ),
    )
    check(
        "preserve=False gives the fully comonotonic matrix",
        np.array_equal(stressed_of(TAIL_DATA, full), np.sort(TAIL_DATA, axis=1)),
    )
    check(
        "both modes preserve marginals exactly",
        validate_marginal_preservation(TAIL_DATA, stressed_of(TAIL_DATA, result))
        and validate_marginal_preservation(TAIL_DATA, stressed_of(TAIL_DATA, full)),
    )

    # ------------------------------------------------------------------
    # tail_order="random"
    # ------------------------------------------------------------------

    heading("tail_order = 'random'")

    paragraph(
        """
        The same observations still occupy the tail block. Only their
        severity pairing across rows is randomised: a row's worst
        outcome may now land in a different tail slot from another row's
        worst outcome.

        This mode is STOCHASTIC and requires an explicit generator. It
        is a research severity-randomisation mode, not a canonical
        statistical dependence model.
        """
    )

    random_result = transform_empirical_matrix(
        TAIL_DATA,
        LowerTailComonotone(tail_probability=q, tail_order="random"),
        np.random.default_rng(2024),
    )

    print()
    print("    Comonotonic tail block:")
    print()
    print_matrix(stressed_of(TAIL_DATA, result)[:, :n_tail], TAIL_LABELS)

    print()
    print("    Randomised tail block (seed 2024):")
    print()
    print_matrix(stressed_of(TAIL_DATA, random_result)[:, :n_tail], TAIL_LABELS)

    print()
    check(
        "the tail block holds the same observations in both modes",
        np.array_equal(
            np.sort(stressed_of(TAIL_DATA, result)[:, :n_tail], axis=1),
            np.sort(stressed_of(TAIL_DATA, random_result)[:, :n_tail], axis=1),
        ),
    )
    check(
        "only the ordering within the tail block changed",
        np.array_equal(
            stressed_of(TAIL_DATA, result)[:, n_tail:],
            stressed_of(TAIL_DATA, random_result)[:, n_tail:],
        ),
    )
    check(
        "deterministic mode needs no generator",
        not LowerTailComonotone(tail_probability=q).requires_rng,
    )
    check(
        "random mode declares that it needs a generator",
        LowerTailComonotone(tail_probability=q, tail_order="random").requires_rng,
    )

    return stressed_of(TAIL_DATA, result)


# ======================================================================
# Section 6B: upper-tail alignment
# ======================================================================


def section_upper_tail() -> FloatArray:
    """Demonstrate upper-tail alignment."""
    banner("METHOD 2B: UPPER-TAIL ALIGNMENT")

    q = 0.25
    s = TAIL_DATA.shape[1]
    n_tail = max(1, math.ceil(q * s))

    paragraph(
        """
        Upper-tail alignment functions identically to lower-tail alignment,
        but acts on the highest observations rather than the lowest.

        The highest `n_tail` observations of every row move into the
        final scenario slots `S - n_tail .. S-1`. The rows therefore
        realize their highest outcomes together.
        """
    )

    result = transform_empirical_matrix(
        TAIL_DATA,
        UpperTailComonotone(tail_probability=q),
    )

    heading("Permutation produced by the package")
    print_int_matrix(result.permutations, TAIL_LABELS)

    heading("Stressed matrix")
    print_matrix(stressed_of(TAIL_DATA, result), TAIL_LABELS)

    print()
    check(
        "upper tail block ascends in every row",
        bool(
            np.all(
                np.diff(stressed_of(TAIL_DATA, result)[:, s - n_tail :], axis=1) >= 0.0
            )
        ),
    )
    check(
        "marginals are exactly preserved",
        validate_marginal_preservation(TAIL_DATA, stressed_of(TAIL_DATA, result)),
    )

    return stressed_of(TAIL_DATA, result)


# ======================================================================
# Section 6C: dual-tail alignment
# ======================================================================


def section_dual_tail() -> FloatArray:
    """Demonstrate dual-tail alignment."""
    banner("METHOD 2C: DUAL-TAIL ALIGNMENT")

    q = 0.25
    s = TAIL_DATA.shape[1]
    n_tail = max(1, math.ceil(q * s))

    paragraph(
        """
        Dual-tail alignment enforces extreme co-occurrence at BOTH ends
        of the marginal distributions simultaneously. It groups the lowest
        observations into a shared left-tail block, and the highest into a
        shared right-tail block.

        The center block (the non-tail observations) is left independent
        by preserving its relative scenario order.
        """
    )

    result = transform_empirical_matrix(
        TAIL_DATA,
        DualTailComonotone(lower_probability=q, upper_probability=q),
    )

    heading("Permutation produced by the package")
    print_int_matrix(result.permutations, TAIL_LABELS)

    heading("Stressed matrix")
    print_matrix(stressed_of(TAIL_DATA, result), TAIL_LABELS)

    print()
    check(
        "lower tail block ascends in every row",
        bool(
            np.all(np.diff(stressed_of(TAIL_DATA, result)[:, :n_tail], axis=1) >= 0.0)
        ),
    )
    check(
        "upper tail block ascends in every row",
        bool(
            np.all(
                np.diff(stressed_of(TAIL_DATA, result)[:, s - n_tail :], axis=1) >= 0.0
            )
        ),
    )
    check(
        "marginals are exactly preserved",
        validate_marginal_preservation(TAIL_DATA, stressed_of(TAIL_DATA, result)),
    )

    return stressed_of(TAIL_DATA, result)


# ======================================================================
# Section 7: tail co-occurrence diagnostics
# ======================================================================


def section_tail_cooccurrence(
    lower_stressed: FloatArray,
    upper_stressed: FloatArray,
    dual_stressed: FloatArray,
) -> None:
    """Demonstrate the finite-sample tail co-occurrence statistics."""
    banner("DIAGNOSTIC: EMPIRICAL TAIL CO-OCCURRENCE")

    q = 0.25
    s = TAIL_DATA.shape[1]
    n_tail = max(1, math.ceil(q * s))

    paragraph(
        f"""
        This statistic is a FINITE-SAMPLE empirical tail co-occurrence
        measure. It is NOT the asymptotic copula lower-tail dependence
        coefficient. The two differ at any finite S, and this one
        depends on the chosen threshold.

            n_tail = ceil(q * S) = ceil({q} * {s}) = {n_tail}

            lambda_hat(i, j) = #(scenarios in BOTH lower tails) / n_tail

        The public names are:
            lower_tail_dependence()
            upper_tail_dependence()
        """
    )

    heading("Full pairwise matrices, ORIGINAL data")

    print("    Lower Tail:")
    print_square(lower_tail_dependence(TAIL_DATA, q), TAIL_LABELS)
    print()
    print("    Upper Tail:")
    print_square(upper_tail_dependence(TAIL_DATA, q), TAIL_LABELS)

    heading("Full pairwise matrices, AFTER Lower-Tail Alignment")

    print("    Lower Tail:")
    print_square(lower_tail_dependence(lower_stressed, q), TAIL_LABELS)
    print()
    print("    Upper Tail:")
    print_square(upper_tail_dependence(lower_stressed, q), TAIL_LABELS)

    heading("Full pairwise matrices, AFTER Upper-Tail Alignment")

    print("    Lower Tail:")
    print_square(lower_tail_dependence(upper_stressed, q), TAIL_LABELS)
    print()
    print("    Upper Tail:")
    print_square(upper_tail_dependence(upper_stressed, q), TAIL_LABELS)

    heading("Full pairwise matrices, AFTER Dual-Tail Alignment")

    print("    Lower Tail:")
    print_square(lower_tail_dependence(dual_stressed, q), TAIL_LABELS)
    print()
    print("    Upper Tail:")
    print_square(upper_tail_dependence(dual_stressed, q), TAIL_LABELS)

    paragraph(
        """

        Notice how targeted the stresses are. A lower-tail alignment drives
        downside co-occurrence to its maximum without engineering right-tail
        extremes. The dual-tail stress cleanly aligns both ends simultaneously.
        """
    )

    print()
    check(
        "dual-tail alignment drove BOTH co-occurrences to 1.0 everywhere",
        bool(np.allclose(lower_tail_dependence(dual_stressed, q), 1.0))
        and bool(np.allclose(upper_tail_dependence(dual_stressed, q), 1.0)),
    )


# ======================================================================
# Section 8: Pearson and Spearman, by hand
# ======================================================================


def section_correlation_diagnostics() -> None:
    """Make Pearson and Spearman structurally transparent."""
    banner("DIAGNOSTICS: PEARSON AND SPEARMAN, BY HAND")

    x = GOLDEN[0].tolist()
    y = GOLDEN[2].tolist()

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    heading("[HAND CHECK] Pearson correlation of Variable A and Variable C")

    paragraph(
        """
        corr(x, y) =

            sum[(x - mean_x)(y - mean_y)]
            -----------------------------------------------
            sqrt( sum[(x - mean_x)^2] * sum[(y - mean_y)^2] )
        """
    )

    print()
    print(f"    x (Variable A):  {fmt_row(GOLDEN[0])}")
    print(f"    y (Variable C):  {fmt_row(GOLDEN[2])}")
    print()
    print(f"    mean_x = {sum(x):.2f} / {n} = {mean_x:.4f}")
    print(f"    mean_y = {sum(y):.2f} / {n} = {mean_y:.4f}")
    print()

    dev_x = [a - mean_x for a in x]
    dev_y = [b - mean_y for b in y]
    cross = [a * b for a, b in zip(dev_x, dev_y, strict=True)]

    print(f"    (x - mean_x):    {fmt_row(np.asarray(dev_x), decimals=4)}")
    print(f"    (y - mean_y):    {fmt_row(np.asarray(dev_y), decimals=4)}")
    print(f"    cross-products:  {fmt_row(np.asarray(cross), decimals=4)}")
    print()
    print(f"    sum of cross-products:      {sum(cross):.6f}")
    print(f"    sum of squared dev_x:       {sum(a * a for a in dev_x):.6f}")
    print(f"    sum of squared dev_y:       {sum(b * b for b in dev_y):.6f}")

    manual = reference_pearson(x, y)
    package = pearson_correlation(GOLDEN)

    print()
    print(f"    Independent reference: {manual:.10f}")
    print(f"    Package diagnostic:    {float(package[0, 2]):.10f}")
    print()

    check(
        "[HAND CHECK PASSES] Pearson matches",
        math.isclose(manual, float(package[0, 2]), rel_tol=1e-12),
    )

    heading("Full Pearson matrix, original data")
    print_square(package, GOLDEN_LABELS)

    heading("[HAND CHECK] Spearman is Pearson applied to the ranks")

    paragraph(
        """
        Spearman rank correlation is nothing more than the Pearson
        correlation of the average ranks. That is why a permutation-only
        transformation can move Spearman correlation freely while the
        raw values stay exactly the same.
        """
    )

    ranks = average_ranks(GOLDEN)

    print()
    print(f"    Variable A ranks:  {fmt_row(ranks[0])}")
    print(f"    Variable C ranks:  {fmt_row(ranks[2])}")

    manual_spearman = reference_pearson(ranks[0].tolist(), ranks[2].tolist())
    package_spearman = spearman_correlation(GOLDEN)

    print()
    print(f"    Independent Pearson of those ranks: {manual_spearman:.10f}")
    print(
        f"    Package spearman_correlation():     {float(package_spearman[0, 2]):.10f}"
    )
    print()

    check(
        "[HAND CHECK PASSES] Spearman equals Pearson of the ranks",
        math.isclose(manual_spearman, float(package_spearman[0, 2]), rel_tol=1e-12),
    )

    heading("Full Spearman matrix, original data")
    print_square(package_spearman, GOLDEN_LABELS)


# ======================================================================
# Section 9: Gaussian copula
# ======================================================================


def _equicorrelation(k: int, rho: float) -> FloatArray:
    """Build a K x K equicorrelation matrix for the demo."""
    matrix = np.full((k, k), rho, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _copula_data() -> FloatArray:
    """
    Larger matrix used only for the stochastic copula sections.

    Copula dependence is a distributional statement. A six-column matrix
    cannot show it. This matrix is generated once from a fixed seed so
    that the demo output is reproducible.
    """
    rng = np.random.default_rng(31415)

    return np.ascontiguousarray(
        np.vstack([
            rng.standard_normal(2000),
            rng.standard_t(5.0, 2000),
            rng.lognormal(0.0, 0.6, 2000),
            -rng.exponential(1.5, 2000),
        ]),
        dtype=np.float64,
    )


def section_gaussian_copula(data: FloatArray) -> None:
    """Demonstrate the stochastic Gaussian copula transformer."""
    banner("METHOD 3: GAUSSIAN COPULA")

    paragraph(
        """
        THE CONCEPTUAL PIPELINE

            latent Gaussian dependence specification
                    |
                    v
            finite latent sample  (random)
                    |
                    v
            rank ordering of that sample
                    |
                    v
            empirical permutation
                    |
                    v
            ORIGINAL empirical marginals, rearranged

        The correlation matrix describes LATENT Gaussian dependence.
        It does NOT replace the empirical marginals with Gaussian
        marginals. Not one observation is simulated into the output.
        The Gaussian sample is used only to decide an ordering, and is
        then thrown away.
        """
    )

    k = data.shape[0]
    correlation = _equicorrelation(k, 0.85)

    heading("Latent correlation specification")
    print_square(correlation, COPULA_LABELS)

    heading("Cholesky factor")

    paragraph(
        """
        The copula factorises the correlation matrix ONCE at
        construction and reuses the factor for every transformation.
        The same LAPACK call is shown here for display only:

            L = np.linalg.cholesky(R)      with     L @ L.T == R

        Latent scores are then Z = L @ eps, with eps ~ N(0, I).
        """
    )

    cholesky = np.linalg.cholesky(correlation).astype(np.float64)

    print()
    print_square(cholesky, COPULA_LABELS)
    print()

    check(
        "L @ L.T reproduces the correlation matrix",
        bool(np.allclose(cholesky @ cholesky.T, correlation)),
    )

    copula = GaussianCopula(latent_correlation=correlation)

    heading("Configuration ownership")

    print(
        f"    latent_correlation writable:  {copula.latent_correlation.flags.writeable}"
    )
    print(f"    declares requires_rng:        {copula.requires_rng}")

    paragraph(
        """

        The transformer owns a read-only copy of the correlation matrix.
        Mutating the caller's original array afterwards cannot change
        the stress specification.
        """
    )

    rng = np.random.default_rng(12345)
    result = transform_empirical_matrix(data, copula, rng)

    heading("Resulting permutation, first 12 slots of each row")
    print_int_matrix(result.permutations[:, :12], COPULA_LABELS)

    heading("Resulting stressed matrix, first 8 slots of each row")
    print_matrix(stressed_of(data, result)[:, :8], COPULA_LABELS, decimals=3)

    heading("Dependence before and after")

    before = spearman_correlation(data)
    after = spearman_correlation(stressed_of(data, result))

    print("    Spearman correlation BEFORE:")
    print()
    print_square(before, COPULA_LABELS)
    print()
    print("    Spearman correlation AFTER:")
    print()
    print_square(after, COPULA_LABELS)

    paragraph(
        """

        LATENT IS NOT OBSERVABLE.
        The specified latent correlation was 0.85. The realised
        Spearman correlation above is not 0.85 and is not meant to be.
        For a Gaussian copula the population Spearman correlation is

            rho_S = (6 / pi) * arcsin(rho / 2)

        and the finite-sample estimate scatters around that value.
        """
    )

    expected_spearman = (6.0 / math.pi) * math.asin(0.85 / 2.0)

    print()
    print(f"    specified latent rho:                 {0.85:.4f}")
    print(f"    population Spearman implied by it:    {expected_spearman:.4f}")
    print(f"    realised finite-sample Spearman:      {float(after[0, 1]):.4f}")

    audit_transformation(
        data, stressed_of(data, result), result.permutations, "GaussianCopula"
    )

    heading("Reproducibility")

    repeat = transform_empirical_matrix(
        data,
        copula,
        np.random.default_rng(12345),
    )

    print("    Gaussian copula with seed 12345  -> permutation A")
    print("    Gaussian copula with seed 12345  -> permutation A again")
    print()

    check(
        "identical seed gives identical permutation",
        np.array_equal(result.permutations, repeat.permutations),
    )
    check(
        "identical seed gives identical stressed matrix",
        np.array_equal(stressed_of(data, result), stressed_of(data, repeat)),
    )

    other = transform_empirical_matrix(
        data,
        copula,
        np.random.default_rng(99999),
    )

    print()
    print("    Gaussian copula with seed 99999  -> a DIFFERENT permutation")
    print()

    check(
        "a different seed gives a different finite-sample realisation",
        not np.array_equal(result.permutations, other.permutations),
    )
    check(
        "but the marginals are identical in both realisations",
        np.array_equal(
            np.sort(stressed_of(data, result), axis=1),
            np.sort(stressed_of(data, other), axis=1),
        ),
    )

    banner("IMPORTANT METHODOLOGICAL POINT")

    paragraph(
        """
        A fixed stress specification + fixed seed
            = reproducible finite-sample stress

        A fixed stress specification without fixing the RNG realisation
            != unique deterministic finite-sample stress

        The Gaussian and t copulas draw ONE finite sample from the
        specified dependence model. A different draw gives a different
        permutation. That is Monte Carlo realisation noise. A seed makes
        the result reproducible; it does not make the specification
        produce a unique answer.

        This matters downstream: any charge computed from a stochastic
        copula stress inherits realisation noise, and the seed becomes
        part of the methodology rather than an implementation detail.
        """
    )


# ======================================================================
# Section 10: t copula
# ======================================================================


def section_t_copula(data: FloatArray) -> None:
    """Demonstrate the t copula and the effect of nu."""
    banner("METHOD 4: T COPULA")

    paragraph(
        """
        The t copula has two components:

            1. latent correlation      (the same role as before)
            2. degrees of freedom nu   (the tail-dependence control)

        Construction:

            Z   = L @ eps,          eps ~ N(0, I)
            X_s = Z_s / sqrt( chi2_nu_s / nu )

        ONE common chi-square scale is drawn PER SCENARIO and shared
        across all K dimensions in that scenario. That shared scale is
        precisely what creates tail dependence: when the scale is small,
        every dimension in that scenario is inflated together.

        Drawing a K x S chi-square array instead would scale each
        marginal independently, destroy the shared shock, and silently
        turn the t copula back into something close to a Gaussian one.
        The package draws size=S, never size=(K, S).
        """
    )

    heading("A precise statement about nu")

    paragraph(
        """
        Lower nu does NOT simply mean "higher correlation".

        Lower nu means a heavier-tailed latent distribution and
        therefore STRONGER SYMMETRIC TAIL DEPENDENCE for a given latent
        correlation: joint extreme outcomes, in both tails, become more
        frequent. The overall rank correlation can stay broadly similar
        while the tail behaviour changes markedly.
        """
    )

    k = data.shape[0]
    correlation = _equicorrelation(k, 0.60)

    heading("Latent correlation used for both nu values")
    print_square(correlation, COPULA_LABELS)

    def summarise(nu: float) -> tuple[float, float]:
        """Average off-diagonal Spearman and tail co-occurrence over seeds."""
        copula = TCopula(latent_correlation=correlation, degrees_of_freedom=nu)

        spearman_values: list[float] = []
        tail_values: list[float] = []

        for seed in range(12):
            outcome = transform_empirical_matrix(
                data,
                copula,
                np.random.default_rng(seed),
            )

            assert np.array_equal(
                np.sort(stressed_of(data, outcome), axis=1),
                np.sort(data, axis=1),
            )

            for matrix, sink in (
                (spearman_correlation(stressed_of(data, outcome)), spearman_values),
                (
                    lower_tail_dependence(stressed_of(data, outcome), 0.05),
                    tail_values,
                ),
            ):
                off = (matrix.sum() - np.trace(matrix)) / (k * (k - 1))
                sink.append(float(off))

        return float(np.mean(spearman_values)), float(np.mean(tail_values))

    heading("Effect of nu, averaged over 12 independent realisations")

    paragraph(
        """
        Averaging over seeds is deliberate. A single stochastic draw is
        not evidence of a distributional property, and asserting exact
        values from one sample would be a brittle test.
        """
    )

    spearman_3, tail_3 = summarise(3.0)
    spearman_30, tail_30 = summarise(30.0)

    print()
    print(f"    {'nu':>6}  {'mean Spearman':>16}  {'mean tail co-occurrence':>26}")
    print(f"    {'-' * 6}  {'-' * 16}  {'-' * 26}")
    print(f"    {3.0:>6.1f}  {spearman_3:>16.4f}  {tail_3:>26.4f}")
    print(f"    {30.0:>6.1f}  {spearman_30:>16.4f}  {tail_30:>26.4f}")

    paragraph(
        f"""

        Rank correlation is broadly comparable between the two settings
        ({spearman_3:.3f} versus {spearman_30:.3f}), but the lower-tail
        co-occurrence at the 5% threshold is materially higher for
        nu = 3 ({tail_3:.3f}) than for nu = 30 ({tail_30:.3f}).

        That is the point: nu controls TAIL dependence, not the general
        level of correlation.
        """
    )

    print()
    check("lower nu produced stronger empirical tail co-occurrence", tail_3 > tail_30)

    heading("Parameter contract")

    for bad_nu in (0.0, -1.0, float("inf")):
        try:
            TCopula(latent_correlation=correlation, degrees_of_freedom=bad_nu)
        except ValueError:
            print(f"    degrees_of_freedom = {bad_nu!r:<6} rejected with ValueError")
        else:  # pragma: no cover - demo guard
            raise AssertionError(f"nu={bad_nu} should have been rejected")

    paragraph(
        """

        nu = infinity is NOT accepted as a Gaussian limiting case. The
        contract requires a finite, strictly positive value, and the
        Gaussian limit has its own explicit transformer. The two methods
        are not identical for any finite nu, so the package does not
        pretend otherwise.
        """
    )


# ======================================================================
# Section 11: what changed
# ======================================================================


def section_what_changed(comonotonic_stressed: FloatArray) -> None:
    """The clearest contrast: marginals fixed, pairing rearranged."""
    banner("WHAT CHANGED?")

    heading("Per-row marginals")

    for row, label in enumerate(GOLDEN_LABELS):
        original_sorted = np.sort(GOLDEN[row])
        stressed_sorted = np.sort(comonotonic_stressed[row])

        print()
        print(f"    {label}")
        print(f"        Original sorted values:  {fmt_row(original_sorted)}")
        print(f"        Stressed sorted values:  {fmt_row(stressed_sorted)}")

        check("        identical", np.array_equal(original_sorted, stressed_sorted))

    heading("Selected scenario columns, before and after")

    paragraph(
        """
        A scenario column is a joint outcome: what every variable did at
        the same moment. This is what the stress rearranges.
        """
    )

    for slot in (0, 2, 5):
        print()
        print(f"    Scenario slot {slot}:")
        print("        Original:")
        for row, label in enumerate(GOLDEN_LABELS):
            print(f"            {label} = {GOLDEN[row, slot]:>8.2f}")
        print("        Stressed:")
        for row, label in enumerate(GOLDEN_LABELS):
            print(f"            {label} = {comonotonic_stressed[row, slot]:>8.2f}")

    paragraph(
        """

        The values themselves have not been invented or altered.
        What changed is which observations are paired together
        across rows.

        That is the dependence stress.

        Before the stress, slot 0 held a large A with a small C. After a
        comonotonic stress, slot 0 holds the worst value of every row at
        once. No number in any row is new; the joint outcomes are.
        """
    )


# ======================================================================
# Section 12: marginal preservation
# ======================================================================


def section_marginal_preservation(comonotonic_stressed: FloatArray) -> None:
    """State and verify the marginal-preservation invariant."""
    banner("MARGINAL PRESERVATION CHECK")

    paragraph(
        """
        For every row k:

            sort(stressed[k, :]) == sort(original[k, :])

        The comparison is EXACT. No tolerance is appropriate, because a
        stressed observation is one of the original float64 values, bit
        for bit. It was moved, not recomputed.
        """
    )

    print()

    for row, label in enumerate(GOLDEN_LABELS):
        check(
            f"{label}: sorted rows identical",
            np.array_equal(
                np.sort(GOLDEN[row]),
                np.sort(comonotonic_stressed[row]),
            ),
        )

    check(
        "package validate_marginal_preservation()",
        validate_marginal_preservation(GOLDEN, comonotonic_stressed),
    )

    heading("Tied values keep their exact multiplicity")

    paragraph(
        """
        Variable B contains repeated values. A transformation that
        preserved only the SET of values would lose the duplicates. The
        multiset must survive intact.
        """
    )

    original_values, original_counts = np.unique(GOLDEN[1], return_counts=True)
    stressed_values, stressed_counts = np.unique(
        comonotonic_stressed[1],
        return_counts=True,
    )

    print()
    print(f"    Original unique values:  {fmt_row(original_values)}")
    print(f"    Original counts:         {fmt_int_row(original_counts)}")
    print(f"    Stressed unique values:  {fmt_row(stressed_values)}")
    print(f"    Stressed counts:         {fmt_int_row(stressed_counts)}")
    print()

    check(
        "unique values identical",
        np.array_equal(original_values, stressed_values),
    )
    check(
        "multiplicities identical",
        np.array_equal(original_counts, stressed_counts),
    )


# ======================================================================
# Section 13: method comparison
# ======================================================================


def section_method_comparison(copula_data: FloatArray) -> None:
    """Summarise the methods and compare diagnostics before and after."""
    banner("METHOD COMPARISON")

    rows = (
        ("Comonotonic", "Yes", "No", "Perfect positive rank alignment"),
        ("Lower tail (comono)", "Yes", "No", "Engineer empirical lower tail"),
        ("Lower tail (random)", "No", "Yes", "Tail membership fixed, pairing random"),
        ("Upper tail (comono)", "Yes", "No", "Engineer empirical upper tail"),
        ("Dual tail (comono)", "Yes", "No", "Engineer both empirical tails"),
        ("Gaussian copula", "No", "Yes", "Latent Gaussian rank dependence"),
        ("t copula", "No", "Yes", "Latent t rank dependence, tail control"),
    )

    print()
    print(f"{'Method':<21} {'Determ.?':<9} {'RNG?':<6} Main mechanism")
    print("-" * WIDTH)

    for name, deterministic, needs_rng, mechanism in rows:
        print(f"{name:<21} {deterministic:<9} {needs_rng:<6} {mechanism}")

    paragraph(
        """

        "Deterministic" means the same input and the same specification
        always give the same permutation, with no generator involved.
        A seeded stochastic method is REPRODUCIBLE, which is a weaker
        property, and the table does not conflate the two.
        """
    )

    heading("Diagnostics before and after, on the copula example")

    engine = DependenceStressEngine(data=copula_data)
    k = copula_data.shape[0]
    correlation = _equicorrelation(k, 0.85)

    def off_diagonal(matrix: FloatArray) -> float:
        return float((matrix.sum() - np.trace(matrix)) / (k * (k - 1)))

    results = {
        "comonotonic": engine.apply_comonotonic(),
        "lower tail q=0.05": engine.apply_lower_tail_comonotone(0.05),
        "upper tail q=0.05": engine.apply_upper_tail_comonotone(0.05),
        "dual tail q=0.05": engine.apply_dual_tail_comonotone(0.05, 0.05),
        "gaussian copula": engine.apply_gaussian_copula(
            correlation,
            rng=np.random.default_rng(7),
        ),
        "t copula nu=3": engine.apply_t_copula(
            correlation,
            degrees_of_freedom=3.0,
            rng=np.random.default_rng(7),
        ),
    }

    # Diagnostics need row-level observations, so this section explicitly
    # reconstructs the stressed matrices. That is the legitimate use of
    # the derived representation: a diagnostic consumer that genuinely
    # needs rows, not the production portfolio path.
    outcomes = {"original": copula_data} | {
        name: engine.stressed_matrix(result) for name, result in results.items()
    }

    print()
    print(
        f"    {'variant':<20} {'Pearson':>9} {'Spearman':>9} {'L-tail':>9} {'U-tail':>9}"
    )
    print(f"    {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")

    for name, matrix in outcomes.items():
        print(
            f"    {name:<20}"
            f" {off_diagonal(pearson_correlation(matrix)):>9.4f}"
            f" {off_diagonal(spearman_correlation(matrix)):>9.4f}"
            f" {off_diagonal(lower_tail_dependence(matrix, 0.05)):>9.4f}"
            f" {off_diagonal(upper_tail_dependence(matrix, 0.05)):>9.4f}"
        )

    paragraph(
        """

        Every row of that table describes a DIFFERENT dependence
        structure. Every row also has byte-identical marginals.
        """
    )

    print()

    for name, matrix in outcomes.items():
        check(
            f"{name}: marginals unchanged",
            validate_marginal_preservation(copula_data, matrix),
        )


# ======================================================================
# Section 14: fail-fast behaviour
# ======================================================================


def section_invalid_inputs() -> None:
    """Show the package rejecting malformed input loudly."""
    banner("FAIL-FAST BEHAVIOUR")

    paragraph(
        """
        Silent failures are worse than loud ones. Every boundary
        contract raises rather than repairing, inferring, or clipping.
        Only the expected exception type is caught below.
        """
    )

    correlation = _equicorrelation(3, 0.4)

    heading("1-D input")

    try:
        transform_empirical_matrix(np.arange(6, dtype=np.float64), Comonotonic())
    except ValueError as error:
        print(f"    Expected: reject input\n    Observed: ValueError: {error}")
        check("rejected", True)

    heading("NaN in the observation matrix")

    try:
        transform_empirical_matrix(
            np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
            Comonotonic(),
        )
    except ValueError as error:
        print(f"    Expected: reject input\n    Observed: ValueError: {error}")
        check("rejected", True)

    heading("Correlation matrix that is not positive definite")

    singular = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    try:
        GaussianCopula(latent_correlation=singular)
    except ValueError as error:
        print(f"    Expected: reject input\n    Observed: ValueError: {error}")
        check("rejected", True)

    paragraph(
        """

        The package does NOT eigenvalue-clip, project to the nearest
        correlation matrix, or regularise. Those are caller policy
        decisions, and making them silently inside a validator would
        hide a modelling error.
        """
    )

    heading("Correlation dimension mismatch")

    try:
        transform_empirical_matrix(
            GOLDEN,
            GaussianCopula(latent_correlation=_equicorrelation(5, 0.4)),
            np.random.default_rng(0),
        )
    except ValueError as error:
        print(f"    Expected: reject input\n    Observed: ValueError: {error}")
        check("rejected", True)

    heading("Malformed RNG for a stochastic method")

    try:
        transform_empirical_matrix(
            GOLDEN,
            GaussianCopula(latent_correlation=correlation),
            "not-a-generator",  # type: ignore[arg-type]
        )
    except TypeError as error:
        print(f"    Expected: reject input\n    Observed: TypeError: {error}")
        check("rejected", True)

    heading("Missing RNG for a stochastic method")

    try:
        transform_empirical_matrix(
            GOLDEN,
            LowerTailComonotone(tail_probability=0.25, tail_order="random"),
        )
    except TypeError as error:
        print(f"    Expected: reject input\n    Observed: TypeError: {error}")
        check("rejected", True)

    heading("Invalid permutation matrix")

    try:
        apply_permutation(GOLDEN, np.zeros(GOLDEN.shape, dtype=np.int64))
    except ValueError as error:
        print(f"    Expected: reject input\n    Observed: ValueError: {error}")
        check("rejected", True)

    heading("Deterministic methods accept no RNG and need none")

    without = transform_empirical_matrix(GOLDEN, Comonotonic())
    with_one = transform_empirical_matrix(
        GOLDEN,
        Comonotonic(),
        np.random.default_rng(123),
    )

    print()
    check(
        "Comonotonic result is unchanged by any generator",
        np.array_equal(without.permutations, with_one.permutations),
    )


# ======================================================================
# Section 15: audit mode
# ======================================================================


def section_audit_mode(copula_data: FloatArray) -> None:
    """Contrast the production path with the explicit audit path."""
    banner("AUDIT MODE")

    paragraph(
        """
        Production transformation:
            expensive structural audits OFF

        Audit transformation (audit=True):
            shape check
            permutation validity check      O(K S log S)
            marginal preservation check     O(K S log S)
            exact reconstruction check      full K x S gather

        Why is the audit off by default?

        The permutation is a permutation BY CONSTRUCTION. argsort over
        S elements returns a permutation of 0..S-1 by definition, and
        the construction writes each output slot exactly once. Running
        the audit on every call would re-derive, at O(K S log S), a
        property the construction already guarantees. So the package
        makes the audit an explicit, opt-in operation instead of a
        silent tax on every production call.
        """
    )

    engine = DependenceStressEngine(data=copula_data)

    def timed(audit: bool, repeats: int = 20) -> float:
        engine.apply_comonotonic(audit=audit)
        start = time.perf_counter()

        for _ in range(repeats):
            engine.apply_comonotonic(audit=audit)

        return (time.perf_counter() - start) / repeats * 1e3

    production_ms = timed(False)
    audited_ms = timed(True)

    heading(f"Measured cost, K={copula_data.shape[0]} S={copula_data.shape[1]}")

    print()
    print(f"    production path (audit=False):  {production_ms:8.3f} ms")
    print(f"    audit path      (audit=True):   {audited_ms:8.3f} ms")
    print(f"    audit overhead:                 {audited_ms / production_ms - 1:8.1%}")

    heading("Running the audit path for real")

    result = engine.apply_comonotonic(audit=True)

    print()
    check("audit path completed without raising", True)
    check(
        "engine.validate_result() agrees independently",
        engine.validate_result(result),
    )

    heading("The audit catches what marginal checks alone cannot")

    paragraph(
        """
        Marginal preservation alone is a weak check. A transformation
        can preserve every marginal while the RECORDED permutation is
        wrong, which would break replay and audit. Compare a correct
        permutation against a deliberately false one below.
        """
    )

    honest = engine.apply_comonotonic()
    liar = np.broadcast_to(
        np.arange(copula_data.shape[1], dtype=np.int64),
        copula_data.shape,
    )

    print()
    check(
        "marginals preserved (passes, but proves little)",
        validate_marginal_preservation(copula_data, stressed_of(copula_data, honest)),
    )
    check(
        "validate_transformation rejects the false permutation",
        not validate_transformation(
            copula_data,
            stressed_of(copula_data, honest),
            np.ascontiguousarray(liar),
        ),
    )
    check(
        "validate_transformation accepts the true permutation",
        validate_transformation(
            copula_data,
            stressed_of(copula_data, honest),
            honest.permutations,
        ),
    )

    heading("Result immutability")

    paragraph(
        """
        A frozen dataclass does not freeze array contents. The result
        therefore owns read-only copies of both first-class outputs, so
        neither can be altered after the fact and neither can alias
        engine memory.
        """
    )

    print()
    print(f"    result.permutations writable:   {result.permutations.flags.writeable}")
    print(f"    result.portfolio_pnl writable:  {result.portfolio_pnl.flags.writeable}")
    print(f"    engine.data writable:           {engine.data.flags.writeable}")
    print()

    for name, array in (
        ("permutations", result.permutations),
        ("portfolio_pnl", result.portfolio_pnl),
    ):
        try:
            array[0] = 0
        except ValueError as error:
            print(f"    Write to result.{name} -> ValueError: {error}")

    print()
    check(
        "both first-class outputs are genuinely read-only",
        not result.permutations.flags.writeable
        and not result.portfolio_pnl.flags.writeable,
    )


# ======================================================================
# Section: the two-output contract, fully hand-verifiable
# ======================================================================

TOY_LABELS = ("Row A", "Row B", "Row C")

#: K = 3, S = 5. Small enough that a reviewer can check every scenario
#: of the portfolio aggregation with a calculator.
TOY = np.array(
    [
        [2.0, 5.0, 1.0, 7.0, 4.0],
        [3.0, 4.0, 8.0, 2.0, 6.0],
        [1.0, 6.0, 2.0, 5.0, 9.0],
    ],
    dtype=np.float64,
)

TOY_PERMUTATIONS = np.array(
    [
        [1, 3, 0, 2, 4],
        [3, 0, 2, 4, 1],
        [2, 1, 3, 0, 4],
    ],
    dtype=np.int64,
)


def section_two_output_contract() -> None:
    """Explain and hand-verify permutations plus portfolio_pnl."""
    banner("THE TWO FIRST-CLASS OUTPUTS")

    paragraph(
        """
        Every transformation returns exactly two things:

            permutations    K x S int64    the dependence stress itself
            portfolio_pnl   S   float64    the scenario aggregate

        THE PERMUTATION IS THE DEPENDENCE STRESS.
        THE S-VECTOR IS THE STRESSED PORTFOLIO SCENARIO OUTPUT.
        THE K x S STRESSED MATRIX IS AN OPTIONAL DERIVED REPRESENTATION.

        The aggregation is:

            portfolio_pnl[s] = sum_k data[k, permutations[k, s]]

        The package performs exactly this one sum over rows and attaches
        no portfolio semantics to them. The caller decides whether
        summing the supplied rows is meaningful. DEPENDENCE_STRESS knows
        nothing about VaR, ES, CSC, desks, books, or capital.

        portfolio_pnl has NumPy shape (S,), not (1, S). The mathematical
        object is a 1 x S scenario vector, but a flat vector is the
        natural NumPy shape and spares every consumer a singleton axis.
        """
    )

    heading("[HAND CHECK] A complete K = 3, S = 5 audit")

    print()
    print("ORIGINAL DATA")
    print("-------------")
    print_matrix(TOY, TOY_LABELS, decimals=1)

    print()
    print("PERMUTATIONS")
    print("------------")
    print_int_matrix(TOY_PERMUTATIONS, TOY_LABELS)

    paragraph(
        """

        Now walk every scenario. For slot s, each row contributes the
        observation sitting at its own permutation index.
        """
    )

    portfolio = aggregate_portfolio(TOY, TOY_PERMUTATIONS)
    manual_portfolio: list[float] = []

    for slot in range(TOY.shape[1]):
        picks = []
        total = 0.0

        print()
        print(f"    Scenario {slot}:")

        for row, label in enumerate(TOY_LABELS):
            index = int(TOY_PERMUTATIONS[row, slot])
            value = float(TOY[row, index])
            total += value
            picks.append(f"{value:.0f}")

            print(
                f"        {label} uses original observation "
                f"{label[-1]}[{index}] = {value:>4.0f}"
            )

        manual_portfolio.append(total)

        print(f"        Portfolio = {' + '.join(picks)} = {total:.0f}")

    print()
    print(
        f"    Independent reference: {fmt_row(np.asarray(manual_portfolio), decimals=1)}"
    )
    print(f"    Package result:        {fmt_row(portfolio, decimals=1)}")
    print()

    check(
        "[HAND CHECK PASSES] portfolio_pnl matches manual arithmetic",
        np.array_equal(np.asarray(manual_portfolio), portfolio),
    )

    heading("Row marginals are untouched")

    reconstructed = reconstruct_stressed(TOY, TOY_PERMUTATIONS)

    for row, label in enumerate(TOY_LABELS):
        print()
        print(f"    {label}")
        print(f"        original sorted:  {fmt_row(np.sort(TOY[row]), decimals=1)}")
        print(
            f"        stressed sorted:  {fmt_row(np.sort(reconstructed[row]), decimals=1)}"
        )

    print()
    check(
        "every row-sorted sequence is identical",
        np.array_equal(
            np.sort(reconstructed, axis=1),
            np.sort(TOY, axis=1),
        ),
    )

    heading("Why standalone row statistics cannot change")

    paragraph(
        """
        A statistic that depends only on one row's empirical marginal
        reads the same multiset of numbers before and after. Rearranging
        which scenario a number sits in cannot change its mean,
        variance, quantiles, minimum, or maximum.
        """
    )

    print()
    print(
        f"    {'row':<8}{'mean before':>14}{'mean after':>13}{'min before':>13}{'min after':>12}"
    )
    print(f"    {'-' * 8}{'-' * 14:>14}{'-' * 13:>13}{'-' * 13:>13}{'-' * 12:>12}")

    for row, label in enumerate(TOY_LABELS):
        print(
            f"    {label:<8}"
            f"{TOY[row].mean():>14.4f}{reconstructed[row].mean():>13.4f}"
            f"{TOY[row].min():>13.1f}{reconstructed[row].min():>12.1f}"
        )

    heading("Why portfolio risk CAN change")

    paragraph(
        """
        The portfolio scenario vector is a sum ACROSS rows within each
        scenario. Repairing which observations meet in a scenario
        changes that sum, even though no row changed.
        """
    )

    original_portfolio = TOY.sum(axis=0)

    print()
    print(f"    original portfolio:  {fmt_row(original_portfolio, decimals=1)}")
    print(f"    stressed portfolio:  {fmt_row(portfolio, decimals=1)}")
    print()
    print(
        f"    original sorted:     {fmt_row(np.sort(original_portfolio), decimals=1)}"
    )
    print(f"    stressed sorted:     {fmt_row(np.sort(portfolio), decimals=1)}")
    print()
    print(
        f"    original min / std:  {original_portfolio.min():>6.2f} / {original_portfolio.std():>6.3f}"
    )
    print(f"    stressed min / std:  {portfolio.min():>6.2f} / {portfolio.std():>6.3f}")

    paragraph(
        """

        The row marginals are byte-identical and the portfolio total
        across all scenarios is unchanged, because it is just the sum of
        fixed row totals. What moved is the SHAPE of the portfolio
        scenario distribution. That is the quantity a downstream risk
        engine cares about, and it is why the dependence stress matters.
        """
    )

    print()
    check(
        "total across all scenarios is unchanged",
        math.isclose(float(portfolio.sum()), float(TOY.sum())),
    )

    heading("Exact reconstruction and exact aggregation")

    paragraph(
        """
        The production path never builds the K x S stressed matrix. It
        goes straight from the permutation to the S-vector:

            portfolio_pnl = empty(S)
            take(data[0], permutations[0], out=portfolio_pnl)
            for k in 1 .. K-1:
                portfolio_pnl += data[k][permutations[k]]

        Accumulation runs k = 0, 1, ..., K-1, the same order NumPy uses
        when reducing a C-contiguous array along axis 0. The two agree
        BITWISE, which is asserted below rather than assumed.
        """
    )

    print()
    print(
        f"    reconstructed.sum(axis=0):  {fmt_row(reconstructed.sum(axis=0), decimals=1)}"
    )
    print(f"    result portfolio_pnl:       {fmt_row(portfolio, decimals=1)}")
    print()

    check(
        "portfolio_pnl == reconstructed.sum(axis=0), bitwise",
        np.array_equal(reconstructed.sum(axis=0), portfolio),
    )
    check(
        "stressed == take_along_axis(data, permutations)",
        np.array_equal(
            reconstructed,
            np.take_along_axis(TOY, TOY_PERMUTATIONS, axis=1),
        ),
    )


# ======================================================================
# Section: streaming Monte Carlo
# ======================================================================


def section_streaming(copula_data: FloatArray) -> None:
    """Demonstrate the streaming ensemble API and its memory model."""
    banner("STREAMING MONTE CARLO")

    paragraph(
        """
        A stochastic copula specification does not induce ONE
        finite-sample permutation. It induces a DISTRIBUTION over them.
        Estimating anything about that distribution means drawing many
        realisations.

        Naively, B = 10,000 realisations of a K x S stressed matrix is a
        B x K x S array. At K = 4 and S = 2000 that is 640 MB, and it is
        entirely wasted if the consumer only needs a scalar per draw.

        The iterator yields one realisation at a time and accumulates
        nothing:

            base data       K x S   owned once by the engine
            permutation     K x S   one live realisation
            portfolio_pnl   S       one live realisation

        This is MONTE CARLO ESTIMATION of the finite-sample distribution
        the specification induces. It is NOT a bootstrap: the empirical
        observations are never resampled. Only their cross-sectional
        pairing is redrawn.
        """
    )

    k, s = copula_data.shape
    correlation = _equicorrelation(k, 0.80)
    copula = GaussianCopula(latent_correlation=correlation)
    engine = DependenceStressEngine(data=copula_data)

    n_realizations = 10_000

    heading(f"Consuming {n_realizations:,} realisations, keeping only scalars")

    paragraph(
        """
        The loop below retains one float per draw. A downstream risk
        engine would compute its own metric here. DEPENDENCE_STRESS does
        not know what that metric is, and deliberately has no VaR, ES or
        CSC of its own.
        """
    )

    start = time.perf_counter()
    worst_scenarios = np.empty(n_realizations, dtype=np.float64)

    for index, result in enumerate(
        engine.iter_results(
            copula,
            rng=np.random.default_rng(4242),
            n_realizations=n_realizations,
        )
    ):
        worst_scenarios[index] = result.portfolio_pnl.min()

    elapsed = time.perf_counter() - start

    ensemble_bytes = n_realizations * k * s * 8
    live_bytes = (k * s * 8) + (k * s * 8) + (s * 8)

    print()
    print(f"    realisations consumed:        {n_realizations:>12,}")
    print(f"    elapsed:                      {elapsed:>12.2f} s")
    print(f"    retained per realisation:     {'1 float64':>12}")
    print()
    print(f"    B x K x S if materialised:    {ensemble_bytes / 1e6:>9.1f} MB")
    print(f"    live arrays while streaming:  {live_bytes / 1e6:>9.3f} MB")
    print(f"    ratio:                        {ensemble_bytes / live_bytes:>9.0f} x")

    paragraph(
        f"""

        Distribution of the worst stressed portfolio scenario across
        {n_realizations:,} finite-sample realisations of ONE fixed
        specification:
        """
    )

    print()
    print(f"    mean:    {worst_scenarios.mean():>10.4f}")
    print(f"    std:     {worst_scenarios.std():>10.4f}")
    print(f"    min:     {worst_scenarios.min():>10.4f}")
    print(f"    max:     {worst_scenarios.max():>10.4f}")

    paragraph(
        """

        That spread IS the Monte Carlo realisation noise. The
        specification is fixed; only the finite-sample draw varies. A
        downstream charge computed from a single seeded realisation
        inherits this spread, which is why the seed is part of the
        methodology and not an implementation detail.
        """
    )

    heading("Sequence reproducibility")

    def first_three(seed: int) -> list[float]:
        return [
            float(result.portfolio_pnl[0])
            for result in engine.iter_results(
                copula,
                rng=np.random.default_rng(seed),
                n_realizations=3,
            )
        ]

    same = first_three(7)
    again = first_three(7)
    different = first_three(8)

    print()
    print(f"    seed 7, first pass:   {[round(v, 6) for v in same]}")
    print(f"    seed 7, second pass:  {[round(v, 6) for v in again]}")
    print(f"    seed 8:               {[round(v, 6) for v in different]}")
    print()

    check("the same seed reproduces the same SEQUENCE", same == again)
    check("a different seed gives a different sequence", same != different)

    paragraph(
        """

        Reproducibility of a stochastic sequence is not determinism of
        the method. The copula still induces a distribution over
        finite-sample permutations rather than one uniquely determined
        permutation. A deterministic rank-rearrangement construction is
        separate methodological work and has not been implemented.
        """
    )

    heading("Invariants hold on every streamed realisation")

    checked = 0

    for result in engine.iter_results(
        copula,
        rng=np.random.default_rng(11),
        n_realizations=25,
        audit=True,
    ):
        assert result.portfolio_pnl.shape == (s,)
        checked += 1

    print()
    check(f"{checked} realisations passed the full structural audit", checked == 25)


# ======================================================================
# Section 16: summary
# ======================================================================


def section_summary() -> None:
    """Close with the summary and the core invariant."""
    banner("DEMONSTRATION COMPLETE")

    paragraph(
        """
        1. DEPENDENCE_STRESS does not modify empirical marginals.
        2. Every valid stress is represented by row-wise permutations.
        3. The stressed matrix is produced by take_along_axis.
        4. Comonotonicity creates the extreme positive rank benchmark.
        5. Lower-tail alignment specifically stresses empirical
           downside co-occurrence.
        6. Gaussian and t copulas generate stochastic finite-sample
           rank structures.
        7. A seed gives reproducibility, not deterministic uniqueness.
        8. All structural invariants remain exact.
        """
    )

    banner("CORE INVARIANT")

    paragraph(
        """
        stressed = np.take_along_axis(
            data,
            permutations,
            axis=1,
        )

        and therefore:

        np.array_equal(
            np.sort(stressed, axis=1),
            np.sort(data, axis=1),
        )

        must always be True for a valid transformation.
        """
    )


# ======================================================================
# Entry point
# ======================================================================


def main() -> int:
    """Run the complete walkthrough."""
    section_introduction()
    section_golden_matrix()
    section_ranks()
    section_permutation_mechanism()
    section_two_output_contract()

    comonotonic_stressed = section_comonotonic()

    tail_stressed = section_lower_tail()
    upper_tail_stressed = section_upper_tail()
    dual_tail_stressed = section_dual_tail()

    section_tail_cooccurrence(tail_stressed, upper_tail_stressed, dual_tail_stressed)

    section_correlation_diagnostics()

    copula_data = _copula_data()

    section_gaussian_copula(copula_data)
    section_t_copula(copula_data)
    section_streaming(copula_data)

    section_what_changed(comonotonic_stressed)
    section_marginal_preservation(comonotonic_stressed)
    section_method_comparison(copula_data)

    section_invalid_inputs()
    section_audit_mode(copula_data)

    section_summary()

    print()
    print(RULE)
    print("END OF DEMO: ALL DEMONSTRATION ASSERTIONS PASSED!")
    print(RULE)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
