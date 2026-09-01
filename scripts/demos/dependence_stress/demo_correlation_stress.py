# scripts/demos/dependence_stress/demo_correlation_stress.py

"""
Correlation-effect walkthrough for riskquant.dependence_stress.

Run from the repository root::

    python scripts/demos/dependence_stress/demo_correlation_stress.py

Scope
-----
The general demonstration, ``demo_dependence_stress.py``, explains the
package ARCHITECTURE: permutations, ranks, RNG ownership, streaming, the
audit path, and the hand-verifiable mechanics.

This script explains the package EFFECT. It answers one question:

    Given the same empirical observations, what happens to the
    cross-sectional dependence structure when each stress is applied?

It therefore does not re-derive the mechanics. It applies each stress to
one fixed matrix and prints what the dependence measures do.

Everything algorithmic comes from the production API. The only logic
owned here is synthetic-data construction, printing, and assertions.
"""

from __future__ import annotations

import math
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import expon, lognorm, norm
from scipy.stats import t as student_t

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"

if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from riskquant.dependence_stress.copulas import GaussianCopula, TCopula  # noqa: E402
from riskquant.dependence_stress.diagnostics import (  # noqa: E402
    empirical_lower_tail_cooccurrence,
    empirical_upper_tail_cooccurrence,
    pearson_correlation,
    spearman_correlation,
)
from riskquant.dependence_stress.engine import DependenceStressEngine  # noqa: E402
from riskquant.dependence_stress.types import (  # noqa: E402
    DependenceStressResult,
    FloatArray,
)
from riskquant.dependence_stress.validation import (  # noqa: E402
    validate_marginal_preservation,
    validate_permutation_matrix,
)

WIDTH = 78
RULE = "=" * WIDTH

SEED = 20260901
NUM_SCENARIOS = 2_000
TAIL_Q = 0.05
COPULA_SEEDS = (100, 200, 300, 400, 500)

LABELS = (
    "A_eq",
    "B_eq",
    "C_eq",
    "D_cr",
    "E_cr",
    "F_hedge",
    "G_idio",
)

K = len(LABELS)


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


def paragraph(text: str) -> None:
    """Print a dedented block of explanatory prose."""
    dedented = textwrap.dedent(text)

    if dedented.startswith("\n\n"):
        print()

    for line in dedented.strip("\n").splitlines():
        print(line.rstrip())


def print_square(
    matrix: FloatArray,
    labels: Sequence[str] = LABELS,
    decimals: int = 4,
    signed: bool = False,
    label_rows: bool = True,
    label_cols: bool = True,
) -> None:
    """
    Print a labelled K x K matrix.

    ``signed`` forces an explicit sign column, which matters for delta
    matrices where the direction of the change is the whole point.
    """
    label_width = (max(len(label) for label in labels) + 2) if label_rows else 0
    cell = 10
    fmt = f"{'+' if signed else ''}{cell}.{decimals}f"

    # Only print column headers if requested
    if label_cols:
        print(" " * label_width + " ".join(f"{label:>{cell}}" for label in labels))

    for row, label in enumerate(labels):
        cells = " ".join(f"{float(value):{fmt}}" for value in matrix[row])
        if label_rows:
            print(f"{label:<{label_width}}" + cells)
        else:
            print(cells)


#: Glyph ramp for the terminal heat map. Deliberately not matplotlib:
#: the demo must stay useful over SSH with no display and no extra
#: dependency. The numeric matrices above remain the authority.
_RAMP = " .:-=+*#%@"


def print_heat(
    matrix: FloatArray,
    labels: Sequence[str] = LABELS,
    low: float = -1.0,
    high: float = 1.0,
) -> None:
    """Render a K x K matrix as a coarse terminal heat map."""
    label_width = max(len(label) for label in labels) + 2
    span = high - low

    print(" " * label_width + " ".join(f"{label[:3]:>3}" for label in labels))

    for row, label in enumerate(labels):
        cells = []

        for value in matrix[row]:
            position = (float(value) - low) / span
            index = min(len(_RAMP) - 1, max(0, int(position * len(_RAMP))))
            cells.append(_RAMP[index] * 3)

        print(f"{label:<{label_width}}" + " ".join(cells))

    print()
    print(
        f"{' ' * label_width}scale: '{_RAMP[0]}' = {low:+.2f}   '{_RAMP[-1]}' = {high:+.2f}"
    )


def check(description: str, condition: bool) -> None:
    """Print a PASS line and fail loudly if the assertion is broken."""
    print(f"    {description:<62} {'PASS' if condition else 'FAIL'}")

    assert condition, f"ASSERTION FAILED: {description}"


def off_diagonal_mean(matrix: FloatArray) -> float:
    """
    Mean over off-diagonal entries only.

    Diagonals are 1.0 by construction for all three diagnostics, so
    including them would dilute every comparison by a constant.
    """
    k = matrix.shape[0]

    return float((matrix.sum() - np.trace(matrix)) / (k * (k - 1)))


# ======================================================================
# Synthetic data
# ======================================================================


def build_synthetic_data() -> tuple[FloatArray, FloatArray]:
    """
    Build a K x S matrix with a deliberately non-trivial structure.

    Construction
    ------------
    1. Draw latent Gaussian scores with a specified correlation.
    2. Map each row through the Gaussian CDF to uniforms.
    3. Map each uniform through a DIFFERENT inverse marginal CDF.

    Step 3 uses strictly increasing maps, so the rank structure of the
    latent draw survives intact while the magnitudes do not. That is
    what makes the raw Spearman matrix track the latent specification
    closely while the raw Pearson matrix does not: exactly the
    latent-versus-observable distinction this package cares about.

    Returns
    -------
    tuple
        The K x S observation matrix and the latent correlation used to
        generate it.
    """
    latent = np.eye(K, dtype=np.float64)

    def link(i: int, j: int, value: float) -> None:
        latent[i, j] = latent[j, i] = value

    # Equity-like block: strong mutual positive dependence.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        link(i, j, 0.75)

    # Credit-like block: strong internally, weaker to the equity block.
    link(3, 4, 0.65)

    for i in (0, 1, 2):
        for j in (3, 4):
            link(i, j, 0.25)

    # A hedge row: genuinely negatively related to the equity block.
    for i in (0, 1, 2):
        link(i, 5, -0.55)

    for j in (3, 4):
        link(j, 5, -0.20)

    # A near-idiosyncratic row.
    for i in range(6):
        link(i, 6, 0.10)

    rng = np.random.default_rng(SEED)

    cholesky = np.linalg.cholesky(latent)
    scores = cholesky @ rng.standard_normal((K, NUM_SCENARIOS))
    uniforms = norm.cdf(scores)

    rows = [
        norm.ppf(uniforms[0], loc=0.0, scale=1.0),
        norm.ppf(uniforms[1], loc=0.0, scale=2.5),
        student_t.ppf(uniforms[2], df=3.5),
        norm.ppf(uniforms[3], loc=-0.5, scale=1.2),
        lognorm.ppf(uniforms[4], s=0.8) - 1.0,
        # Increasing in u so the latent NEGATIVE link survives the
        # marginal map: -expon.ppf(1 - u) rises with u and carries a
        # heavy LEFT tail. Writing -expon.ppf(u) instead would be
        # decreasing, and the two negations would silently cancel the
        # negative relationship this row exists to demonstrate.
        -expon.ppf(1.0 - uniforms[5], scale=1.5),
        norm.ppf(uniforms[6], loc=0.0, scale=0.6),
    ]

    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64), latent


# ======================================================================
# Diagnostic bundle
# ======================================================================


@dataclass(frozen=True, slots=True, eq=False)
class Diagnostics:
    """The complete set of dependence measures for one matrix."""

    pearson: FloatArray
    spearman: FloatArray
    tail: FloatArray
    upper_tail: FloatArray

    @property
    def summary(self) -> tuple[float, float, float, float]:
        """Off-diagonal means of the measures."""
        return (
            off_diagonal_mean(self.pearson),
            off_diagonal_mean(self.spearman),
            off_diagonal_mean(self.tail),
            off_diagonal_mean(self.upper_tail),
        )


def measure(matrix: FloatArray, tail_q: float = TAIL_Q) -> Diagnostics:
    """Compute all package diagnostics for one matrix."""
    return Diagnostics(
        pearson=pearson_correlation(matrix),
        spearman=spearman_correlation(matrix),
        tail=empirical_lower_tail_cooccurrence(matrix, tail_q),
        upper_tail=empirical_upper_tail_cooccurrence(matrix, tail_q),
    )


def assert_diagnostic_shape(diagnostics: Diagnostics, label: str) -> None:
    """Assert symmetry and unit diagonals for all measures."""
    for name, matrix in (
        ("Pearson", diagnostics.pearson),
        ("Spearman", diagnostics.spearman),
        ("lower-tail co-occurrence", diagnostics.tail),
        ("upper-tail co-occurrence", diagnostics.upper_tail),
    ):
        assert matrix.shape == (K, K), f"{label} {name} shape"
        assert np.allclose(matrix, matrix.T, atol=1e-12), f"{label} {name} symmetry"
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-12), f"{label} {name} diagonal"


# ======================================================================
# Stress application
# ======================================================================


@dataclass(frozen=True, slots=True, eq=False)
class StressOutcome:
    """One applied stress, with its reconstruction and diagnostics."""

    name: str
    result: DependenceStressResult
    stressed: FloatArray
    diagnostics: Diagnostics


def apply_stress(
    engine: DependenceStressEngine,
    name: str,
    result: DependenceStressResult,
) -> StressOutcome:
    """
    Reconstruct, audit, and measure one stress result.

    The production path returns only ``permutations`` and
    ``portfolio_pnl``. A correlation study genuinely needs row-level
    observations, so this reconstructs the stressed matrix explicitly.
    That is the legitimate use of the optional derived representation.
    """
    stressed = engine.stressed_matrix(result)

    validate_permutation_matrix(result.permutations, expected_shape=engine.shape)

    assert validate_marginal_preservation(engine.data, stressed)
    assert np.array_equal(stressed.sum(axis=0), result.portfolio_pnl)

    diagnostics = measure(stressed)
    assert_diagnostic_shape(diagnostics, name)

    return StressOutcome(
        name=name,
        result=result,
        stressed=stressed,
        diagnostics=diagnostics,
    )


def report_stress(outcome: StressOutcome, raw: Diagnostics) -> None:
    """Print the stressed matrices and the delta matrices."""
    heading(f"{outcome.name}: stressed Pearson")
    print_square(outcome.diagnostics.pearson)

    heading("PEARSON DELTA = STRESSED - RAW")
    print_square(outcome.diagnostics.pearson - raw.pearson, signed=True)

    heading(f"{outcome.name}: stressed Spearman")
    print_square(outcome.diagnostics.spearman)

    heading("SPEARMAN DELTA = STRESSED - RAW")
    print_square(outcome.diagnostics.spearman - raw.spearman, signed=True)

    heading(f"{outcome.name}: empirical lower-tail co-occurrence (q={TAIL_Q})")
    print_square(outcome.diagnostics.tail)

    heading("LOWER TAIL DELTA = STRESSED - RAW")
    print_square(outcome.diagnostics.tail - raw.tail, signed=True)

    heading(f"{outcome.name}: empirical upper-tail co-occurrence (q={TAIL_Q})")
    print_square(outcome.diagnostics.upper_tail)

    heading("UPPER TAIL DELTA = STRESSED - RAW")
    print_square(outcome.diagnostics.upper_tail - raw.upper_tail, signed=True)

    pearson, spearman, tail, upper_tail = outcome.diagnostics.summary
    raw_pearson, raw_spearman, raw_tail, raw_upper_tail = raw.summary

    heading("Off-diagonal means")

    print()
    print(f"    {'measure':<26}{'before':>10}{'after':>10}{'change':>10}")
    print(f"    {'-' * 26}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}")
    print(
        f"    {'Pearson':<26}{raw_pearson:>10.4f}{pearson:>10.4f}{pearson - raw_pearson:>+10.4f}"
    )
    print(
        f"    {'Spearman':<26}{raw_spearman:>10.4f}{spearman:>10.4f}{spearman - raw_spearman:>+10.4f}"
    )
    print(
        f"    {'lower-tail co-occ.':<26}{raw_tail:>10.4f}{tail:>10.4f}{tail - raw_tail:>+10.4f}"
    )
    print(
        f"    {'upper-tail co-occ.':<26}{raw_upper_tail:>10.4f}{upper_tail:>10.4f}{upper_tail - raw_upper_tail:>+10.4f}"
    )

    print()
    check(
        f"{outcome.name}: marginals preserved exactly",
        validate_marginal_preservation_of(outcome),
    )


def validate_marginal_preservation_of(outcome: StressOutcome) -> bool:
    """Exact marginal check against the module-level base matrix."""
    return bool(
        np.array_equal(
            np.sort(outcome.stressed, axis=1),
            np.sort(BASE_DATA, axis=1),
        )
    )


# ======================================================================
# Section 0: hand-checkable miniature
# ======================================================================


def section_hand_check() -> None:
    """A 3 x 6 example where the correlation change is checkable."""
    banner("0. HAND-CHECKABLE MINIATURE")

    paragraph(
        """
        Before the large example, one tiny case establishes the claim
        that correlation(stressed) is nothing more than the correlation
        of the SAME observations under a different pairing.
        """
    )

    labels = ("X", "Y", "Z")

    small = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [2.0, 4.0, 1.0, 6.0, 3.0, 5.0],
        ]
    )

    heading("Original matrix")
    print_square(
        small,
        labels=("X", "Y", "Z"),
        decimals=1,
        label_cols=False,
    )

    print()
    for row, label in enumerate(labels):
        print(f"    {label}: {small[row].tolist()}")

    heading("Original Spearman")
    print_square(spearman_correlation(small), labels=labels)

    paragraph(
        """

        X ascends and Y descends over the same scenarios, so their rank
        correlation is exactly -1. Now apply the comonotonic permutation,
        which sorts every row ascending.
        """
    )

    engine = DependenceStressEngine(data=small)
    result = engine.apply_comonotonic()
    stressed = engine.stressed_matrix(result)

    heading("Permutation")
    print()
    for row, label in enumerate(labels):
        print(f"    {label}: {result.permutations[row].tolist()}")

    heading("Stressed matrix")
    print()
    for row, label in enumerate(labels):
        print(f"    {label}: {stressed[row].tolist()}")

    stressed_spearman = spearman_correlation(stressed)

    heading("Stressed Spearman")
    print_square(stressed_spearman, labels=labels)

    heading("SPEARMAN DELTA = STRESSED - RAW")
    print_square(
        stressed_spearman - spearman_correlation(small),
        labels=labels,
        signed=True,
    )

    paragraph(
        """

        Y still holds exactly {1, 2, 3, 4, 5, 6}. Not one value was
        created, deleted or rescaled. Its rank correlation with X went
        from -1 to +1 purely because the pairing changed.
        """
    )

    print()
    check(
        "row marginals unchanged",
        np.array_equal(np.sort(stressed, axis=1), np.sort(small, axis=1)),
    )
    check(
        "raw Spearman(X, Y) == -1",
        math.isclose(float(spearman_correlation(small)[0, 1]), -1.0),
    )
    check(
        "stressed Spearman(X, Y) == +1",
        math.isclose(float(stressed_spearman[0, 1]), 1.0),
    )


# ======================================================================
# Section 1-2: data and raw dependence
# ======================================================================


def section_data(latent: FloatArray) -> Diagnostics:
    """Describe the synthetic data and its raw dependence structure."""
    banner("1. SYNTHETIC EMPIRICAL DATA")

    paragraph(
        f"""
        Seed:          {SEED}
        Shape:         {K} variables x {NUM_SCENARIOS} scenarios
        dtype:         {BASE_DATA.dtype}
        C-contiguous:  {BASE_DATA.flags.c_contiguous}

        The rows are deliberately NOT independent and deliberately do
        not share a marginal shape:

            A_eq     standard normal
            B_eq     normal, scale 2.5
            C_eq     Student-t, df 3.5      (heavy tailed)
            D_cr     normal, shifted
            E_cr     shifted lognormal      (right skewed)
            F_hedge  negative exponential   (heavy LEFT tail)
            G_idio   normal, scale 0.6

        The rows were generated from a latent Gaussian structure and
        then pushed through strictly increasing marginal transforms.
        Monotone maps preserve ranks, so the latent rank structure
        survives while the magnitudes do not.
        """
    )

    heading("Latent correlation used to GENERATE the data")

    paragraph(
        """
        This is the generator's specification, printed for reference. It
        is not a stress and not an input to any transformation below.
        """
    )

    print()
    print_square(latent)

    heading("Row summary statistics")

    print()
    print(f"    {'row':<10}{'mean':>10}{'std':>10}{'min':>10}{'5% qtl':>10}{'max':>10}")
    print(
        f"    {'-' * 10}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}"
    )

    for row, label in enumerate(LABELS):
        values = BASE_DATA[row]
        print(
            f"    {label:<10}{values.mean():>10.4f}{values.std():>10.4f}"
            f"{values.min():>10.4f}{np.quantile(values, 0.05):>10.4f}{values.max():>10.4f}"
        )

    paragraph(
        """

        Every number in this table is a function of one row's empirical
        marginal alone. NONE of them will change under any stress in
        this demonstration. Verify that claim in section 9.
        """
    )

    banner("2. RAW DEPENDENCE STRUCTURE")

    raw = measure(BASE_DATA)
    assert_diagnostic_shape(raw, "raw")

    heading("RAW PEARSON")
    print_square(raw.pearson)

    heading("RAW SPEARMAN")
    print_square(raw.spearman)

    heading("RAW EMPIRICAL LOWER-TAIL COOCCURRENCE")

    paragraph(
        f"""
        Finite-sample statistic at q = {TAIL_Q}:

            n_tail = max(1, ceil(q * S)) = {max(1, math.ceil(TAIL_Q * NUM_SCENARIOS))}
            lambda_hat(i,j) = #(scenarios in BOTH lower tails) / n_tail

        This is NOT the asymptotic copula lower-tail dependence
        coefficient. It is an empirical co-occurrence count at a chosen
        threshold, and it depends on that threshold.
        """
    )

    print()
    print_square(raw.tail)

    heading("RAW EMPIRICAL UPPER-TAIL COOCCURRENCE")
    print()
    print_square(raw.upper_tail)

    heading("Raw Spearman, terminal heat map")
    print()
    print_heat(raw.spearman)

    paragraph(
        """

        Read the structure: the A/B/C block is strongly positive, D/E
        form a second weaker block, F_hedge is negative against the
        equity block, and G_idio is nearly independent of everything.

        Note also that raw Pearson and raw Spearman DISAGREE, sometimes
        materially. Both describe the same data. Spearman tracks the
        generating rank structure; Pearson is distorted by the differing
        marginal shapes, especially the heavy-tailed C_eq and the
        skewed E_cr. That gap is not an error: it is the reason the
        package reports both.
        """
    )

    pearson_mean, spearman_mean, tail_mean, upper_tail_mean = raw.summary

    print()
    print(f"    off-diagonal mean Pearson:      {pearson_mean:>8.4f}")
    print(f"    off-diagonal mean Spearman:     {spearman_mean:>8.4f}")
    print(f"    off-diagonal mean lower-tail:   {tail_mean:>8.4f}")
    print(f"    off-diagonal mean upper-tail:   {upper_tail_mean:>8.4f}")

    print()
    check("raw matrix has the expected shape", BASE_DATA.shape == (K, NUM_SCENARIOS))
    check("raw Pearson symmetric", bool(np.allclose(raw.pearson, raw.pearson.T)))
    check("raw Spearman symmetric", bool(np.allclose(raw.spearman, raw.spearman.T)))
    check("raw lower-tail matrix symmetric", bool(np.allclose(raw.tail, raw.tail.T)))
    check(
        "raw upper-tail matrix symmetric",
        bool(np.allclose(raw.upper_tail, raw.upper_tail.T)),
    )
    pearson_diagonal_error = float(np.abs(np.diag(raw.pearson) - 1.0).max())
    spearman_diagonal_error = float(np.abs(np.diag(raw.spearman) - 1.0).max())

    check(
        f"Pearson diagonal is 1 to within {pearson_diagonal_error:.1e}",
        pearson_diagonal_error <= 4.0 * float(np.finfo(np.float64).eps),
    )
    check(
        f"Spearman diagonal is 1 to within {spearman_diagonal_error:.1e}",
        spearman_diagonal_error <= 4.0 * float(np.finfo(np.float64).eps),
    )
    check(
        "tail co-occurrence diagonals are EXACTLY 1",
        np.array_equal(np.diag(raw.tail), np.ones(K))
        and np.array_equal(np.diag(raw.upper_tail), np.ones(K)),
    )

    paragraph(
        """

        One precise point on those three checks. The tail diagonal is
        exactly 1.0 because it is an integer count divided by itself:
        n_tail / n_tail. The Pearson and Spearman diagonals come from
        np.corrcoef, whose normalisation is a floating-point division,
        so a diagonal entry can land one ULP below 1.0. Asserting
        np.array_equal on those diagonals would be wrong, and would fail
        on perfectly correct output. They are checked to within a few
        machine epsilon instead.
        """
    )

    return raw


# ======================================================================
# Section 3: comonotonic
# ======================================================================


def section_comonotonic(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> StressOutcome:
    """Apply and interpret the comonotonic benchmark."""
    banner("3. COMONOTONIC STRESS")

    paragraph(
        """
        Comonotonicity aligns every row's worst observation into the
        same scenario, every second-worst into the next, and so on. It
        is the maximum positive RANK alignment available under the given
        empirical marginals.

        It is deterministic and needs no generator.
        """
    )

    outcome = apply_stress(engine, "Comonotonic", engine.apply_comonotonic())

    report_stress(outcome, raw)

    heading("Why Pearson does not reach 1")

    paragraph(
        """
        Comonotonicity is a RANK property. Pearson correlation reaches 1
        only when two rows are increasing affine transforms of one
        another. These rows have different marginal shapes, so their
        comonotonic Pearson entries sit BELOW 1, sometimes well below.

        Look at the pairs involving C_eq (Student-t, df 3.5) and E_cr
        (lognormal). Their extreme values are shaped very differently
        from a normal row's, so aligning ranks does not align magnitudes.
        The correct statement is maximality, not equality with 1.
        """
    )

    pearson = outcome.diagnostics.pearson
    spearman = outcome.diagnostics.spearman

    print()
    print(
        f"    minimum off-diagonal comonotonic Pearson:   "
        f"{float(pearson[~np.eye(K, dtype=bool)].min()):>8.4f}"
    )
    print(
        f"    maximum off-diagonal comonotonic Pearson:   "
        f"{float(pearson[~np.eye(K, dtype=bool)].max()):>8.4f}"
    )
    print(
        f"    minimum off-diagonal comonotonic Spearman:  "
        f"{float(spearman[~np.eye(K, dtype=bool)].min()):>8.4f}"
    )

    heading("Maximality check against random rearrangements")

    paragraph(
        """
        The honest invariant is the rearrangement inequality: no row-wise
        permutation of these observations achieves a higher pairwise
        Spearman correlation than the comonotonic one. Tested against
        random rearrangements below.
        """
    )

    rng = np.random.default_rng(7)
    beaten = 0

    for _ in range(200):
        shuffled = np.take_along_axis(
            BASE_DATA,
            np.argsort(rng.random(BASE_DATA.shape), axis=1),
            axis=1,
        )

        if np.any(spearman_correlation(shuffled) > spearman + 1e-12):
            beaten += 1

    print()
    check(
        "no random rearrangement beats comonotonic Spearman (200 draws)",
        beaten == 0,
    )
    check(
        "every off-diagonal Spearman rose to the achievable maximum",
        bool(np.all(spearman >= raw.spearman - 1e-12)),
    )
    check(
        "F_hedge flipped from negative to positive rank dependence",
        raw.spearman[0, 5] < -0.3 < 0.3 < spearman[0, 5],
    )

    return outcome


# ======================================================================
# Section 4A: lower-tail alignment
# ======================================================================


def section_lower_tail(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> StressOutcome:
    """Apply and interpret lower-tail alignment."""
    banner("4A. LOWER-TAIL STRESS")

    n_tail = max(1, math.ceil(TAIL_Q * NUM_SCENARIOS))

    paragraph(
        f"""
        Configuration:

            tail_probability        = {TAIL_Q}
            tail_order              = "comonotonic"
            preserve_non_tail_order = True

            n_tail = max(1, ceil({TAIL_Q} * {NUM_SCENARIOS})) = {n_tail}

        q = {TAIL_Q} is chosen to match the conventional 5% tail used by
        downstream risk work, and it leaves {n_tail} scenarios in the
        aligned block: enough to be statistically meaningful, small
        enough that the rest of the matrix is genuinely untouched.

        This stress does NOT try to maximise overall correlation. It
        moves each row's {n_tail} worst observations into the same
        scenario slots and leaves the relative ordering of everything
        else alone.
        """
    )

    outcome = apply_stress(
        engine,
        f"LowerTailAlignment(q={TAIL_Q})",
        engine.apply_lower_tail_alignment(TAIL_Q),
    )

    report_stress(outcome, raw)

    heading("Targeted, not global")

    raw_pearson, raw_spearman, raw_tail, _ = raw.summary
    pearson, spearman, tail, _ = outcome.diagnostics.summary

    comonotonic_tail = 1.0

    paragraph(
        f"""
        Compare the three movements:

            Pearson             {raw_pearson:+.4f} -> {pearson:+.4f}   ({pearson - raw_pearson:+.4f})
            Spearman            {raw_spearman:+.4f} -> {spearman:+.4f}   ({spearman - raw_spearman:+.4f})
            lower-tail co-occ.  {raw_tail:+.4f} -> {tail:+.4f}   ({tail - raw_tail:+.4f})

        The tail statistic moves to its ceiling of {comonotonic_tail:.1f}: by
        construction, every row now shares the same {n_tail} worst
        scenarios. The overall correlations move far less, because only
        {n_tail} of {NUM_SCENARIOS} scenarios were rearranged.

        This is the case that a single overall correlation coefficient
        would badly understate. A reviewer looking only at mean Spearman
        would conclude that little happened. The downside co-occurrence
        says otherwise, and it is the downside that drives a capital
        charge.
        """
    )

    print()
    check(
        "lower-tail co-occurrence increased materially",
        tail > raw_tail + 0.3,
    )
    check(
        "lower-tail co-occurrence reached its ceiling",
        bool(np.allclose(outcome.diagnostics.tail, 1.0)),
    )
    check(
        "overall Spearman moved far less than the tail statistic",
        abs(spearman - raw_spearman) < (tail - raw_tail),
    )

    return outcome


# ======================================================================
# Section 4B: upper-tail alignment
# ======================================================================


def section_upper_tail(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> StressOutcome:
    """Apply and interpret upper-tail alignment."""
    banner("4B. UPPER-TAIL STRESS")

    n_tail = max(1, math.ceil(TAIL_Q * NUM_SCENARIOS))

    paragraph(
        f"""
        Configuration:

            tail_probability        = {TAIL_Q}
            tail_order              = "comonotonic"
            preserve_non_tail_order = True

            n_tail = max(1, ceil({TAIL_Q} * {NUM_SCENARIOS})) = {n_tail}

        UpperTailAlignment functions identically to LowerTailAlignment but
        operates on the highest values in the distribution. This ensures
        co-occurrence of right-tail extremes, which is critical when
        applying stresses to risk factors (where portfolio exposure can
        be short).
        """
    )

    outcome = apply_stress(
        engine,
        f"UpperTailAlignment(q={TAIL_Q})",
        engine.apply_upper_tail_alignment(TAIL_Q),
    )

    report_stress(outcome, raw)

    _, _, raw_tail, raw_upper_tail = raw.summary
    _, _, tail, upper_tail = outcome.diagnostics.summary

    print()
    check(
        "upper-tail co-occurrence increased materially",
        upper_tail > raw_upper_tail + 0.3,
    )
    check(
        "upper-tail co-occurrence reached its ceiling",
        bool(np.allclose(outcome.diagnostics.upper_tail, 1.0)),
    )
    check(
        "lower-tail co-occurrence moved far less than the upper-tail statistic",
        abs(tail - raw_tail) < (upper_tail - raw_upper_tail),
    )

    return outcome


# ======================================================================
# Section 4C: dual-tail alignment
# ======================================================================


def section_dual_tail(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> StressOutcome:
    """Apply and interpret dual-tail alignment."""
    banner("4C. DUAL-TAIL STRESS")

    n_tail = max(1, math.ceil(TAIL_Q * NUM_SCENARIOS))

    paragraph(
        f"""
        Configuration:

            lower_probability       = {TAIL_Q}
            upper_probability       = {TAIL_Q}
            tail_order              = "comonotonic"
            preserve_non_tail_order = True

            n_lower = n_upper = max(1, ceil({TAIL_Q} * {NUM_SCENARIOS})) = {n_tail}

        DualTailAlignment enforces extreme co-occurrence at BOTH ends
        of the marginal distributions simultaneously. It groups the lowest
        observations into a shared left-tail block, and the highest into a
        shared right-tail block, leaving the center independent.
        """
    )

    outcome = apply_stress(
        engine,
        f"DualTailAlignment(q={TAIL_Q})",
        engine.apply_dual_tail_alignment(TAIL_Q, TAIL_Q),
    )

    report_stress(outcome, raw)

    raw_pearson, raw_spearman, raw_tail, raw_upper_tail = raw.summary
    pearson, spearman, tail, upper_tail = outcome.diagnostics.summary

    print()
    check(
        "lower-tail co-occurrence reached its ceiling",
        bool(np.allclose(outcome.diagnostics.tail, 1.0)),
    )
    check(
        "upper-tail co-occurrence reached its ceiling",
        bool(np.allclose(outcome.diagnostics.upper_tail, 1.0)),
    )

    return outcome


# ======================================================================
# Section 5: Gaussian copula
# ======================================================================


def stress_latent_correlation() -> FloatArray:
    """
    An explicit stress specification, not the raw empirical matrix.

    Deliberately different from the generating structure: the equity
    block is pushed higher, the credit block is pushed higher, the
    cross-block link is raised, and the hedge row's negative
    relationship is REVERSED to positive. This is the classic stress
    narrative that diversification fails when it is most needed.
    """
    latent = np.full((K, K), 0.60, dtype=np.float64)
    np.fill_diagonal(latent, 1.0)

    def link(i: int, j: int, value: float) -> None:
        latent[i, j] = latent[j, i] = value

    for i, j in ((0, 1), (0, 2), (1, 2)):
        link(i, j, 0.90)

    link(3, 4, 0.85)

    for i in (0, 1, 2):
        link(i, 5, 0.55)

    for i in range(6):
        link(i, 6, 0.40)

    return latent


def _copula_ensemble(
    engine: DependenceStressEngine,
    transformer: GaussianCopula | TCopula,
    label: str,
) -> tuple[StressOutcome, dict[str, tuple[float, float, float]]]:
    """
    Run one stochastic specification across several seeds.

    Returns the first realisation for matrix display plus per-measure
    (mean, min, max) across the ensemble. A single stochastic draw does
    not establish a distributional property, so nothing here asserts on
    one realisation.
    """
    outcomes = [
        apply_stress(
            engine,
            f"{label} seed={seed}",
            engine.apply_transformer(transformer, rng=np.random.default_rng(seed)),
        )
        for seed in COPULA_SEEDS
    ]

    summaries = np.array([outcome.diagnostics.summary for outcome in outcomes])

    spread = {
        name: (
            float(summaries[:, index].mean()),
            float(summaries[:, index].min()),
            float(summaries[:, index].max()),
        )
        for index, name in enumerate(("pearson", "spearman", "tail", "upper_tail"))
    }

    return outcomes[0], spread


def _print_ensemble(spread: dict[str, tuple[float, float, float]]) -> None:
    """Print the mean and range of each measure across seeds."""
    print()
    print(f"    {'measure':<22}{'mean':>10}{'min':>10}{'max':>10}{'range':>10}")
    print(f"    {'-' * 22}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}{'-' * 10:>10}")

    for name, (mean, low, high) in spread.items():
        print(
            f"    {name:<22}{mean:>10.4f}{low:>10.4f}{high:>10.4f}{high - low:>10.4f}"
        )


def section_gaussian_copula(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> StressOutcome:
    """Apply and interpret the Gaussian copula stress."""
    banner("5. GAUSSIAN COPULA STRESS")

    latent = stress_latent_correlation()

    heading("Latent correlation STRESS SPECIFICATION")

    paragraph(
        """
        This matrix is an explicit stress specification. It is not the
        raw empirical correlation matrix and it is not the generating
        structure. Note in particular that F_hedge's relationship with
        the equity block has been forced from -0.55 to +0.55: the hedge
        is specified to stop working.
        """
    )

    print()
    print_square(latent)

    outcome, spread = _copula_ensemble(
        engine,
        GaussianCopula(latent_correlation=latent),
        "GaussianCopula",
    )

    paragraph(
        f"""

        Matrices below are the realisation from seed {COPULA_SEEDS[0]}.
        The ensemble across seeds {list(COPULA_SEEDS)} follows.
        """
    )

    report_stress(outcome, raw)

    heading(f"Ensemble across {len(COPULA_SEEDS)} seeds")
    _print_ensemble(spread)

    heading("LATENT IS NOT OBSERVABLE")

    equity_rho = 0.90
    implied = (6.0 / math.pi) * math.asin(equity_rho / 2.0)
    realised = float(outcome.diagnostics.spearman[0, 1])

    paragraph(
        f"""
        The specification says the latent Gaussian correlation of A_eq
        and B_eq is {equity_rho:.2f}. It does NOT say the realised
        empirical Spearman correlation will be {equity_rho:.2f}.

        For a Gaussian copula the population relationship is:

            rho_S = (6 / pi) * arcsin(rho / 2)
                  = (6 / pi) * arcsin({equity_rho:.2f} / 2)
                  = {implied:.4f}

            realised finite-sample Spearman(A_eq, B_eq) = {realised:.4f}

        The finite-sample value fluctuates around the population value.
        It is a realisation, not the specification. Saying "the Gaussian
        copula sets the correlation to {equity_rho:.2f}" is only correct
        if "correlation" means the LATENT Gaussian parameter.
        """
    )

    print()
    check(
        "realised Spearman is near the implied population value",
        abs(realised - implied) < 0.05,
    )
    check(
        "specified positive stress raised overall rank dependence",
        spread["spearman"][0] > off_diagonal_mean(raw.spearman),
    )
    check(
        "the hedge relationship was reversed as specified",
        raw.spearman[0, 5] < 0.0 < outcome.diagnostics.spearman[0, 5],
    )

    return outcome


# ======================================================================
# Section 6: t copula
# ======================================================================


def section_t_copula(
    engine: DependenceStressEngine,
    raw: Diagnostics,
) -> tuple[StressOutcome, StressOutcome]:
    """Contrast two degrees of freedom under one latent correlation."""
    banner("6. T-COPULA STRESS")

    latent = stress_latent_correlation()

    paragraph(
        """
        The t copula holds the same latent correlation and varies one
        further parameter: the degrees of freedom nu. The comparison
        below fixes the latent matrix and changes only nu.

        Lower nu does NOT mean "higher correlation". Lower nu means a
        heavier-tailed latent distribution and therefore STRONGER
        SYMMETRIC TAIL DEPENDENCE for a fixed latent correlation. The
        construction shares one chi-square scale per scenario across all
        K rows, and it is that shared scale which creates joint extremes.
        """
    )

    outcomes: dict[float, StressOutcome] = {}
    spreads: dict[float, dict[str, tuple[float, float, float]]] = {}

    for nu in (30.0, 3.0):
        outcome, spread = _copula_ensemble(
            engine,
            TCopula(latent_correlation=latent, degrees_of_freedom=nu),
            f"TCopula(nu={nu:g})",
        )
        outcomes[nu] = outcome
        spreads[nu] = spread

        heading(f"nu = {nu:g}: ensemble across {len(COPULA_SEEDS)} seeds")
        _print_ensemble(spread)

    heading("nu = 3: stressed Spearman")
    print_square(outcomes[3.0].diagnostics.spearman)

    heading("nu = 3: SPEARMAN DELTA = STRESSED - RAW")
    print_square(outcomes[3.0].diagnostics.spearman - raw.spearman, signed=True)

    heading("nu = 3: empirical lower-tail co-occurrence")
    print_square(outcomes[3.0].diagnostics.tail)

    heading("nu = 30 versus nu = 3")

    high_spearman = spreads[30.0]["spearman"][0]
    low_spearman = spreads[3.0]["spearman"][0]
    high_tail = spreads[30.0]["tail"][0]
    low_tail = spreads[3.0]["tail"][0]
    high_upper = spreads[30.0]["upper_tail"][0]
    low_upper = spreads[3.0]["upper_tail"][0]

    print()
    print(f"    {'':<14}{'mean Spearman':>16}{'mean tail 5%':>16}{'mean upper 5%':>16}")
    print(f"    {'-' * 14}{'-' * 16:>16}{'-' * 16:>16}{'-' * 16:>16}")
    print(
        f"    {'nu = 30':<14}{high_spearman:>16.4f}{high_tail:>16.4f}{high_upper:>16.4f}"
    )
    print(f"    {'nu = 3':<14}{low_spearman:>16.4f}{low_tail:>16.4f}{low_upper:>16.4f}")
    print(
        f"    {'difference':<14}{low_spearman - high_spearman:>+16.4f}{low_tail - high_tail:>+16.4f}{low_upper - high_upper:>+16.4f}"
    )

    paragraph(
        f"""

        Broad rank correlation is comparable between the two settings
        ({high_spearman:.4f} against {low_spearman:.4f}). The tail
        co-occurrence (both upper and lower) is materially higher at nu = 3.

        This is exactly why the empirical tail statistics earn their
        place alongside Pearson and Spearman. Two specifications that a
        single correlation number would call nearly identical behave
        very differently precisely where a risk measure looks.
        """
    )

    print()
    check(
        "lower nu produced stronger tail co-occurrence (mean over seeds)",
        low_tail > high_tail and low_upper > high_upper,
    )
    check(
        "broad rank correlation stayed comparable between nu settings",
        abs(low_spearman - high_spearman) < abs(low_tail - high_tail),
    )
    check(
        "both nu settings preserved marginals exactly",
        validate_marginal_preservation_of(outcomes[3.0])
        and validate_marginal_preservation_of(outcomes[30.0]),
    )

    return outcomes[30.0], outcomes[3.0]


# ======================================================================
# Section 7: comparison
# ======================================================================


def section_comparison(
    raw: Diagnostics,
    outcomes: Sequence[tuple[str, StressOutcome]],
) -> None:
    """Print the compact cross-method summary table."""
    banner("7. CORRELATION DELTA COMPARISON")

    paragraph(
        """
        All means below are over OFF-DIAGONAL pairwise entries only.
        Diagonals are 1.0 by construction for all three measures, so
        including them would add a constant and dilute every comparison.

        Deltas are measured against the raw matrix.
        """
    )

    raw_pearson, raw_spearman, raw_tail, raw_upper_tail = raw.summary

    print()
    header = (
        f"    {'Method':<30}{'Pearson':>9}{'Spearman':>10}{'L-Tail':>8}{'U-Tail':>8}"
        f"{'dPearson':>10}{'dSpearman':>11}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    print(
        f"    {'Raw (unstressed)':<30}{raw_pearson:>9.4f}{raw_spearman:>10.4f}"
        f"{raw_tail:>8.4f}{raw_upper_tail:>8.4f}{'-':>10}{'-':>11}"
    )

    for name, outcome in outcomes:
        pearson, spearman, tail, upper_tail = outcome.diagnostics.summary
        print(
            f"    {name:<30}{pearson:>9.4f}{spearman:>10.4f}{tail:>8.4f}{upper_tail:>8.4f}"
            f"{pearson - raw_pearson:>+10.4f}{spearman - raw_spearman:>+11.4f}"
        )

    paragraph(
        """

        Read the table by column, not by row. Each stress has a
        signature:

        Comonotonic          moves everything to the achievable maximum.
                             The benchmark, not a plausible scenario.

        Lower-tail q=0.05    L-Tail column at its ceiling, correlation
                             columns barely moved. A surgical stress.

        Dual-tail q=0.05     Both L-Tail and U-Tail at their ceilings,
                             center is untouched.

        Gaussian copula      broad, uniform lift across the correlation
                             columns; tails rise as a by-product.

        t copula nu=30       close to the Gaussian case, as expected.

        t copula nu=3        similar correlation columns to nu=30, but
                             clearly higher tail columns.

        The last two rows are the argument for reporting multiple
        measures. A summary that reported only mean Spearman would treat
        nu=3 and nu=30 as interchangeable.
        """
    )


# ======================================================================
# Section 8: portfolio consequence
# ======================================================================


def section_portfolio(
    engine: DependenceStressEngine,
    outcomes: Sequence[tuple[str, StressOutcome]],
) -> None:
    """Show the diversification consequence at portfolio level."""
    banner("8. PORTFOLIO DISTRIBUTION CONSEQUENCE")

    paragraph(
        """
        The package's second first-class output is the scenario vector:

            portfolio_pnl[s] = sum_k data[k, permutations[k, s]]

        This is a plain additive aggregation of the supplied rows. The
        package attaches no portfolio semantics to them and computes no
        VaR, ES or CSC. The statistics below are DESCRIPTIVE only; a
        downstream risk engine owns the actual measures.
        """
    )

    raw_portfolio = engine.data.sum(axis=0)

    heading("Descriptive statistics of the scenario vector")

    print()
    header = f"    {'Method':<26}{'mean':>10}{'std':>10}{'min':>10}{'5% qtl':>10}{'95% qtl':>10}{'max':>10}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    def line(name: str, values: FloatArray) -> None:
        print(
            f"    {name:<26}{values.mean():>10.3f}{values.std():>10.3f}"
            f"{values.min():>10.3f}{np.quantile(values, 0.05):>10.3f}{np.quantile(values, 0.95):>10.3f}{values.max():>10.3f}"
        )

    line("Raw (unstressed)", raw_portfolio)

    for name, outcome in outcomes:
        line(name, outcome.result.portfolio_pnl)

    paragraph(
        """

        The mean column is constant across every row. That is not a
        coincidence and not luck: the portfolio total across all
        scenarios is the sum of fixed row totals, and rearranging which
        scenario a value sits in cannot change a row's total.

        Every other column moves. Standard deviation rises as positive
        dependence is imposed, the minimum falls, and the 5% quantile
        falls. That is diversification benefit being removed.
        """
    )

    print()
    check(
        "portfolio mean identical across every stress",
        all(
            math.isclose(
                float(outcome.result.portfolio_pnl.mean()),
                float(raw_portfolio.mean()),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for _, outcome in outcomes
        ),
    )

    comonotonic = dict(outcomes)["Comonotonic"]

    check(
        "comonotonic stress widened the scenario distribution",
        float(comonotonic.result.portfolio_pnl.std()) > float(raw_portfolio.std()),
    )
    check(
        "comonotonic stress lowered the 5% scenario quantile",
        float(np.quantile(comonotonic.result.portfolio_pnl, 0.05))
        < float(np.quantile(raw_portfolio, 0.05)),
    )

    heading("The connection to a downstream charge")

    tail_outcome = dict(outcomes)[f"LowerTailAlignment(q={TAIL_Q})"]
    raw_q05 = float(np.quantile(raw_portfolio, 0.05))
    tail_q05 = float(np.quantile(tail_outcome.result.portfolio_pnl, 0.05))

    paragraph(
        f"""
        Take the lower-tail stress specifically. Its mean Spearman
        barely moved, so a correlation-only review would call it a mild
        scenario. Its effect on the 5% scenario quantile:

            raw       5% quantile = {raw_q05:>10.3f}
            stressed  5% quantile = {tail_q05:>10.3f}
            change                = {tail_q05 - raw_q05:>+10.3f}

        A downstream measure that reads the lower tail of this vector
        sees a materially worse distribution, produced without inventing
        a single new observation. That is the mechanism by which a
        dependence stress feeds a Correlation Stress Charge.
        """
    )

    print()
    check(
        "lower-tail stress worsened the 5% scenario quantile",
        tail_q05 < raw_q05,
    )


# ======================================================================
# Section 9: marginal preservation audit
# ======================================================================


def section_marginal_audit(outcomes: Sequence[tuple[str, StressOutcome]]) -> None:
    """Prove that none of the correlation changes came from the data."""
    banner("9. MARGINAL PRESERVATION AUDIT")

    paragraph(
        """
                    SAME ROW VALUES
                           +
             DIFFERENT CROSS-SECTIONAL PAIRING
                           =
             DIFFERENT DEPENDENCE STRUCTURE

        Every correlation change shown in this demonstration was caused
        by re-pairing observations across rows. None of it came from
        altering, rescaling, or simulating observations. The check below
        is exact, not tolerance-based: a stressed value is one of the
        original float64 values, bit for bit.
        """
    )

    print()

    for name, outcome in outcomes:
        check(
            f"{name}: marginals preserved exactly",
            validate_marginal_preservation_of(outcome),
        )

    heading("Row statistics, before and after every stress")

    paragraph(
        """
        Each cell is a statistic of one row alone. Under every stress,
        every cell is unchanged.
        """
    )

    print()
    print(
        f"    {'row':<10}{'statistic':<12}{'raw':>12}"
        + "".join(f"{name.split('(')[0][:10]:>12}" for name, _ in outcomes)
    )
    print("    " + "-" * (34 + 12 * len(outcomes)))

    for row in (0, 2, 5):
        for statistic, function in (
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
        ):
            cells = "".join(
                f"{float(function(outcome.stressed[row])):>12.4f}"
                for _, outcome in outcomes
            )
            print(
                f"    {LABELS[row]:<10}{statistic:<12}"
                f"{float(function(BASE_DATA[row])):>12.4f}{cells}"
            )

    print()

    for row in range(K):
        check(
            f"{LABELS[row]}: every stress leaves the sorted row identical",
            all(
                np.array_equal(
                    np.sort(outcome.stressed[row]),
                    np.sort(BASE_DATA[row]),
                )
                for _, outcome in outcomes
            ),
        )


# ======================================================================
# Section 10: conclusions
# ======================================================================


def section_conclusions() -> None:
    """Close with the interpretation."""
    banner("10. CONCLUSIONS")

    paragraph(
        """
        The stress does not change what observations exist in any
        individual variable. It changes which observations occur
        together.

        Therefore the marginal distributions remain exactly fixed, while
        the cross-sectional dependence structure changes.

        Pearson measures the resulting linear dependence in the
        empirical magnitudes. It is sensitive to marginal shape, which
        is why comonotonic alignment does not drive it to 1.

        Spearman measures the resulting rank dependence. It is the
        measure the permutation mechanism acts on most directly.

        Tail co-occurrence isolates the extreme dependence targeted by
        tail stresses (whether lower, upper, or dual), and reveals effects
        that a single overall correlation coefficient can obscure. The
        t-copula comparison at nu = 3 against nu = 30 is the clearest
        case: near identical rank correlation, materially different
        downside behaviour.

        The resulting portfolio P&L distribution can therefore change
        materially even though every standalone marginal is unchanged.

        That is the mathematical mechanism by which DEPENDENCE_STRESS
        alters diversification and ultimately affects a downstream risk
        measure such as VaR, ES, or a Correlation Stress Charge. Those
        measures are computed downstream. This package stops at the
        stressed scenario vector.
        """
    )

    heading("A caution on the stochastic methods")

    paragraph(
        """
        The Gaussian and t copula sections reported ensembles across
        several seeds rather than one realisation, because a single
        stochastic draw does not establish a distributional property. A
        copula specification induces a DISTRIBUTION over finite-sample
        permutations, not one determined permutation. The seed provides
        reproducibility, not uniqueness, and the spread reported in
        sections 5 and 6 is the size of that realisation noise.
        """
    )


# ======================================================================
# Entry point
# ======================================================================


BASE_DATA, LATENT_GENERATOR = build_synthetic_data()


def main() -> int:
    """Run the correlation-effect walkthrough."""
    banner("CORRELATION STRESS DEMONSTRATION")

    paragraph(
        """
        This script measures the EFFECT of each dependence stress on the
        cross-sectional dependence structure. It does not re-derive the
        mechanics; demo_dependence_stress.py covers permutations, ranks,
        RNG ownership, streaming and the audit architecture.
        """
    )

    section_hand_check()

    raw = section_data(LATENT_GENERATOR)

    engine = DependenceStressEngine(data=BASE_DATA)

    comonotonic = section_comonotonic(engine, raw)
    lower_tail = section_lower_tail(engine, raw)
    upper_tail = section_upper_tail(engine, raw)
    dual_tail = section_dual_tail(engine, raw)
    gaussian = section_gaussian_copula(engine, raw)
    t_high, t_low = section_t_copula(engine, raw)

    outcomes = [
        ("Comonotonic", comonotonic),
        (f"LowerTailAlignment(q={TAIL_Q})", lower_tail),
        (f"UpperTailAlignment(q={TAIL_Q})", upper_tail),
        (f"DualTailAlignment(q={TAIL_Q})", dual_tail),
        ("GaussianCopula", gaussian),
        ("TCopula(nu=30)", t_high),
        ("TCopula(nu=3)", t_low),
    ]

    section_comparison(raw, outcomes)
    section_portfolio(engine, outcomes)
    section_marginal_audit(outcomes)
    section_conclusions()

    print()
    print(RULE)
    print("END OF DEMO: ALL CORRELATION-DEMO ASSERTIONS PASSED!")
    print(RULE)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
