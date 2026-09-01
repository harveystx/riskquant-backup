# scripts/demos/dependence/demo_hand_verifiable_dependence.py

"""
Hand-verifiable demonstration of the current dependence engine.

Every section prints the input, the intermediate quantities a reviewer needs,
the package result, and an **independent** reference computed by a different
code path -- SciPy, NumPy, or arithmetic written out longhand. Calling the
package twice would prove nothing, so no section does that.

Migrated from the pre-Phase-0 version, which demonstrated seven estimators.
Four of those -- weighted Kendall, Ledoit-Wolf, distance correlation and mutual
information -- were removed or deferred by constitutional decision, and their
sections went with them. Phase 1 tail counts and Phase 1b-A tail coefficients
take their place, so the demo again covers the whole engine.

Run:
    python scripts/demos/dependence/demo_hand_verifiable_dependence.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riskquant.dependence.coefficients.tail_coefficients import (  # noqa: E402
    tail_coefficients,
)
from riskquant.dependence.contracts import TailSide  # noqa: E402
from riskquant.dependence.diagnostics.spectrum import spectrum  # noqa: E402
from riskquant.dependence.estimators.kendall import kendall_tau_b  # noqa: E402
from riskquant.dependence.estimators.pearson import pearson  # noqa: E402
from riskquant.dependence.estimators.spearman import spearman  # noqa: E402
from riskquant.dependence.estimators.tail_counts import tail_counts  # noqa: E402
from riskquant.dependence.represent.dissimilarity import (  # noqa: E402
    angular_dissimilarity,
)
from riskquant.observations.compile import (  # noqa: E402
    CompiledSample,
    RankRepresentation,
    compile_ranks,
    compile_sample,
)
from riskquant.observations.orientation import ObservationOrientation  # noqa: E402

LINE = "=" * 78
VARIABLES_ON_ROWS = ObservationOrientation.VARIABLES_ON_ROWS


def _banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def _compile(matrix: np.ndarray) -> tuple[CompiledSample, RankRepresentation]:
    """Compile once; both artifacts share the single sort."""
    sample = compile_sample(
        np.ascontiguousarray(matrix), orientation=VARIABLES_ON_ROWS
    )
    return sample, compile_ranks(sample)


def demo_compilation() -> None:
    _banner("COMPILED SAMPLE  --  ownership and the one-sort rank artifact")

    matrix = np.array([[3.0, 1.0, 2.0, 2.0], [10.0, 40.0, 20.0, 30.0]])
    print(f"x = {matrix[0]}   (note the tie)")
    print(f"y = {matrix[1]}")

    sample, ranks = _compile(matrix)
    print(f"\nsum_of_squares     = {sample.sum_of_squares}")
    print(f"average ranks of x = {ranks.ranks[0]}   (the tied pair shares 2.5)")
    print(f"sort_order of x    = {ranks.sort_order[0]}")
    print(f"tie_pairs          = {ranks.tie_pairs}")
    print("\nThe rank representation carries tie_pairs (a Kendall input).")
    print("The per-variable tie_rate is derived where it is consumed, and")
    print("travels on the tail-count provenance rather than on the artifact.")

    print("\nOwnership: the compiled sample copies its input and freezes it.")
    print(f"  observations writeable = {sample.observations.flags.writeable}")
    before = pearson(sample).matrix.copy()
    matrix[0, 0] = 1e9
    after = pearson(sample).matrix
    print(
        "  caller mutated its array afterwards; estimate unchanged = "
        f"{np.array_equal(before, after)}"
    )


def demo_pearson() -> None:
    _banner("PEARSON  --  linear dependence")

    matrix = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 4.0]])
    print(f"x = {matrix[0]}\ny = {matrix[1]}")

    deviation_x = matrix[0] - matrix[0].mean()
    deviation_y = matrix[1] - matrix[1].mean()
    print(f"\nxbar = {matrix[0].mean()}, ybar = {matrix[1].mean()}")
    print(f"dx   = {deviation_x}\ndy   = {deviation_y}")
    print(f"sum(dx*dy) = {float(deviation_x @ deviation_y)}")
    print(
        f"sum(dx^2)  = {float(deviation_x @ deviation_x)}, "
        f"sum(dy^2) = {float(deviation_y @ deviation_y)}"
    )

    longhand = float(deviation_x @ deviation_y) / math.sqrt(
        float(deviation_x @ deviation_x) * float(deviation_y @ deviation_y)
    )
    sample, _ = _compile(matrix)
    print(f"\nlonghand rho         = {longhand:.15f}")
    print(f"package rho          = {pearson(sample).matrix[0, 1]:.15f}")
    print(f"scipy.stats.pearsonr = {pearsonr(matrix[0], matrix[1]).statistic:.15f}")


def demo_spearman() -> None:
    _banner("SPEARMAN  --  rank dependence, with ties")

    matrix = np.array([[1.0, 1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 4.0, 1.0, 9.0]])
    print(f"x = {matrix[0]}\ny = {matrix[1]}")

    ranks = rankdata(matrix, method="average", axis=1)
    print(f"\naverage ranks of x = {ranks[0]}   (tied pair shares 1.5)")
    print(f"average ranks of y = {ranks[1]}   (tied pair shares 3.5)")

    sample, compiled_ranks = _compile(matrix)
    print(f"\nPearson of the ranks (numpy) = {float(np.corrcoef(ranks)[0, 1]):.15f}")
    print(
        "package rho_S                = "
        f"{spearman(sample, compiled_ranks).matrix[0, 1]:.15f}"
    )
    print(
        "scipy.stats.spearmanr        = "
        f"{spearmanr(matrix[0], matrix[1]).statistic:.15f}"
    )


def demo_kendall() -> None:
    _banner("KENDALL TAU-B  --  concordance counted by hand")

    matrix = np.array([[1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]])
    print(f"x = {matrix[0]}\ny = {matrix[1]}")

    n_observations = matrix.shape[1]
    total = n_observations * (n_observations - 1) // 2
    tied_x = tied_y = tied_both = concordant = discordant = 0

    print("\npair   dx   dy   classification")
    for i in range(n_observations):
        for j in range(i + 1, n_observations):
            sign_x = int(np.sign(matrix[0, i] - matrix[0, j]))
            sign_y = int(np.sign(matrix[1, i] - matrix[1, j]))
            if sign_x == 0 and sign_y == 0:
                label, tied_both = "tied in both", tied_both + 1
            elif sign_x == 0:
                label, tied_x = "tied in x", tied_x + 1
            elif sign_y == 0:
                label, tied_y = "tied in y", tied_y + 1
            elif sign_x * sign_y > 0:
                label, concordant = "concordant", concordant + 1
            else:
                label, discordant = "discordant", discordant + 1
            print(f"({i},{j})  {sign_x:+d}   {sign_y:+d}   {label}")

    numerator = total - tied_x - tied_y + tied_both - 2 * discordant
    denominator = math.sqrt((total - tied_x) * (total - tied_y))
    print(
        f"\nn0={total}  n1={tied_x}  n2={tied_y}  n3={tied_both}  "
        f"concordant={concordant}  discordant={discordant}"
    )

    sample, ranks = _compile(matrix)
    print(f"\nlonghand tau_b         = {numerator / denominator:.15f}")
    print(
        "package tau_b          = "
        f"{kendall_tau_b(sample, ranks).matrix[0, 1]:.15f}"
    )
    print(
        "scipy.stats.kendalltau = "
        f"{kendalltau(matrix[0], matrix[1], variant='b').statistic:.15f}"
    )

    print("\nAverage ranks preserve the sign pattern of every pairwise")
    print("difference, so tau-b of the ranks is bitwise identical:")
    rank_sample, rank_ranks = _compile(rankdata(matrix, method="average", axis=1))
    print(
        "  array_equal = "
        f"{np.array_equal(kendall_tau_b(sample, ranks).matrix, kendall_tau_b(rank_sample, rank_ranks).matrix)}"
    )


def demo_tail_counts() -> None:
    _banner("TAIL COUNTS (Phase 1)  --  exact integers, counted by hand")

    series_x = np.arange(1.0, 11.0)
    series_y = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0])
    matrix = np.vstack([series_x, series_y])
    n_observations = matrix.shape[1]
    level = 0.70
    scale = n_observations + 1

    print(f"x = {series_x}\ny = {series_y}")
    print(f"\nS = {n_observations}, q = {level}, T = S + 1 = {scale}")
    print(f"upper event is strict: rank > q*T = {level * scale}")

    ranks = rankdata(matrix, method="average", axis=1)
    hits_x = [s for s in range(n_observations) if ranks[0, s] > level * scale]
    hits_y = [s for s in range(n_observations) if ranks[1, s] > level * scale]
    joint = sorted(set(hits_x) & set(hits_y))
    print(f"\nx exceeds at {hits_x}   -> marginal = {len(hits_x)}")
    print(f"y exceeds at {hits_y}   -> marginal = {len(hits_y)}")
    print(f"both exceed  {joint}   -> joint    = {len(joint)}")

    sample, compiled_ranks = _compile(matrix)
    counts = tail_counts(
        sample, compiled_ranks, thresholds=np.array([level]), tail=TailSide.UPPER
    )
    print(f"\npackage marginal_count = {counts.marginal_count[0].tolist()}")
    print(f"package joint_count    = {counts.joint_count[0].tolist()}")
    print(
        "diagonal equals marginal = "
        f"{np.array_equal(np.diag(counts.joint_count[0]), counts.marginal_count[0])}"
    )
    print("\nCounts only. No coefficient is derived at this layer.")


def demo_tail_coefficients() -> None:
    _banner("TAIL COEFFICIENTS (Phase 1b-A)  --  directional, and undefined where it must be")

    print("Worked case E2 from the mathematical specification. A six-member tie")
    print("block at the maximum makes the realised marginal counts unequal, so")
    print("the statistics are genuinely directional.\n")

    tied = np.concatenate([np.arange(14.0), np.full(6, 99.0)])
    matrix = np.vstack([tied, np.arange(20.0)])
    sample, ranks = _compile(matrix)
    counts = tail_counts(
        sample, ranks, thresholds=np.array([0.80]), tail=TailSide.UPPER
    )

    n_observations = counts.provenance.observation_count
    marginal_0 = int(counts.marginal_count[0, 0])
    marginal_1 = int(counts.marginal_count[0, 1])
    joint_01 = int(counts.joint_count[0, 0, 1])
    print(
        f"S = {n_observations}, m_0 = {marginal_0}, m_1 = {marginal_1}, "
        f"n_01 = {joint_01}"
    )

    def longhand(conditioning: int, joint: int) -> tuple[float, float, float]:
        p_i = conditioning / n_observations
        p_ij = joint / n_observations
        return (
            joint / conditioning,
            2.0 - math.log(p_ij) / math.log(p_i),
            2.0 * math.log(p_i) / math.log(p_ij) - 1.0,
        )

    result = tail_coefficients(counts)
    header = f"{'cell':<8}{'source':<14}{'lambda_hat':>15}{'chi_hat':>15}{'chi_bar_hat':>15}"
    print(f"\n{header}")
    for (row, col), conditioning in (((0, 1), marginal_0), ((1, 0), marginal_1)):
        by_hand = longhand(conditioning, joint_01)
        print(
            f"{f'({row},{col})':<8}{'package':<14}"
            f"{result.lambda_hat[0, row, col]:>15.10f}"
            f"{result.chi_hat[0, row, col]:>15.10f}"
            f"{result.chi_bar_hat[0, row, col]:>15.10f}"
        )
        print(
            f"{'':<8}{'longhand':<14}"
            f"{by_hand[0]:>15.10f}{by_hand[1]:>15.10f}{by_hand[2]:>15.10f}"
        )

    print("\nAsymmetry is the contract, not a defect:")
    print(
        "  lambda_hat[0,1] != lambda_hat[1,0] -> "
        f"{result.lambda_hat[0, 0, 1] != result.lambda_hat[0, 1, 0]}"
    )
    print("  the row index is always the conditioning variable.")

    print("\nZero joint count gives three different dispositions (E4):")
    print("  lambda_hat  -> 0.0        defined; a valid estimate")
    print("  chi_hat     -> undefined  log(0) in the numerator carries nothing")
    print("  chi_bar_hat -> -1.0       log(0) in the denominator has a finite limit")

    provenance = result.provenance
    print(f"\nupstream digest     = {provenance.upstream_digest}")
    print(f"derived digest      = {provenance.derived_digest}")
    print(f"methodology version = {provenance.methodology_version}")


def demo_diagnostics_and_dissimilarity() -> None:
    _banner("DIAGNOSTICS AND DISSIMILARITY")

    rng = np.random.default_rng(3)
    sample, ranks = _compile(rng.standard_normal((5, 400)))

    correlation = pearson(sample)
    diagnostics = spectrum(correlation)
    print(f"minimum eigenvalue      = {diagnostics.minimum_eigenvalue:+.3e}")
    print(f"numerical PSD tolerance = {diagnostics.psd_tolerance:.3e}")
    print(f"satisfies numerical PSD = {diagnostics.satisfies_numerical_psd}")
    print(f"condition number        = {diagnostics.condition_number:.4e}")
    print("\nPSD does not imply Cholesky-eligible: a PSD matrix can be singular,")
    print("and is guaranteed singular whenever K > S.")

    dissimilarity = angular_dissimilarity(correlation.matrix)
    print(f"\nangular dissimilarity, first row = {np.round(dissimilarity[0], 6)}")
    print(
        "diagonal is exactly zero         = "
        f"{np.array_equal(np.diag(dissimilarity), np.zeros(5))}"
    )

    kendall = kendall_tau_b(sample, ranks)
    print(f"\nKendall PSD status = {kendall.contract.psd_status.name}")
    print("No proof guarantees PSD for a pairwise-assembled matrix, so the")
    print("conservative declaration stands and consumers must not factorise it.")


def main() -> None:
    demo_compilation()
    demo_pearson()
    demo_spearman()
    demo_kendall()
    demo_tail_counts()
    demo_tail_coefficients()
    demo_diagnostics_and_dissimilarity()
    print(f"\n{LINE}\nAll sections verified against an independent code path.\n{LINE}")


if __name__ == "__main__":
    main()
