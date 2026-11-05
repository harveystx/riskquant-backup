# src/riskquant/ingest_fixtures/credit/jtd/mixed/minimal_mixed.py
"""
Canonical mixed JTD ingest case.

Combines every per-model canonical JTD fixture into ONE mixed portfolio by
re-basing each model's obligor uids onto a disjoint contiguous range (an obligor
carries exactly one JTD model, so the cross-frame disjointness invariant must
hold) and concatenating them into a single denormalized frame. The shared
multi-factor loadings relation is re-based to match the multi-factor block's new
uids.

This is assembled programmatically from the individual fixtures — there is no
duplicated per-model data here. `build_canonical_mixed_jtd_bundle()` normalizes
the combined frame into a `JtdIngestBundle` ready for `compile_jtd_portfolio`.

Note: the small canonical universe has only three uniform / idiosyncratic
factors, so the Bernoulli-family and copula blocks reuse U_0x / IDIO_0x across
models. That is fine for the compile pipeline (within-model uniforms stay
distinct); it is a minimal compile-path fixture, not a statistical sample.
"""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.ingest_fixtures.credit.jtd.beta_binomial.minimal_beta_binomial_jtd import (
    CANONICAL_BETA_BINOMIAL_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.forced_default.minimal_forced_default_jtd import (
    CANONICAL_FORCED_DEFAULT_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.independent_bernoulli.minimal_independent_bernoulli_jtd import (
    CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.mf_copula.minimal_mf_copula_gaussian import (
    CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF,
    CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF,
)
from riskquant.ingest_fixtures.credit.jtd.sf_copula_jtd.minimal_sf_copula_gaussian import (
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.sf_copula_jtd.minimal_sf_copula_students_t_global_chi import (
    CANONICAL_T_COPULA_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.stress_regime_bernoulli.minimal_stress_regime_bernoulli_jtd import (
    CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DF,
)

# Per-model sources, in CompiledJtdPortfolio band order. The multi-factor frame
# is last; its loadings relation is re-based with the same offset.
_SOURCES: tuple[pl.DataFrame, ...] = (
    CANONICAL_FORCED_DEFAULT_JTD_INGEST_DF,
    CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DF,
    CANONICAL_BETA_BINOMIAL_JTD_INGEST_DF,
    CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DF,
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
    CANONICAL_T_COPULA_JTD_INGEST_DF,
    CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF,
)


def _combine() -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Re-base each fixture's uids onto a disjoint range and concatenate.

    Each fixture numbers its obligors 1..n, so adding the running offset (the
    count of all obligors placed so far) shifts it into a fresh, collision-free
    band while preserving the uid values the loadings relation references. The
    union of columns is a diagonal concat: a column that a fixture lacks is
    null on its rows (the pandas concat gave NaN).
    """
    rebased: list[pl.DataFrame] = []
    offset = 0
    mf_offset = 0
    for df in _SOURCES:
        frame = df.with_columns(pl.col("uid").cast(pl.Int64) + offset)
        if df is CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF:
            mf_offset = offset
        rebased.append(frame)
        offset += df.height

    mixed = pl.concat(rebased, how="diagonal_relaxed")

    loadings = CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF.with_columns(
        pl.col("obligor_uid").cast(pl.Int64) + mf_offset
    )
    return mixed, loadings, offset


CANONICAL_MIXED_JTD_INGEST_DF, CANONICAL_MIXED_JTD_LOADINGS_DF, _N_OBLIGORS = _combine()

# The full obligor-uid range the mixed portfolio spans (1 .. N). A consumer
# builds an obligor master covering exactly these uids.
CANONICAL_MIXED_JTD_OBLIGOR_UIDS: list[int] = list(range(1, _N_OBLIGORS + 1))


def build_canonical_mixed_jtd_bundle() -> JtdIngestBundle:
    """Normalize the combined frame into a JtdIngestBundle (the mixed master)."""
    return JtdIngestBundle.from_instrument_frame(
        CANONICAL_MIXED_JTD_INGEST_DF,
        mf_loadings_df=CANONICAL_MIXED_JTD_LOADINGS_DF,
    )


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_success_message,
    )

    print_calling_file()
    print(f"Mixed JTD portfolio: {_N_OBLIGORS} obligors across all JTD models.")
    print_df(
        CANONICAL_MIXED_JTD_INGEST_DF,
        title="Canonical Mixed JTD Ingest Frame (denormalized, head/tail)",
    )
    print_df(
        CANONICAL_MIXED_JTD_LOADINGS_DF,
        title="Canonical Mixed JTD multi-factor loadings (re-based)",
        float_format="{:>_.2f}",
    )
    bundle = build_canonical_mixed_jtd_bundle()
    print("Bundle slots populated:")
    for model, frame in bundle.frames_by_model().items():
        print(f"  - {model.name:<28} {len(frame):>2d} obligor(s)")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
