# src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_data.py
# Refactored canonical credit factor universe
# Key changes:
# 1. CHI factors generated independently via chi-square (correct marginal)
# 2. No volatility scaling applied to CHI factors
# 3. Clear separation between credit returns and chi mixing variables

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.constants import CANONICAL_HISTORY_BUSINESS_DAYS
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_COUNT,
    CANONICAL_CREDIT_FACTOR_NAME_TO_UIX,
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_T_COPULA_CHI_COLUMNS_UIX,
    CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
)
from riskquant.ingest_fixtures.rng import get_canonical_rng

_logger: logging.Logger = logging.getLogger(__name__)
_rng: np.random.Generator = get_canonical_rng()

# ==============================================================================
# INDICES
# ==============================================================================

_GLOBAL: Final[int] = CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["GLOBAL_CREDIT"]
_IDIO_1: Final[int] = CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_01"]
_IDIO_2: Final[int] = CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_02"]
_IDIO_3: Final[int] = CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_03"]

# ==============================================================================
# CORRELATION (CREDIT BLOCK ONLY)
# ==============================================================================

CREDIT_FACTOR_CORRELATION: Final[npt.NDArray[np.float64]] = np.array(
    [
        [1.00, 0.35, 0.30, 0.25],
        [0.35, 1.00, 0.10, 0.05],
        [0.30, 0.10, 1.00, 0.10],
        [0.25, 0.05, 0.10, 1.00],
    ],
    dtype=np.float64,
)

CREDIT_CHOLESKY: Final[npt.NDArray[np.float64]] = np.asarray(
    np.linalg.cholesky(CREDIT_FACTOR_CORRELATION),
    dtype=np.float64,
)

# ==============================================================================
# VOLS (CREDIT ONLY)
# ==============================================================================

CREDIT_FACTOR_VOLS: Final[npt.NDArray[np.float64]] = _rng.uniform(
    low=0.10,
    high=0.50,
    size=4,
).round(2)

# ==============================================================================
# TIME SERIES SETUP
# ==============================================================================

T: Final[int] = CANONICAL_HISTORY_BUSINESS_DAYS
DATES: Final[npt.NDArray[np.datetime64]] = np.sort(
    np.busday_offset(CANONICAL_AS_OF_DATE, -np.arange(T - 1, -1, -1))
)

# ==============================================================================
# GENERATE CREDIT FACTORS (GAUSSIAN)
# ==============================================================================

_independent_credit = _rng.standard_normal((T, 4))
_credit_returns = (_independent_credit @ CREDIT_CHOLESKY.T) * CREDIT_FACTOR_VOLS

# ==============================================================================
# GENERATE CHI FACTORS (TRUE CHI-SQUARE, INDEPENDENT)
# ==============================================================================

_chi_factors = np.column_stack([
    _rng.chisquare(df=nu, size=T) for nu in CANONICAL_T_COPULA_NU_BY_CHI_FACTOR
])

# ==============================================================================
# COMBINE INTO FULL ARRAY (ORDERED BY UIX)
# ==============================================================================

_full_array = np.zeros((T, CANONICAL_CREDIT_FACTOR_COUNT), dtype=np.float64)

# fill credit
for i, uix in enumerate([_GLOBAL, _IDIO_1, _IDIO_2, _IDIO_3]):
    _full_array[:, uix] = _credit_returns[:, i]

# fill chi
for i, uix in enumerate(CANONICAL_T_COPULA_CHI_COLUMNS_UIX):
    _full_array[:, uix] = _chi_factors[:, i]

# ==============================================================================
# DATAFRAMES (EXPLICIT SEMANTICS)
# ==============================================================================

# Each frame holds a leading `date` column (pl.Date) and one Float64 column per
# factor, in the order of the arrays above. Polars has no row index, so the
# dates that pandas held as the index are a column here.
_CREDIT_COLUMNS: Final[list[str]] = ["GLOBAL_CREDIT", "IDIO_01", "IDIO_02", "IDIO_03"]
_CHI_COLUMNS: Final[list[str]] = [
    CANONICAL_CREDIT_FACTOR_NAMES[uix] for uix in CANONICAL_T_COPULA_CHI_COLUMNS_UIX
]


def _dated_frame(values: npt.NDArray[np.float64], columns: list[str]) -> pl.DataFrame:
    data: dict[str, pl.Series] = {
        "date": pl.Series(values=DATES.astype("datetime64[D]"), dtype=pl.Date)
    }
    for j, name in enumerate(columns):
        data[name] = pl.Series(values=values[:, j], dtype=pl.Float64)
    return pl.DataFrame(data)


CREDIT_FACTOR_RETURNS_DF: Final[pl.DataFrame] = _dated_frame(
    _credit_returns, _CREDIT_COLUMNS
)

CHI_MIXING_VARIABLES_DF: Final[pl.DataFrame] = _dated_frame(_chi_factors, _CHI_COLUMNS)

FULL_FACTOR_DF: Final[pl.DataFrame] = _dated_frame(
    _full_array, list(CANONICAL_CREDIT_FACTOR_NAMES)
)


def _validate_chi_independence() -> None:
    corr = np.corrcoef(_full_array, rowvar=False)

    chi_set = set(CANONICAL_T_COPULA_CHI_COLUMNS_UIX.tolist())
    # credit_set = set(range(CANONICAL_CREDIT_FACTOR_COUNT)) - chi_set

    for chi in chi_set:
        for other in range(CANONICAL_CREDIT_FACTOR_COUNT):
            if other == chi:
                continue
            val = corr[chi, other]
            if abs(val) > 0.15:
                raise AssertionError(
                    f"CHI factor correlation leakage: {chi} vs {other} = {val:.4f}"
                )

    _logger.info("CHI independence validation passed.")


def _validate_psd() -> None:
    eig = np.linalg.eigvalsh(CREDIT_FACTOR_CORRELATION)
    min_eig = float(eig.min())

    if min_eig < -1e-12:
        raise AssertionError("Credit correlation not PSD")

    cond = float(eig.max() / min_eig) if min_eig > 1e-12 else float("inf")

    _logger.info(f"PSD OK (min eig={min_eig:.6f}, cond={cond:.1f})")


_validate_psd()
_validate_chi_independence()

# ==============================================================================
# EXPORTS
# ==============================================================================

CANONICAL_CREDIT_FACTOR_TIME_SERIES_DF = FULL_FACTOR_DF

# ==============================================================================
# DEBUGGING / DIAGNOSTICS
# ==============================================================================


def _debug_print() -> None:
    print("=============================")
    print("=== CREDIT FACTOR RETURNS ===")
    print("=============================")
    print(CREDIT_FACTOR_RETURNS_DF.tail(20))

    print("")
    print("=================================")
    print("=== CHI MIXING VARIABLES (χ²) ===")
    print("=================================")
    print(CHI_MIXING_VARIABLES_DF.tail(20))

    print("")
    print("======================")
    print("=== FULL FACTOR DF ===")
    print("======================")
    print(FULL_FACTOR_DF.tail(20))

    print("")
    print("======================================")
    print("=== CREDIT CORRELATION (EMPIRICAL) ===")
    print("======================================")
    corr = np.asarray(
        np.corrcoef(_credit_returns, rowvar=False), dtype=np.float64
    ).round(4)
    print(
        pl.DataFrame(
            {"factor": _CREDIT_COLUMNS}
            | {name: corr[:, j] for j, name in enumerate(_CREDIT_COLUMNS)}
        )
    )


def _debug_plot() -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))

    # --- CREDIT FACTORS (returns) ---
    ax1 = fig.add_subplot(2, 1, 1)
    for j, col in enumerate(_CREDIT_COLUMNS):
        ax1.plot(DATES, _credit_returns[:, j], label=col)
    ax1.set_title("Credit Factor Returns (Gaussian)")
    ax1.legend()
    ax1.axhline(0, linestyle="--", alpha=0.3)

    # --- CHI FACTORS (levels, NOT returns) ---
    ax2 = fig.add_subplot(2, 1, 2)
    for j, col in enumerate(_CHI_COLUMNS):
        ax2.plot(DATES, _chi_factors[:, j], label=col)
    ax2.set_title("Chi-square Mixing Variables (χ²)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def _debug_chi_diagnostics() -> None:
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    _fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    for i, col in enumerate(_CHI_COLUMNS):
        data = np.asarray(_chi_factors[:, i], dtype=np.float64)
        nu = CANONICAL_T_COPULA_NU_BY_CHI_FACTOR[i]

        # --- Top row: histogram + PDF ---
        ax_hist = axes[0, i]
        x = np.linspace(0, np.percentile(data, 99), 200)

        ax_hist.hist(data, bins=30, density=True, alpha=0.5, label="empirical")
        ax_hist.plot(x, stats.chi2.pdf(x, df=nu), label=f"χ² pdf (nu={nu})")

        ax_hist.set_title(f"{col} Distribution")
        ax_hist.legend()

        # --- Bottom row: QQ plot ---
        ax_qq = axes[1, i]
        stats.probplot(data, dist="chi2", sparams=(nu,), plot=ax_qq)
        ax_qq.set_title(f"{col} QQ vs χ²")

    plt.tight_layout()
    plt.show()


def _debug_full_correlation() -> None:
    import matplotlib.pyplot as plt

    corr = np.asarray(np.corrcoef(_full_array, rowvar=False), dtype=np.float64)
    names = list(CANONICAL_CREDIT_FACTOR_NAMES)

    _fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr[i, j]) > 0.7 else "black",
                fontsize=7,
            )

    plt.colorbar(im, ax=ax)
    ax.set_title("Full Factor Correlation Matrix")

    plt.tight_layout()
    plt.show()


def _debug_chi_moments() -> None:
    print("")
    print("========================")
    print("=== CHI MOMENT CHECK ===")
    print("========================")

    for i, col in enumerate(_CHI_COLUMNS):
        data = _chi_factors[:, i]
        nu = CANONICAL_T_COPULA_NU_BY_CHI_FACTOR[i]

        mean = data.mean()
        var = data.var()

        print(f"{col}: mean={mean:.3f} (target {nu}), var={var:.3f} (target {2 * nu})")


def _debug_chi_tail() -> None:
    print("")
    print("=======================")
    print("=== CHI TAIL RATIOS ===")
    print("=======================")

    for i, col in enumerate(_CHI_COLUMNS):
        data = _chi_factors[:, i]

        q95 = np.percentile(data, 95)
        q99 = np.percentile(data, 99)

        print(f"{col}: q99/q95 = {q99 / q95:.3f}")


def _debug_t_factor() -> None:
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    print("")
    print("=========================")
    print("=== T-FACTOR QQ CHECK ===")
    print("=========================")

    print("\nSee plot.")

    _fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    for i, (ax, col) in enumerate(zip(axes, _CREDIT_COLUMNS, strict=False)):
        z = _credit_returns[:, i]
        w = _chi_factors[:, i]
        nu = CANONICAL_T_COPULA_NU_BY_CHI_FACTOR[i]

        x = z / np.sqrt(w / nu)

        stats.probplot(x, dist="t", sparams=(nu,), plot=ax)
        ax.set_title(f"{col} → t(nu={nu})")

    plt.tight_layout()
    plt.show()


def _debug_rank_correlation() -> None:
    import scipy.stats as stats

    print("")
    print("===================================")
    print("=== RANK CORRELATION (SPEARMAN) ===")
    print("===================================")

    # numpy/scipy in place of DataFrame.corr(method="spearman") (WS backlog).
    names = list(CANONICAL_CREDIT_FACTOR_NAMES)
    rank_corr = np.round(
        100 * np.asarray(stats.spearmanr(_full_array).statistic, dtype=np.float64),
        3,
    )
    print(
        pl.DataFrame(
            {"factor": names} | {name: rank_corr[:, j] for j, name in enumerate(names)}
        )
    )


def main() -> None:
    _debug_print()
    _debug_plot()
    _debug_chi_diagnostics()
    _debug_full_correlation()
    _debug_chi_moments()
    _debug_chi_tail()
    _debug_t_factor()
    _debug_rank_correlation()


if __name__ == "__main__":
    main()
