# src/riskquant/market_data/calibrate/option_quotes/master.py
"""
This module defines the contract for calibrating implied volatilities from raw option
market data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.constants import MIN_RELIABLE_VEGA_OVER_FWD


@dataclass(slots=True, frozen=True, kw_only=True)
class ValidatedOptionQuotes:
    fwd_price: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]
    tte: npt.NDArray[np.float64]
    call_put_flag: npt.NDArray[np.int8]
    discount_factor: npt.NDArray[np.float64]
    option_price: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.fwd_price.shape[0]

        if self.strike.shape[0] != n:
            raise ValueError("strike length mismatch")

        if self.tte.shape[0] != n:
            raise ValueError("tte length mismatch")

        if self.call_put_flag.shape[0] != n:
            raise ValueError("call_put_flag length mismatch")

        if self.discount_factor.shape[0] != n:
            raise ValueError("discount_factor length mismatch")

        if self.option_price.shape[0] != n:
            raise ValueError("option_price length mismatch")

    @property
    def n_quotes(self) -> int:
        return self.fwd_price.shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_quotes={self.n_quotes:,})"

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, max_rows: int = 10) -> str:
        """
        Return a human-readable preview of the validated option quotes.

        Only the first ``max_rows`` rows are displayed.
        """

        n_quotes = self.n_quotes
        n_display = min(n_quotes, max_rows)

        lines: list[str] = [
            (f"{self.__class__.__name__}(n_quotes={n_quotes:,})"),
            "",
            (
                f"{'Row':>8} | {'Fwd':>14} | {'Strike':>14} | {'TTE':>12} | "
                f"{'C/P':>4} | {'DF':>10} | {'Price':>14}"
            ),
            "-" * 94,
        ]

        for i in range(n_display):
            cp = "C" if self.call_put_flag[i] == 1 else "P"
            lines.append(
                f"{i:8d} | "
                f"{self.fwd_price[i]:14.6f} | "
                f"{self.strike[i]:14.6f} | "
                f"{self.tte[i]:12.6f} | "
                f"{cp:>4} | "
                f"{self.discount_factor[i]:10.6f} | "
                f"{self.option_price[i]:14.6f}"
            )

        if n_quotes > n_display:
            lines.append("-" * 94)
            lines.append(f"... {n_quotes - n_display:,} additional rows not shown")

        return "\n".join(lines)


@dataclass(slots=True, frozen=True, kw_only=True)
class CanonicalOptionQuotes:
    fwd_price: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]
    tte: npt.NDArray[np.float64]
    undiscounted_call_price: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.fwd_price.shape[0]

        if self.strike.shape[0] != n:
            raise ValueError("strike length mismatch")

        if self.tte.shape[0] != n:
            raise ValueError("tte length mismatch")

        if self.undiscounted_call_price.shape[0] != n:
            raise ValueError("undiscounted_call_price length mismatch")

    @property
    def n_quotes(self) -> int:
        return self.fwd_price.shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_quotes={self.fwd_price.shape[0]:,})"

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, max_rows: int = 10) -> str:
        """
        Return a human-readable preview of the canonical option quotes.

        Only the first ``max_rows`` rows are displayed.
        """

        n_quotes = self.fwd_price.shape[0]
        n_display = min(n_quotes, max_rows)

        lines: list[str] = [
            (f"{self.__class__.__name__}(n_quotes={n_quotes:,})"),
            "",
            (
                f"{'Row':>8} | {'Fwd':>14} | {'Strike':>14} | {'TTE':>12} | {'Undisc Call':>14}"
            ),
            "-" * 74,
        ]

        for i in range(n_display):
            lines.append(
                f"{i:8d} | "
                f"{self.fwd_price[i]:14.6f} | "
                f"{self.strike[i]:14.6f} | "
                f"{self.tte[i]:12.6f} | "
                f"{self.undiscounted_call_price[i]:14.6f}"
            )

        if n_quotes > n_display:
            lines.append("-" * 74)
            lines.append(f"... {n_quotes - n_display:,} additional rows not shown")

        return "\n".join(lines)


@dataclass(slots=True, frozen=True, kw_only=True)
class CalibratedOptionQuotes:
    """
    Calibrated Black-76 implied volatilities, index-aligned to the canonical
    quotes they were solved from.

    Contract
    --------
    The master is *full-length* and never compacted: ``fwd_price``, ``strike``,
    ``tte``, ``iv``, ``vega`` and ``success_mask`` share one index, preserving
    the join back to the originating quotes (and, downstream, to instrument
    identity).

    Failure channel (decision (a)):
        - ``success_mask[i]`` is the contract surface. ``True`` => solved.
        - ``iv[i] == IV_SOLVER_FAILED_SENTINEL`` (negative) marks a failed
          inversion.
        - ``iv[i] == 0.0`` is a *valid* at-intrinsic zero-vol solve, not a
          failure.
        - The invariant ``success_mask == (iv >= 0.0)`` is enforced at
          construction.

    Quality channel:
        - ``vega[i]`` is dC/dvol at the solved iv (undiscounted Black-76).
        - It is the natural quality weight: 0.0 for failed solves and for
          at-intrinsic zero-vol solves, tiny in flat-vega regions where the iv
          is under-determined, large where the inversion is well-conditioned.
        - A solved iv with near-zero vega reprices its target but does not pin
          down the vol; weight residuals by ``vega`` (or gate via
          ``reliable_mask``) when fitting a surface.

    The mask is intentionally denormalized against the iv sign so consumers gate
    with a branchless ``iv[success_mask]`` instead of re-deriving the convention.
    """

    fwd_price: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]
    tte: npt.NDArray[np.float64]
    iv: npt.NDArray[np.float64]
    vega: npt.NDArray[np.float64]
    success_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        n = self.fwd_price.shape[0]

        if self.strike.shape[0] != n:
            raise ValueError("strike length mismatch")

        if self.tte.shape[0] != n:
            raise ValueError("tte length mismatch")

        if self.iv.shape[0] != n:
            raise ValueError("iv length mismatch")

        if self.vega.shape[0] != n:
            raise ValueError("vega length mismatch")

        if self.success_mask.shape[0] != n:
            raise ValueError("success_mask length mismatch")

        # NaN must never enter the output: that is the whole point of rejecting
        # the in-band-NaN channel in favour of an explicit sentinel + mask.
        if np.isnan(self.iv).any():
            raise ValueError(
                "iv contains NaN; failed solves must use the "
                "IV_SOLVER_FAILED_SENTINEL value, not NaN"
            )

        # success_mask is the named contract; enforce it against the iv sign so a
        # mask that disagrees with the sentinel can never be silently constructed.
        if not np.array_equal(self.success_mask, self.iv >= 0.0):
            raise ValueError(
                "success_mask inconsistent with iv sign contract "
                "(failed solves must be the negative sentinel)"
            )

        # Vega is a sensitivity magnitude: non-negative, finite, and exactly 0
        # wherever the solve failed. Note the converse does NOT hold: a *solved*
        # at-intrinsic (iv == 0.0) quote also carries zero vega, which is correct.
        if np.isnan(self.vega).any():
            raise ValueError("vega contains NaN")

        if np.any(self.vega < 0.0):
            raise ValueError("vega must be non-negative")

        if not np.all(self.vega[~self.success_mask] == 0.0):
            raise ValueError("failed solves must carry zero vega")

    @property
    def n_quotes(self) -> int:
        return self.fwd_price.shape[0]

    @property
    def n_solved(self) -> int:
        return int(np.count_nonzero(self.success_mask))

    @property
    def n_failed(self) -> int:
        return self.n_quotes - self.n_solved

    def reliable_mask(
        self, min_vega_over_fwd: float = MIN_RELIABLE_VEGA_OVER_FWD
    ) -> npt.NDArray[np.bool_]:
        """
        Solved quotes whose vega is large enough for the iv to be well-determined.

        ``reliable = success_mask & (vega >= min_vega_over_fwd * fwd_price)``.

        The vega floor is scaled by forward so the gate is dimensionless across
        notionals. The default is a heuristic; pass an explicit floor to tune.
        Returns a fresh boolean array.
        """
        return self.success_mask & (self.vega >= min_vega_over_fwd * self.fwd_price)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_quotes={self.n_quotes:,}, "
            f"n_solved={self.n_solved:,}, "
            f"n_failed={self.n_failed:,})"
        )

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, max_rows: int = 10) -> str:
        """
        Return a human-readable preview of the calibrated quotes.

        Only the first ``max_rows`` rows are displayed.
        """

        n_quotes = self.n_quotes
        n_display = min(n_quotes, max_rows)

        lines: list[str] = [
            (
                f"{self.__class__.__name__}(n_quotes={n_quotes:,}, "
                f"n_solved={self.n_solved:,}, n_failed={self.n_failed:,})"
            ),
            "",
            (
                f"{'Row':>8} | {'Fwd':>14} | {'Strike':>14} | "
                f"{'TTE':>12} | {'IV':>14} | {'Vega':>14} | {'Status':>8}"
            ),
            "-" * 101,
        ]

        for i in range(n_display):
            status = "ok" if self.success_mask[i] else "FAILED"
            lines.append(
                f"{i:8d} | "
                f"{self.fwd_price[i]:14.6f} | "
                f"{self.strike[i]:14.6f} | "
                f"{self.tte[i]:12.6f} | "
                f"{self.iv[i]:14.6f} | "
                f"{self.vega[i]:14.6f} | "
                f"{status:>8}"
            )

        if n_quotes > n_display:
            lines.append("-" * 101)
            lines.append(f"... {n_quotes - n_display:,} additional rows not shown")

        return "\n".join(lines)
