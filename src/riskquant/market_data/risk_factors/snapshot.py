# riskquant/market_data/risk_factors/snapshot.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from riskquant.utils.dates import ordinal32_to_datetime64D


@dataclass(frozen=True, slots=True)
class RiskFactorLevelSnapshot:
    """Immutable snapshot of risk factor levels at a single as-of date.

    Attributes
    ----------
    as_of_date : datetime.date
        The calendar date for which the snapshot applies.
    factor_ids : np.ndarray[int32]
        1-D array of unique risk-factor identifiers (dtype int32, C-contiguous).
        Must be sorted ascending; if not sorted, the constructor will re-order
        the ids and corresponding `factor_levels`.
    factor_levels : np.ndarray[float32|float64]
        1-D numeric array of factor levels aligned with `factor_ids`.
        Must be the same length as `factor_ids` and C-contiguous.

    Invariants (enforced on construction)
    -------------------------------------
    - `factor_ids` is 1D, dtype int32, C-contiguous, sorted ascending and unique
    - `factor_levels` is 1D, dtype float32 or float64, C-contiguous
    - lengths of `factor_ids` and `factor_levels` match

    The class is intentionally lightweight and validates eagerly so downstream
    numerical kernels can assume compact, contiguous, and deterministic
    memory layouts (fast gathers, stable hashing, etc.).
    """

    as_of_date: date
    factor_ids: np.ndarray  # (N0,) int32
    factor_levels: np.ndarray  # (N0,) float32/float64

    def __post_init__(self) -> None:
        """Validate and canonicalize snapshot arrays.

        The method performs cheap, early checks and performs the minimal amount
        of work required to restore canonical invariants (sorting + contiguity).
        It raises clear exceptions on invalid input rather than attempting to
        silently coerce data into a valid form.
        """
        # Local aliases to make subsequent checks concise and explicit.
        ids = self.factor_ids
        lv = self.factor_levels

        # ----- shape checks -----
        # Ensure both arrays are 1-D vectors.
        if ids.ndim != 1:
            raise ValueError(f"factor_ids must be 1D; got shape={ids.shape}")
        if lv.ndim != 1:
            raise ValueError(f"factor_levels must be 1D; got shape={lv.shape}")

        # Ensure arrays have the same length (alignment requirement).
        if ids.shape[0] != lv.shape[0]:
            raise ValueError(
                f"factor_ids and factor_levels length mismatch: "
                f"{ids.shape[0]} vs {lv.shape[0]}"
            )

        # ----- dtype checks -----
        # factor_ids must be 32-bit integers for compactness and predictable
        # behavior when used as indices or keys in mappings.
        if ids.dtype != np.int32:
            raise TypeError(f"factor_ids must be int32; got {ids.dtype}")

        # factor_levels must be floating point (single or double precision).
        if lv.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"factor_levels must be float32 or float64; got {lv.dtype}"
            )

        # ----- memory layout checks -----
        # Many numerical routines assume C-contiguous 1-D arrays; check early so
        # callers get an explicit error instead of subtle performance bugs.
        if not ids.flags["C_CONTIGUOUS"]:
            raise ValueError("factor_ids must be C-contiguous")
        if not lv.flags["C_CONTIGUOUS"]:
            raise ValueError("factor_levels must be C-contiguous")

        # Fast path: zero or one element requires no further sorting/uniqueness checks.
        n = ids.size
        if n <= 1:
            return

        # ----- sorted + unique enforcement -----
        # If ids are already non-decreasing, perform an adjacent-duplicate check
        # which is allocation-free. If not sorted, build a stable permutation and
        # reorder both arrays so the object is canonical afterwards.
        if not np.any(ids[1:] < ids[:-1]):
            # ids are non-decreasing — ensure strict uniqueness.
            if np.any(ids[1:] == ids[:-1]):
                raise ValueError("factor_ids must be unique")
            return

        # ids are unsorted: compute a stable sort permutation (mergesort).
        perm = np.argsort(ids, kind="mergesort")

        # Apply the permutation to ids (produces a new array view/copy).
        ids_sorted = ids[perm]

        # Check for duplicates after sorting (adjacent duplicates are easy to detect).
        if np.any(ids_sorted[1:] == ids_sorted[:-1]):
            raise ValueError("factor_ids must be unique")

        # Reorder factor levels to keep alignment with the now-sorted ids.
        lv_sorted = lv[perm]

        # Fancy indexing typically returns a contiguous array for 1-D, but
        # explicitly enforce C-contiguity to preserve the class invariant.
        if not lv_sorted.flags["C_CONTIGUOUS"]:
            lv_sorted = np.ascontiguousarray(lv_sorted)

        # Bypass dataclass immutability to install the canonicalized arrays.
        object.__setattr__(self, "factor_ids", ids_sorted)
        object.__setattr__(self, "factor_levels", lv_sorted)

    def to_df(self) -> pd.DataFrame:
        """Return the snapshot as a 1-row wide DataFrame."""
        # Reshape the 1D levels array into a 2D row vector (1, N)
        data_2d = self.factor_levels[np.newaxis, :]

        df = pd.DataFrame(
            data=data_2d,
            index=pd.DatetimeIndex([self.as_of_date], name="date"),
            columns=pd.Index(self.factor_ids, name="factor_id"),
        )
        return df

    def to_series(self) -> pd.Series:
        """Return the snapshot as a pandas Series."""
        return pd.Series(
            data=self.factor_levels,
            index=pd.Index(self.factor_ids, name="factor_id"),
            name=self.as_of_date,
        )

    @property
    def df(self) -> pd.DataFrame:
        """Alias for to_df()."""
        return self.to_df()

    @property
    def sr(self) -> pd.Series:
        """Alias for to_series()."""
        return self.to_series()

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)
