# riskquant/portfolio/compiled.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterator

import numpy as np

from riskquant.portfolio.blocks.forward_block import ForwardNodeBlock
from riskquant.portfolio.blocks.spot_block import SpotBlock


@dataclass(frozen=True, slots=True)
class CompiledPortfolio:
    """
    General compiled portfolio container (instrument blocks).
    """

    M: int

    # Factor universes required to align to RiskFactorMatrix columns later
    price_factor_ids: np.ndarray  # (N_price,) int32
    fx_factor_ids: np.ndarray  # (N_fx,) int32

    # Cache of unique and sorted factor IDs (disjoint union of price/fx)
    _factor_ids_sorted_unique: Optional[np.ndarray] = field(
        init=False, default=None
    )

    spot: Optional[SpotBlock] = None
    forward: Optional[ForwardNodeBlock] = None

    def __post_init__(self) -> None:
        # --- FACTOR UNIVERSES (type/shape/layout) --- #
        p = self.price_factor_ids
        f = self.fx_factor_ids

        if p.dtype != np.int32:
            raise TypeError(f"price_factor_ids must be int32; got {p.dtype}")
        if f.dtype != np.int32:
            raise TypeError(f"fx_factor_ids must be int32; got {f.dtype}")
        if p.ndim != 1:
            raise ValueError("price_factor_ids must be 1D")
        if f.ndim != 1:
            raise ValueError("fx_factor_ids must be 1D")
        if not p.flags["C_CONTIGUOUS"]:
            raise ValueError("price_factor_ids must be contiguous")
        if not f.flags["C_CONTIGUOUS"]:
            raise ValueError("fx_factor_ids must be contiguous")

        # --- M CONSISTENCY --- #
        m = 0
        if self.spot is not None:
            m += int(self.spot.trade_uids.shape[0])
        if self.forward is not None:
            m += int(self.forward.trade_uids.shape[0])
        if m != int(self.M):
            raise ValueError(f"M mismatch: header M={self.M}, blocks total={m}")

        # Optional sanity check given your note (2)
        # If you truly require M >= N_price + N_fx:
        if int(self.M) < int(p.size + f.size):
            raise ValueError(
                f"M must be >= N_price + N_fx; got M={self.M}, "
                f"N_price={p.size}, N_fx={f.size}"
            )

        # --- UNIQUENESS + DISJOINTNESS --- #
        # Sorting copies are required unless arrays are already sorted.
        p_sorted = np.sort(p)  # int32, contiguous
        if p_sorted.size > 1 and np.any(p_sorted[1:] == p_sorted[:-1]):
            raise ValueError("price_factor_ids contains duplicates")

        f_sorted = np.sort(f)
        if f_sorted.size > 1 and np.any(f_sorted[1:] == f_sorted[:-1]):
            raise ValueError("fx_factor_ids contains duplicates")

        # Check intersection efficiently: search the smaller into the larger.
        if p_sorted.size and f_sorted.size:
            if p_sorted.size <= f_sorted.size:
                small, large = p_sorted, f_sorted
            else:
                small, large = f_sorted, p_sorted

            idx = np.searchsorted(large, small)
            # clip idx to valid range and compare
            in_range = idx < large.size
            if np.any(in_range & (large[idx] == small)):
                raise ValueError("price_factor_ids and fx_factor_ids intersect")

        # --- BUILD CACHED SORTED UNIVERSE --- #
        # Since they are disjoint and internally unique, the union is just
        # a merge; simplest is concatenate then sort (still O(N log N)).
        # If you want O(N) merge, see note below.
        u = np.sort(np.concatenate((p, f)))
        object.__setattr__(self, "_factor_ids_sorted_unique", u)

    @property
    def num_spot(self) -> int:
        return 0 if self.spot is None else int(self.spot.trade_uids.shape[0])

    @property
    def num_forward(self) -> int:
        return (
            0 if self.forward is None else int(self.forward.trade_uids.shape[0])
        )

    @property
    def factor_ids_sorted_unique(self) -> np.ndarray:
        return self._factor_ids_sorted_unique  # type: ignore[return-value]

    def has_factor_id(self, factor_id: int) -> bool:
        a = self._factor_ids_sorted_unique  # local binding
        i = np.searchsorted(a, np.int32(factor_id))
        return bool(i < a.size and a[i] == factor_id)

    def iter_factor_ids(self) -> Iterator[np.int32]:
        yield from self.price_factor_ids
        yield from self.fx_factor_ids


# # riskquant/portfolio/compiled.py
# from __future__ import annotations
#
# from dataclasses import dataclass
# from typing import Optional
#
# import numpy as np
#
# from riskquant.portfolio.blocks.forward_block import ForwardBlock
# from riskquant.portfolio.blocks.spot_block import SpotBlock
#
#
# @dataclass(frozen=True, slots=True)
# class CompiledPortfolio:
#     """
#     General compiled portfolio container (instrument blocks).
#     """
#
#     # Number of trades in portfolio
#     M: int
#
#     # Factor universes required to align to RiskFactorMatrix columns later
#     price_factor_ids: np.ndarray  # (N_price,) int32
#     fx_factor_ids: np.ndarray  # (N_fx,) int32
#
#     # Blocks (TODO: extend with more instruments later)
#     spot: Optional[SpotBlock] = None
#     forward: Optional[ForwardBlock] = None
