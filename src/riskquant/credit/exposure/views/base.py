# src/riskquant/credit/exposure/views/base.py
"""
Base class for compiled netting-set views.

`_NettingSetView` carries only what is universal to every EAD model: the CSR
band layout that groups portfolio-order MtM rows into contiguous per-netting-set
spans. All model-specific SoA arrays (collateral, notionals, supervisory
factors, ...) live on the concrete subclasses.

`EXPOSURE_MODEL_ID` is the integer the exposure engine dispatches on — it
indexes `EAD_EVALUATORS` directly. The `_UNSET` sentinel is rejected at
class-creation time, a compile-time guard against adding a model view without
wiring its id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

_UNSET_EXPOSURE_MODEL_ID: int = -999


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _NettingSetView:
    """Universal CSR base for compiled netting-set views."""

    EXPOSURE_MODEL_ID: ClassVar[int] = _UNSET_EXPOSURE_MODEL_ID

    # Effective-netting-set identifiers, one per band. dtype: int64
    netting_set_uids: npt.NDArray[np.int64]  # (N,)

    # Exclusive prefix-sum band boundaries into the reordered MtM matrix. int64
    netting_set_offsets: npt.NDArray[np.int64]  # (N + 1,)

    # Permutation: global portfolio MtM rows -> NS-contiguous order. int64
    mtm_row_perm: npt.NDArray[np.int64]  # (M_view,)

    def __init_subclass__(cls, **kwargs: object) -> None:
        is_abstract_base = cls.__name__.startswith("_")
        if not is_abstract_base and (cls.EXPOSURE_MODEL_ID == _UNSET_EXPOSURE_MODEL_ID):
            raise TypeError(
                f"Netting-set view '{cls.__name__}' must set EXPOSURE_MODEL_ID "
                f"(its ExposureModel id) for engine dispatch."
            )

    @property
    def num_netting_sets(self) -> int:
        return self.netting_set_uids.shape[0]

    @property
    def num_trades(self) -> int:
        return self.mtm_row_perm.shape[0]
