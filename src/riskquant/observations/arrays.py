# src/riskquant/observations/arrays.py

"""
Canonical numerical array aliases.

Owned by ``observations`` because the layout contract is a property of the
empirical sample, not of any estimator. ``dependence`` and ``market_state`` both
import from here rather than defining their own aliases.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: (K, S) or (K, K), float64, C-contiguous.
Float64Matrix = npt.NDArray[np.float64]

#: (K,) or (S,), float64, C-contiguous.
Float64Vector = npt.NDArray[np.float64]

#: (K, S) or (Q, K, K), int32, C-contiguous.
Int32Matrix = npt.NDArray[np.int32]

#: (P,), int32, C-contiguous.
Int32Vector = npt.NDArray[np.int32]

#: (K, S), boolean, C-contiguous. Used only as reusable kernel scratch; no
#: public result carries a boolean payload.
BoolMatrix = npt.NDArray[np.bool_]
