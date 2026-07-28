# src/riskquant/sensitivity_pnl/conventions.py
"""
Displacement conventions for the sensitivity-space Taylor expansion.

A sensitivity is a derivative with respect to a *specific* displacement of a
risk factor. Applying a shock in the wrong displacement space is the dominant
silent-wrongness failure mode of every sensitivity PnL engine, so the space is
declared explicitly per factor and resolved at compile time.

Displacement `x_k` for factor `k` given base level `L0` and shocked level `L`:

    ABSOLUTE   x = L - L0          (equity spot delta, DV01, vega, vol nodes)
    RELATIVE   x = L / L0 - 1      (sensitivities quoted per unit return)
    LOG        x = ln(L / L0)      (log-space sensitivities)

Quote base
----------
`quote_base` is the number of quoted units in one natural unit of the factor's
displacement. Order-n sensitivities normalise by `quote_base ** n`, so a single
per-factor scalar is sufficient for first order, second order, and cross terms:

    d1_natural = d1_quoted * base
    d2_natural = d2_quoted * base ** 2
    cross_ij_natural = cross_ij_quoted * base_i * base_j

The normalisation is folded into the compiled arrays at compile time; the
kernels never see a quote convention.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class DisplacementConvention(IntEnum):
    """Space in which a factor's sensitivities are differentiated."""

    ABSOLUTE = 1
    RELATIVE = 2
    LOG = 3


DISPLACEMENT_ABSOLUTE_ID: np.int8 = np.int8(DisplacementConvention.ABSOLUTE)
DISPLACEMENT_RELATIVE_ID: np.int8 = np.int8(DisplacementConvention.RELATIVE)
DISPLACEMENT_LOG_ID: np.int8 = np.int8(DisplacementConvention.LOG)

#: Conventions whose displacement divides by the base level. Compile- and
#: run-time validation rejects non-positive base levels for these factors.
RATIO_DISPLACEMENT_IDS: tuple[np.int8, ...] = (
    DISPLACEMENT_RELATIVE_ID,
    DISPLACEMENT_LOG_ID,
)

#: Named `quote_base` values. The compiled book stores a float64 per factor, so
#: houses with bespoke units pass the scalar directly rather than extending this.
QUOTE_BASE_NATURAL: float = 1.0
QUOTE_BASE_PER_PERCENT: float = 1.0e2
QUOTE_BASE_PER_VOL_POINT: float = 1.0e2
QUOTE_BASE_PER_BASIS_POINT: float = 1.0e4
