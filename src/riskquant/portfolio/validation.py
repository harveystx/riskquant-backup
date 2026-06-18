# src/riskquant/portfolio/validation.py
"""
Pre-compile validation of `IngestBatch` global-ID references against the
`UixMapper` universes.

Runs ONCE between universe discovery (`build_uix_mapper`) and block
compilation. Catches any global ID or ordinal that the upstream resolver
emitted but the universe-builder didn't see — i.e. a referential-integrity
bug somewhere in the ingest pipeline. By doing this before compile, the
hot-path `map_*_to_uix` calls can stay branch-light and trust their input.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_membership
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.universe import UixMapper


def validate_ingest_batches_against_uix_universe(
    *,
    ingest_batches: Sequence[IngestBatch],
    uix_mapper: UixMapper,
) -> None:
    """
    Validate that every non-sentinel ID and ordinal referenced by the
    ingest batches is present in the corresponding `UixMapper` universe.

    Raises `ValueError` with the first batch of offending values it finds,
    grouped by universe.
    """
    # --- MARKET IDs -------------------------------------------------------
    _check_arrays_against_universe(
        arrays=(
            arr
            for b in ingest_batches
            for arr in (
                b.underlying_ids,
                b.volatility_ids,
                b.fx_u2s_ids,
                b.fx_k2s_ids,
            )
        ),
        universe=uix_mapper.market_id_universe,
        label="market IDs",
    )

    # --- DISCOUNT CURVE IDs -----------------------------------------------
    _check_arrays_against_universe(
        arrays=(
            arr
            for b in ingest_batches
            for arr in (
                b.under_discount_curve_ids,
                b.strike_discount_curve_ids,
            )
        ),
        universe=uix_mapper.discount_curve_id_universe,
        label="discount-curve IDs",
    )

    # --- DATES ------------------------------------------------------------
    _check_arrays_against_universe(
        arrays=(
            arr
            for b in ingest_batches
            for arr in (b.settle_ord, b.delivery_ord, b.expiry_ord)
        ),
        universe=uix_mapper.date_ord32_universe,
        label="ordinal dates",
    )


def _check_arrays_against_universe(
    *,
    arrays: Iterable[npt.NDArray[np.int32]],
    universe: npt.NDArray[np.int32],
    label: str,
) -> None:
    """
    For every input array, assert that every value >= 0 also appears in
    `universe`. Vectorized: one `searchsorted` per input array, no Python
    loop over rows.
    """
    for arr in arrays:
        populated = arr[arr >= 0]
        if populated.size == 0:
            continue

        matches = searchsorted_membership(universe=universe, raw=populated)
        if not matches.all():
            missing = np.unique(populated[~matches])
            raise ValueError(
                f"Unknown {label} encountered during UIX validation: {missing}"
            )
