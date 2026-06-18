# src/riskquant/portfolio/ingest/dispatcher.py

"""
Instrument routing for the ingest edge.

This module is the non-OOP dispatch layer that routes materialized ingest rows
to the correct per-instrument validation logic. Routing is performed with dense
lookup tables keyed by integer ids — no class hierarchy, no method overriding,
no string-keyed dictionary lookups in the dispatch path.

Two routing axes mirror the engine's own dual-dispatch model:

  - FAMILY  (primary)   : SPOT / FORWARD / OPTION / CREDIT  -> coarse validators
  - INSTRUMENT (secondary): the concrete Instrument id      -> refinement validators

The block-compiler routing (instrument id -> SoA block compiler) lives
downstream in `portfolio.compilers.registry`; this module deliberately handles
only the validation hop, the single point in the ingest phase that genuinely
needs per-instrument-type branching. It supersedes the hand-wired sequence
previously inlined in `validate.validate_materialized_ingest_df`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.portfolio.ingest.validate import (
    validate_forward_family,
    validate_option_family,
    validate_spot_family,
    validate_unique_uids,
)
from riskquant.portfolio.instruments import (
    INSTRUMENT_NAME_TO_ID_REGISTRY,
    INSTRUMENT_TYPE_TO_FAMILY_ARRAY,
    NUM_INSTRUMENTS,
    InstrumentTypeFamily,
)

# A masked validator consumes the frame plus a boolean row selector.
ValidatorFn: TypeAlias = Callable[[pd.DataFrame, npt.NDArray[np.bool_]], None]

# ==============================================================================
# DISPATCH TABLES (DENSE, INTEGER-KEYED)
# ==============================================================================

_NUM_FAMILIES: Final[int] = len(InstrumentTypeFamily)

# Family id -> coarse validator. Dense list => O(1) branchless indexing.
FAMILY_VALIDATORS: Final[list[ValidatorFn | None]] = [None] * _NUM_FAMILIES
FAMILY_VALIDATORS[InstrumentTypeFamily.SPOT] = validate_spot_family
FAMILY_VALIDATORS[InstrumentTypeFamily.FORWARD] = validate_forward_family
FAMILY_VALIDATORS[InstrumentTypeFamily.OPTION] = validate_option_family

# Instrument id -> refinement validator. Most slots stay None: only instruments
# carrying constraints beyond their family contract register here.
INSTRUMENT_VALIDATORS: Final[list[ValidatorFn | None]] = [None] * NUM_INSTRUMENTS

# Instrument id -> family id, narrowed to int8 and frozen as an ndarray so the
# id->family hop is a single advanced-indexing op rather than a Python lookup.
_FAMILY_LUT: Final[npt.NDArray[np.int8]] = np.asarray(
    INSTRUMENT_TYPE_TO_FAMILY_ARRAY, dtype=np.int8
)


# ==============================================================================
# INSTRUMENT ID RESOLUTION
# ==============================================================================


def resolve_instrument_ids(df: pd.DataFrame) -> npt.NDArray[np.int32]:
    """
    Map the `instrument` name column to int32 instrument ids.

    The strings die here: every downstream stage routes on the integer id.
    """
    mapped = df["instrument"].map(INSTRUMENT_NAME_TO_ID_REGISTRY)

    invalid = mapped.isna().to_numpy(copy=False)
    if invalid.any():
        names = df["instrument"]
        rows = names.index[invalid].tolist()[:10]
        bad = sorted(names[invalid].dropna().unique().tolist())
        raise ValueError(f"Unknown instrument(s):\n{bad}\nRows: {rows}")

    # Generate the contiguous array
    arr = np.ascontiguousarray(mapped.to_numpy(dtype=np.int32))

    # Cast the return value to satisfy the type checker
    return cast(npt.NDArray[np.int32], arr)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def dispatch_ingest_validation(df: pd.DataFrame) -> npt.NDArray[np.int32]:
    """
    Validate a materialized ingest frame via table-driven routing.

    Runs global checks, then routes every row through its family validator and
    any instrument-level refinement validator. The resolved instrument-id array
    is returned so the downstream compiler can reuse it without re-mapping the
    `instrument` strings.

    Parameters
    ----------
    df
        Canonicalized + materialized ingest frame.

    Returns
    -------
    npt.NDArray[np.int32]
        Per-row instrument ids, aligned to `df`.
    """
    # ---- global invariants -------------------------------------------------
    validate_unique_uids(df)

    # ---- id + family resolution (vectorized, no row loops) -----------------
    instrument_ids = resolve_instrument_ids(df)
    family_ids = _FAMILY_LUT[instrument_ids]

    # ---- primary dispatch: family-level validation -------------------------
    for family_id, validator in enumerate(FAMILY_VALIDATORS):
        if validator is None:
            continue
        mask = family_ids == family_id
        if mask.any():
            validator(df, mask)

    # ---- secondary dispatch: instrument-level refinement -------------------
    for instrument_id, validator in enumerate(INSTRUMENT_VALIDATORS):
        if validator is None:
            continue
        mask = instrument_ids == instrument_id
        if mask.any():
            validator(df, mask)

    return instrument_ids
