# src/riskquant/simlab/scenarios/world_column_stream.py
"""
World-column stream: requested ``(d, tau)`` columns from a scenario generator.

D088 §4: a world supplies jointly defined random variables keyed
``(d, tau, h)``. The generators already compute one column per requested
``(source, chart)`` pair (a level-backed ``T_tau`` column for the matrix
engines, a declared direct column for the parametric engine) at the world's
horizon. This module is the one place the D088 seam calls them.

Legacy boundary
---------------
The ``ScenarioGenerator`` protocol still carries the D013 per-row payload
``shock_scale``. Every generator passes that array through to
``ScenarioSetBatch`` unchanged and never reads it when computing shocks, so the
values of ``shocks`` do not depend on it. The protocol requires the argument,
so a neutral array of ones is passed and ``batch.shock_scale`` is never read
here. This adapter performs no proxy weighting, no source selection and no
chart substitution: ``source_factor_ids`` receives the requested drivers ``d``
and ``required_shift_type_id`` receives the requested driver charts ``tau``,
which is what those arguments name in the generators (the chart of the column
to compute), not a target chart.

The request must be unique ``(d, tau)`` pairs. The generators deduplicate
first-seen, so the columns then come back in request order; that is checked
once, and a violation raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.simlab.scenarios.base import ScenarioGenerator

__all__ = ["WorldColumnBatch", "iter_world_column_batches"]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldColumnBatch:
    """
    One batch of world columns.

    ``shocks`` is ``(B, K)`` float32, F-contiguous; column ``k`` is the world
    variable of request position ``k``. ``scenario_offset`` is the index of the
    batch's first scenario in the world's common scenario index.
    """

    shocks: npt.NDArray[np.float32]
    scenario_offset: int


def iter_world_column_batches(
    generator: ScenarioGenerator,
    *,
    driver_rf_ids: npt.NDArray[np.int32],
    chart_ids: npt.NDArray[np.int8],
    batch_size: int,
) -> Iterator[WorldColumnBatch]:
    """
    Stream the requested world columns, in request order.

    Parameters
    ----------
    generator
        A built scenario generator (the world).
    driver_rf_ids, chart_ids
        ``(K,)`` int32 and int8: unique ``(d, tau)`` pairs, the world request.
    batch_size
        Operational only (D041).

    Raises
    ------
    TypeError
        Wrong array type, dtype or rank.
    ValueError
        Length mismatch, a duplicate ``(d, tau)`` pair, or a generator whose
        columns do not follow request order.
    """
    for name, arr, dtype in (
        ("driver_rf_ids", driver_rf_ids, np.int32),
        ("chart_ids", chart_ids, np.int8),
    ):
        if not isinstance(arr, np.ndarray) or arr.dtype != dtype or arr.ndim != 1:
            raise TypeError(f"{name} must be a 1-D {np.dtype(dtype)} ndarray.")
    k = driver_rf_ids.shape[0]
    if chart_ids.shape[0] != k:
        raise ValueError("driver_rf_ids and chart_ids must have one length.")
    pairs = np.stack([driver_rf_ids, chart_ids.astype(np.int32)], axis=1)
    if k and np.unique(pairs, axis=0).shape[0] != k:
        raise ValueError("the world request must list each (d, tau) pair once.")

    # Protocol-required D013 payload; passed through by every generator and
    # never read here (see module docstring).
    neutral_payload = np.ones(k, dtype=np.float32)
    expected = np.arange(k, dtype=np.int32)
    checked = False
    for batch in generator.iter_batches(
        required_shift_type_id=np.ascontiguousarray(chart_ids),
        source_factor_ids=np.ascontiguousarray(driver_rf_ids),
        shock_scale=neutral_payload,
        batch_size=batch_size,
    ):
        if not checked:
            if not np.array_equal(batch.required_to_shock_uix, expected):
                raise ValueError(
                    "the generator did not return the world columns in request "
                    "order; binding by position would be unsafe."
                )
            checked = True
        yield WorldColumnBatch(
            shocks=batch.shocks, scenario_offset=int(batch.scenario_offset)
        )
