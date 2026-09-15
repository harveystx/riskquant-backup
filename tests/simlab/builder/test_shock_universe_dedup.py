# tests/simlab/builder/test_shock_universe_dedup.py
"""
R4 — independent property tests for shock-universe deduplication.

The oracle deduplicates `(source_factor_id, shift_type_id)` pairs with plain
Python tuples and a dict — it does NOT reuse the production int64 bit-packing.
Verified: the mapping is exact, order-independent, collision-free across
distinct shift types, and admits the canonical valid ids including NO_SHIFT=-1.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.builder.shock_universe import build_shock_universe


def _build(src, shift, available=None):
    src = np.asarray(src, dtype=np.int32)
    shift = np.asarray(shift, dtype=np.int8)
    if available is None:
        available = np.unique(src).astype(np.int32)
    return build_shock_universe(
        required_shift_type_id=shift,
        source_factor_ids=src,
        available_factor_ids=np.asarray(available, dtype=np.int32),
    )


def _oracle_unique(src, shift):
    """First-seen unique (src, shift) tuples and the required->uix mapping."""
    seen: dict[tuple[int, int], int] = {}
    uix = []
    for s, t in zip(src, shift, strict=True):
        key = (int(s), int(t))
        if key not in seen:
            seen[key] = len(seen)
        uix.append(seen[key])
    return list(seen.keys()), uix


def test_mapping_is_exact() -> None:
    """Each required exposure maps to the unique shock with its own (src,shift)."""
    src = [10, 20, 10, 30, 20, 10]
    shift = [
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_LOG_RETURN_ID,   # dup of first
        SHIFT_DIFFERENCE_ID,
        SHIFT_SIMPLE_RETURN_ID,  # dup of second
        SHIFT_LOG_RETURN_ID,   # dup of first
    ]
    u = _build(src, shift)
    for i in range(len(src)):
        k = u.required_to_shock_uix[i]
        assert int(u.shock_source_ids[k]) == src[i]
        assert int(u.shock_shift_type_ids[k]) == int(shift[i])


def test_dedup_count_matches_distinct_tuples() -> None:
    src = [10, 20, 10, 30, 20, 10]
    shift = [1, 2, 1, 3, 2, 1]
    u = _build(src, shift)
    distinct, _ = _oracle_unique(src, shift)
    assert u.shock_source_ids.shape[0] == len(distinct)
    # every produced unique shock is itself distinct
    produced = list(
        zip(u.shock_source_ids.tolist(), u.shock_shift_type_ids.tolist(), strict=True)
    )
    assert len(set(produced)) == len(produced)


def test_distinct_shift_types_never_collide() -> None:
    """Same source, different shift types -> distinct shocks."""
    src = [7, 7, 7, 7, 7]
    shift = [
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
        SHIFT_ABSOLUTE_BASE_RETURN_ID,
        SHIFT_NO_SHIFT_ID,
    ]
    u = _build(src, shift)
    assert u.shock_source_ids.shape[0] == 5
    pairs = set(
        zip(u.shock_source_ids.tolist(), u.shock_shift_type_ids.tolist(), strict=True)
    )
    assert pairs == {(7, 1), (7, 2), (7, 3), (7, 4), (7, -1)}


def test_no_shift_minus_one_round_trips() -> None:
    """NO_SHIFT = -1 survives the pack/unpack as int8 -1, not 255."""
    src = [5, 5]
    shift = [SHIFT_NO_SHIFT_ID, SHIFT_LOG_RETURN_ID]
    u = _build(src, shift)
    assert (-1) in u.shock_shift_type_ids.tolist()
    assert u.shock_shift_type_ids.dtype == np.int8
    k = u.required_to_shock_uix[0]
    assert int(u.shock_shift_type_ids[k]) == -1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_order_independence(seed: int) -> None:
    """Permuting the required inputs leaves the unique-shock SET unchanged and
    still resolves every required pair correctly."""
    rng = np.random.default_rng(seed)
    sources = rng.integers(1, 6, size=12).astype(np.int32)
    shifts = rng.choice(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_NO_SHIFT_ID],
        size=12,
    ).astype(np.int8)

    u0 = _build(sources, shifts)
    set0 = set(
        zip(u0.shock_source_ids.tolist(), u0.shock_shift_type_ids.tolist(), strict=True)
    )

    perm = rng.permutation(12)
    u1 = _build(sources[perm], shifts[perm])
    set1 = set(
        zip(u1.shock_source_ids.tolist(), u1.shock_shift_type_ids.tolist(), strict=True)
    )

    assert set0 == set1  # order-independent set of unique shocks
    # mapping still exact under the permutation
    for i in range(12):
        k = u1.required_to_shock_uix[i]
        assert int(u1.shock_source_ids[k]) == int(sources[perm][i])
        assert int(u1.shock_shift_type_ids[k]) == int(shifts[perm][i])


@pytest.mark.parametrize(
    "bad_shift", [0, 5, 99, -2]  # 0 and out-of-range are invalid; valid set is {-1,1,2,3,4}
)
def test_invalid_shift_ids_rejected(bad_shift: int) -> None:
    with pytest.raises((ValueError, Exception)):
        _build([1, 2], [SHIFT_LOG_RETURN_ID, bad_shift])


def test_all_canonical_valid_ids_accepted() -> None:
    src = [1, 2, 3, 4, 5]
    shift = [-1, 1, 2, 3, 4]
    u = _build(src, shift)
    assert u.shock_source_ids.shape[0] == 5
    assert sorted(u.shock_shift_type_ids.tolist()) == [-1, 1, 2, 3, 4]
