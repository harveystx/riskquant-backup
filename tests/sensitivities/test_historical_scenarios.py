# tests/sensitivities/test_historical_scenarios.py
"""
Historical-scenario integration tests.

Covers the seam this increment introduced -- hand-authored observations ->
`RiskFactorMatrix` -> `HistoricalGenerator` -> proxy routing -> shocked
levels -- plus explicit regressions for two production defects found while
wiring it:

  1. `build_risk_factor_matrix_from_df` defaulted to float32, but the
     historical-shock kernel is dtype-pinned to float64 `values`, so the
     canonical builder's default output could not drive the canonical
     historical generator at all.

  2. `_kernel_apply_shocks` carried a single F-contiguous `shocks` signature.
     A trailing batch of exactly one scenario is reported by numpy as both C-
     and F-contiguous and Numba resolves that to 'C', so the strict
     signature failed to match (RQ-AUD-017, consumer side).

The shock mathematics itself is already covered by
`tests/simlab/scenarios/test_historical_math.py` and is not duplicated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_df,
)
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels

_DATES = pd.to_datetime(
    [
        "2026-12-01",
        "2026-12-02",
        "2026-12-03",
        "2026-12-04",
        "2026-12-07",
        "2026-12-08",
        "2026-12-09",
        "2026-12-10",
        "2026-12-11",
        "2026-12-14",
    ]
)
_SOURCE_NAMES = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")
_LEVELS = np.array(
    [
        (1.0700, 196.00, 80.10),
        (1.0750, 198.50, 81.40),
        (1.0720, 197.20, 80.90),
        (1.0810, 201.30, 83.20),
        (1.0780, 200.10, 82.60),
        (1.0835, 203.40, 84.10),
        (1.0795, 199.80, 82.90),
        (1.0760, 198.20, 81.70),
        (1.0825, 202.60, 83.60),
        (1.0800, 200.00, 82.30),
    ],
    dtype=np.float64,
)
_NAME_TO_ID = {"USDEUR": 0, "AAPL_EQ": 1, "BRENT_CRUDE": 2}

# required = [USDEUR, AAPL, BRENT, WTI]; WTI sources BRENT (col 2) x 0.95.
_SOURCE_FACTOR_IDS = np.array([0, 1, 2, 2], dtype=np.int32)
_SHOCK_SCALE = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)
_SHIFT_LOG_RETURN = np.array([1, 1, 1, 1], dtype=np.int8)
_BASE_LEVELS = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(_LEVELS, index=_DATES, columns=list(_SOURCE_NAMES))


def _matrix(**kwargs):
    return build_risk_factor_matrix_from_df(
        df=_frame(), name_to_id_map=_NAME_TO_ID, **kwargs
    )


def _shocked_levels(batch_size: int) -> np.ndarray:
    generator = HistoricalGenerator(matrix=_matrix(), horizon=1)
    chunks: list[np.ndarray] = []
    for batch in generator.iter_batches(
        required_shift_type_id=_SHIFT_LOG_RETURN,
        source_factor_ids=_SOURCE_FACTOR_IDS,
        shock_scale=_SHOCK_SCALE,
        batch_size=batch_size,
    ):
        out = np.empty(
            (_BASE_LEVELS.shape[0], int(batch.shocks.shape[0])),
            dtype=np.float64,
            order="C",
        )
        apply_shocks_to_base_levels(base_levels=_BASE_LEVELS, batch=batch, out=out)
        chunks.append(out)
    return np.concatenate(chunks, axis=1)


# =============================================================================
# Production defect 1: builder float dtype must satisfy the kernel contract
# =============================================================================


def test_builder_default_dtype_drives_the_historical_generator() -> None:
    """
    The canonical builder's DEFAULT output must be usable by the canonical
    historical generator. It previously defaulted to float32 while the
    kernel is pinned to float64, so the default combination raised TypeError.
    """
    matrix = _matrix()
    assert matrix.values.dtype == np.float64
    assert matrix.values.flags["F_CONTIGUOUS"]

    generator = HistoricalGenerator(matrix=matrix, horizon=1)
    assert generator.total_scenarios == _LEVELS.shape[0] - 1

    batches = list(
        generator.iter_batches(
            required_shift_type_id=_SHIFT_LOG_RETURN,
            source_factor_ids=_SOURCE_FACTOR_IDS,
            shock_scale=_SHOCK_SCALE,
            batch_size=16,
        )
    )
    assert batches, "generator must yield at least one batch"


def test_float32_matrix_is_rejected_by_the_kernel_contract() -> None:
    """
    Pins WHY the default is float64: it is the kernel's contract, not a
    preference. An explicitly float32 matrix fails loudly at the generator
    boundary (F5, D017) — clearly and early, not deep inside Numba.
    """
    matrix = _matrix(float_dtype=np.float32)
    assert matrix.values.dtype == np.float32
    with pytest.raises(TypeError, match="must be float64"):
        HistoricalGenerator(matrix=matrix, horizon=1)


def test_canonical_history_fixture_drives_the_generator() -> None:
    """The shipped canonical fixture must also satisfy the kernel contract."""
    from riskquant.ingest_fixtures.market_data.canonical_market_data_history import (
        CANONICAL_HISTORICAL_MARKET_DATA_DF as DF,
    )

    matrix = build_risk_factor_matrix_from_df(
        df=DF, name_to_id_map={c: i for i, c in enumerate(DF.columns)}
    )
    assert matrix.values.dtype == np.float64

    generator = HistoricalGenerator(matrix=matrix, horizon=1)
    factor_count = matrix.factor_ids.shape[0]
    batches = list(
        generator.iter_batches(
            required_shift_type_id=np.ones(factor_count, dtype=np.int8),
            source_factor_ids=matrix.factor_ids.copy(),
            shock_scale=np.ones(factor_count, dtype=np.float32),
            batch_size=64,
        )
    )
    assert batches


# =============================================================================
# Production defect 2: single-row trailing batch layout
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 9, 16])
def test_apply_shocks_accepts_every_batch_geometry(batch_size: int) -> None:
    """
    9 scenarios across assorted batch sizes forces trailing batches of
    several widths, including exactly ONE scenario (batch_size=4 and 8),
    which is the ambiguous C/F-contiguous case.
    """
    levels = _shocked_levels(batch_size)
    assert levels.shape == (4, 9)
    assert np.isfinite(levels).all()


def test_batch_size_does_not_change_historical_scenarios() -> None:
    """
    Unlike the seeded parametric generator, historical scenarios are a
    deterministic function of the observations, so batching is purely a
    streaming detail and must not alter results.
    """
    reference = _shocked_levels(16)
    for batch_size in (1, 2, 4, 5, 8, 9):
        np.testing.assert_allclose(_shocked_levels(batch_size), reference, atol=0.0)


# =============================================================================
# Historical shocks, proxy routing, and economics
# =============================================================================


def test_shock_dtype_axis_and_routing() -> None:
    generator = HistoricalGenerator(matrix=_matrix(), horizon=1)
    batch = next(
        iter(
            generator.iter_batches(
                required_shift_type_id=_SHIFT_LOG_RETURN,
                source_factor_ids=_SOURCE_FACTOR_IDS,
                shock_scale=_SHOCK_SCALE,
                batch_size=16,
            )
        )
    )
    assert batch.shocks.dtype == np.float32
    assert batch.shocks.shape == (9, 3)  # 9 scenarios, 3 OBSERVED factors
    assert batch.required_to_shock_uix.tolist() == [0, 1, 2, 2]
    np.testing.assert_allclose(batch.shock_scale, [1.0, 1.0, 1.0, 0.95], atol=1e-7)


def test_wti_levels_match_an_independent_oracle() -> None:
    """
    Oracle computed straight from the raw observations:
        WTI_level(s) = 70.00 * exp(0.95 * ln(B_{t+1} / B_t))
    No production function is called to build the expectation.
    """
    brent = _LEVELS[:, _SOURCE_NAMES.index("BRENT_CRUDE")]
    brent_log_returns = np.log(brent[1:] / brent[:-1])
    expected = 70.00 * np.exp(0.95 * brent_log_returns)

    levels = _shocked_levels(4)
    np.testing.assert_allclose(levels[3], expected, rtol=1e-6)


def test_observed_factors_match_an_independent_oracle() -> None:
    """Unproxied factors use their own history at unit scale."""
    levels = _shocked_levels(4)
    for required_row, name in enumerate(_SOURCE_NAMES):
        observed = _LEVELS[:, _SOURCE_NAMES.index(name)]
        log_returns = np.log(observed[1:] / observed[:-1])
        expected = _BASE_LEVELS[required_row] * np.exp(log_returns)
        np.testing.assert_allclose(levels[required_row], expected, rtol=1e-6)


def test_wti_keeps_its_own_base_level_not_brents() -> None:
    """Proxying transfers the SHOCK, never the level."""
    levels = _shocked_levels(4)
    assert abs(float(levels[3].mean()) - 70.0) < 2.0
    assert float(np.abs(levels[3] - levels[2]).min()) > 5.0


def test_wti_shock_is_strictly_scaled_by_095() -> None:
    """
    The proxied WTI log return must be exactly 0.95x Brent's -- not 1.0x,
    and not Brent's level.
    """
    levels = _shocked_levels(4)
    wti_log_returns = np.log(levels[3] / 70.00)
    brent_log_returns = np.log(levels[2] / 82.30)
    np.testing.assert_allclose(wti_log_returns, 0.95 * brent_log_returns, rtol=1e-5)
    assert not np.allclose(wti_log_returns, brent_log_returns, rtol=1e-3)


def test_horizon_two_yields_fewer_scenarios() -> None:
    """total_scenarios = T - horizon, per the generator's contract."""
    assert HistoricalGenerator(matrix=_matrix(), horizon=2).total_scenarios == 8
    assert HistoricalGenerator(matrix=_matrix(), horizon=1).total_scenarios == 9


def test_insufficient_history_fails_fast() -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        HistoricalGenerator(matrix=_matrix(), horizon=len(_DATES))
