# tests/simlab/scenarios/test_historical_scenarios.py
"""
Historical-scenario integration tests.

Covers the seam this increment introduced -- hand-authored observations ->
`RiskFactorMatrix` -> `HistoricalGenerator` -> Proxy V1 operator (identity
rows plus one authored term) -> shocked levels -- plus explicit regressions
for two production defects found while wiring it:

  1. `build_risk_factor_matrix_from_df` defaulted to float32, but the
     historical-shock kernel is dtype-pinned to float64 `values`, so the
     canonical builder's default output could not drive the canonical
     historical generator at all.

  2. The shock-application kernel carried a single F-contiguous `shocks`
     signature. A trailing batch of exactly one scenario is reported by numpy
     as both C- and F-contiguous and Numba resolves that to 'C', so the strict
     signature failed to match (RQ-AUD-017, consumer side). The Slice 2
     kernel registers both layouts; the one-scenario batch test keeps the
     regression.

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
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels

_DATES = pd.to_datetime([
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
])
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
# History frames are resolved and unit-normalised through the static config
# (D069 / AUD-06); the shared spot-universe config places USDEUR, AAPL_EQ and
# BRENT_CRUDE at rf_ids 0, 1, 2, all DECIMAL, so the ids and levels below are
# unchanged.
_RSC = None


@pytest.fixture(autouse=True)
def _bind_static_config(rsc):
    global _RSC
    _RSC = rsc
    assert [rsc.rel_name_to_rf_id[n] for n in _SOURCE_NAMES] == [0, 1, 2]


# required = [USDEUR, AAPL, BRENT, WTI]; WTI sources BRENT (col 2) x 0.95.
# World request: the three observed factors, LOG_RETURN columns.
_WORLD_DRIVERS = np.array([0, 1, 2], dtype=np.int32)
_WORLD_CHARTS = np.array([1, 1, 1], dtype=np.int8)
# Proxy V1 operator over the four rows: identity rows for factors 0..2 and
# WTI (row 3) <- 0.95 * Y(factor 2, LOG_RETURN) (D088).
_ROW_OFFSETS = np.array([0, 1, 2, 3, 4], dtype=np.int32)
_TERM_COLUMNS = np.array([0, 1, 2, 2], dtype=np.int32)
_TERM_LOADINGS = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float64)
_SHIFT_LOG_RETURN = np.array([1, 1, 1, 1], dtype=np.int8)
_BASE_LEVELS = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(_LEVELS, index=_DATES, columns=list(_SOURCE_NAMES))


def _matrix(**kwargs):
    return build_risk_factor_matrix_from_df(df=_frame(), static_config=_RSC, **kwargs)


def _shocked_levels(batch_size: int) -> np.ndarray:
    generator = HistoricalGenerator(matrix=_matrix(), horizon=1)
    chunks: list[np.ndarray] = []
    for batch in generator.iter_batches(
        chart_ids=_WORLD_CHARTS,
        driver_rf_ids=_WORLD_DRIVERS,
        batch_size=batch_size,
    ):
        out = np.empty(
            (_BASE_LEVELS.shape[0], int(batch.shocks.shape[0])),
            dtype=np.float64,
            order="C",
        )
        apply_linear_terms_to_base_levels(
            base_levels=_BASE_LEVELS,
            row_term_offsets=_ROW_OFFSETS,
            term_column_uix=_TERM_COLUMNS,
            term_loadings=_TERM_LOADINGS,
            target_shift_types=_SHIFT_LOG_RETURN,
            shocks=batch.shocks,
            out=out,
        )
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
            chart_ids=_WORLD_CHARTS,
            driver_rf_ids=_WORLD_DRIVERS,
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


# -----------------------------------------------------------------------------
# Canonical history fixture through the real D076 boundary (AUD06-F1)
# -----------------------------------------------------------------------------

# Independent unit table for the oracle below (config `unit_scale` strings).
_UNIT_TO_BASE = {"DECIMAL": 1.0, "PERCENT": 0.01, "BP": 1e-4}
_SURFACE_Y_AXIS_KINDS = ("STRIKE", "DELTA_CALL", "LOG_MONEYNESS_FWD")


@pytest.fixture(scope="module")
def canonical_static_config():
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_dict_of_dfs,
    )
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
    )

    return compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


@pytest.fixture(scope="module")
def canonical_history_df() -> pd.DataFrame:
    from riskquant.ingest_fixtures.market_data.canonical_market_data_history import (
        CANONICAL_HISTORICAL_MARKET_DATA_DF,
    )

    return CANONICAL_HISTORICAL_MARKET_DATA_DF


def _configured_factor_oracle(cfg) -> dict[str, tuple[int, float, str | None]]:
    """
    Identity oracle built from the metadata OBJECTS, not from the fixture and
    not from `rel_name_to_rf_id`: canonical relative name -> (intended rf_id,
    unit scale, surface y_axis_kind or None). Surface names carry the bare
    `y_label`; the y-axis kind is metadata, never part of the name.
    """
    oracle: dict[str, tuple[int, float, str | None]] = {}
    for node in cfg.node_meta_dict.values():
        scale = _UNIT_TO_BASE[node.unit_scale]
        oracle[node.node_name] = (int(node.node_id), scale, None)
    for pt in cfg.curve_point_meta_dict.values():
        cm = cfg.curve_meta_dict[pt.curve_id]
        name = f"{cm.curve_name}{pt.tenor_token}"
        oracle[name] = (int(pt.rf_id), _UNIT_TO_BASE[cm.unit_scale], None)
    for pt in cfg.surface_point_meta_dict.values():
        sm = cfg.surface_meta_dict[pt.surface_id]
        name = f"{sm.surface_name}{pt.x_tenor_token}[{pt.y_label}]"
        oracle[name] = (int(pt.rf_id), _UNIT_TO_BASE[sm.unit_scale], sm.y_axis_kind)
    return oracle


def test_canonical_history_fixture_drives_the_generator(
    canonical_static_config, canonical_history_df
) -> None:
    """The FULL canonical fixture passes the real D076 builder and the kernel."""
    from riskquant.ingest_fixtures.constants import CANONICAL_HISTORY_BUSINESS_DAYS
    from riskquant.ingest_fixtures.market_data.canonical_market_data_history import (
        CANONICAL_HISTORICAL_FACTOR_COUNT,
    )

    matrix = build_risk_factor_matrix_from_df(
        df=canonical_history_df, static_config=canonical_static_config
    )
    assert matrix.values.shape == (
        CANONICAL_HISTORY_BUSINESS_DAYS,
        CANONICAL_HISTORICAL_FACTOR_COUNT,
    )
    assert matrix.values.dtype == np.float64

    generator = HistoricalGenerator(matrix=matrix, horizon=1)
    factor_count = matrix.factor_ids.shape[0]
    batches = list(
        generator.iter_batches(
            chart_ids=np.ones(factor_count, dtype=np.int8),
            driver_rf_ids=matrix.factor_ids.copy(),
            batch_size=64,
        )
    )
    assert batches


def test_canonical_history_columns_are_exactly_the_configured_relative_names(
    canonical_static_config, canonical_history_df
) -> None:
    columns = [str(c) for c in canonical_history_df.columns]
    oracle = _configured_factor_oracle(canonical_static_config)

    assert len(columns) == len(set(columns))  # no duplicate names
    name_table = canonical_static_config.rel_name_to_rf_id
    unresolved = [c for c in columns if c not in name_table]
    assert unresolved == []  # the core AUD06-F1 regression assertion
    assert set(columns) == set(oracle)  # exactly the configured universe
    assert len(oracle) == len(name_table)
    for name in columns:
        assert name_table[name] == oracle[name][0], name


def test_canonical_history_maps_every_column_to_its_intended_rf_id_once_scaled(
    canonical_static_config, canonical_history_df
) -> None:
    """
    Identity by VALUE, not position: the matrix column holding each intended
    rf_id equals that name's own source column times its unit scale, applied
    exactly once. A permutation, alias or double scaling fails here.

    The fixture's column order coincides with ascending rf_id, so a positional
    id assignment would look correct on it; the check therefore also runs on a
    seeded column permutation, where only name resolution can pass.
    """
    oracle = _configured_factor_oracle(canonical_static_config)
    expected_ids = sorted(rf_id for rf_id, _, _ in oracle.values())
    order = np.random.default_rng(7).permutation(canonical_history_df.shape[1])

    for frame in (canonical_history_df, canonical_history_df.iloc[:, order]):
        matrix = build_risk_factor_matrix_from_df(
            df=frame, static_config=canonical_static_config
        )
        assert matrix.factor_ids.tolist() == expected_ids
        assert len(set(matrix.factor_ids.tolist())) == len(expected_ids)

        column_of = {int(rf): j for j, rf in enumerate(matrix.factor_ids.tolist())}
        non_unit_scales = 0
        for name, (rf_id, scale, _) in oracle.items():
            source = frame[name].to_numpy(dtype=np.float64)
            np.testing.assert_array_equal(
                matrix.values[:, column_of[rf_id]], source * scale
            )
            non_unit_scales += scale != 1.0
        assert non_unit_scales > 0  # the once-only check is not vacuous


def test_canonical_history_identity_is_independent_of_column_order(
    canonical_static_config, canonical_history_df
) -> None:
    rng = np.random.default_rng(20260917)
    order = rng.permutation(canonical_history_df.shape[1])
    shuffled = canonical_history_df.iloc[:, order]
    assert list(shuffled.columns) != list(canonical_history_df.columns)

    base = build_risk_factor_matrix_from_df(
        df=canonical_history_df, static_config=canonical_static_config
    )
    moved = build_risk_factor_matrix_from_df(
        df=shuffled, static_config=canonical_static_config
    )
    np.testing.assert_array_equal(moved.factor_ids, base.factor_ids)
    np.testing.assert_array_equal(moved.values, base.values)


def test_canonical_history_surface_names_use_the_bare_y_label(
    canonical_static_config, canonical_history_df
) -> None:
    columns = set(map(str, canonical_history_df.columns))
    oracle = _configured_factor_oracle(canonical_static_config)
    surfaces = canonical_static_config.surface_meta_dict.values()
    kinds = {sm.y_axis_kind for sm in surfaces}
    assert kinds == set(_SURFACE_Y_AXIS_KINDS)  # all three naming families covered

    for pt in canonical_static_config.surface_point_meta_dict.values():
        sm = canonical_static_config.surface_meta_dict[pt.surface_id]
        canonical = f"{sm.surface_name}{pt.x_tenor_token}[{pt.y_label}]"
        obsolete = f"{sm.surface_name}{pt.x_tenor_token}[{sm.y_axis_kind}={pt.y_label}]"
        assert canonical in columns and oracle[canonical][2] == sm.y_axis_kind
        assert obsolete not in columns
    assert not [c for c in columns for k in kinds if f"[{k}=" in c]

    for canonical, obsolete in (
        ("BRENT_CRUDE_VOL_SURF[M+0][50.00]", "BRENT_CRUDE_VOL_SURF[M+0][STRIKE=50.00]"),
        ("USDEUR_VOL_SURF[1W][0.10]", "USDEUR_VOL_SURF[1W][DELTA_CALL=0.10]"),
        ("AAPL_VOL_SURF[1M][-0.20]", "AAPL_VOL_SURF[1M][LOG_MONEYNESS_FWD=-0.20]"),
    ):
        assert canonical in columns
        assert obsolete not in columns


# =============================================================================
# Production defect 2: single-row trailing batch layout
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 9, 16])
def test_shock_application_accepts_every_batch_geometry(batch_size: int) -> None:
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
# Historical shocks, the Proxy V1 operator, and economics
# =============================================================================


def test_shock_dtype_and_world_column_axis() -> None:
    generator = HistoricalGenerator(matrix=_matrix(), horizon=1)
    batch = next(
        iter(
            generator.iter_batches(
                chart_ids=_WORLD_CHARTS,
                driver_rf_ids=_WORLD_DRIVERS,
                batch_size=16,
            )
        )
    )
    assert batch.shocks.dtype == np.float32
    assert batch.shocks.shape == (9, 3)  # 9 scenarios, 3 OBSERVED factors


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
